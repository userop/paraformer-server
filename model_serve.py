"""
加载模型，并提供语音识别能力
"""
import threading
import time
import tempfile
import os
import numpy
import subprocess
from funasr import AutoModel
import webrtcvad
from collections import deque
import math
import numpy as np
from typing import List, Tuple, Optional

from config import asr_config


def convert_audio_to_pcm(data: bytes, input_format: str = None, sr: int = 16000, ac: int = 1) -> bytes:
    """
    自动识别音频格式，将任意音频 bytes 转换为 pcm_s16le 格式（返回字节流）。
    - 可自动区分可流式与需临时文件处理的格式。
    - 跨平台（Windows/Linux/macOS）。
    - 已经是PCM16 不做处理
    """

    # --- 可流式与需文件格式分类 ---
    streamable_formats = {"wav", "flac", "mp3", "ogg", "aac", "opus"}
    non_streamable_formats = {"mp4", "m4a", "3gp", "mov"}

    # --- 尝试自动识别输入格式 ---
    if not input_format:
        # 根据常见文件头特征检测
        if data.startswith(b"RIFF"):
            input_format = "wav"
        elif data[4:8] == b"ftyp":
            input_format = "mp4"  # m4a/3gp 都是 MP4 容器
        elif data.startswith(b"ID3"):
            input_format = "mp3"
        elif data[:4] in (b"fLaC", b"OggS"):
            input_format = "flac" if data[:4] == b"fLaC" else "ogg"
        elif len(data) % 2 == 0 and all(abs(b-128) < 128 for b in data[:100]):
            # PCM 裸流
            input_format = "s16le"
        else:
            # 默认尝试 mp3
            input_format = "mp3"

    if input_format.lower() in {"s16le", "pcm_s16le"}:
        return data
    cleanup_paths = []
    try:
        if input_format.lower() in non_streamable_formats:
            # 🧱 不可流式格式：写入临时文件
            tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}")
            tmp_in.write(data)
            tmp_in.close()
            tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_out.close()
            cleanup_paths = [tmp_in.name, tmp_out.name]

            cmd = [
                "ffmpeg", "-y", "-i", tmp_in.name,
                "-ac", str(ac), "-ar", str(sr),
                "-acodec", "pcm_s16le", tmp_out.name,
                "-loglevel", "error"
            ]
            proc = subprocess.run(cmd, capture_output=True)

            if proc.returncode != 0 or not os.path.exists(tmp_out.name):
                raise RuntimeError(f"ffmpeg conversion failed ({input_format})\n{proc.stderr.decode()}")

            # 等待句柄释放（Windows可能锁文件）
            for _ in range(10):
                try:
                    with open(tmp_out.name, "rb") as f:
                        pcm_data = f.read()
                    break
                except PermissionError:
                    time.sleep(0.05)
            else:
                raise PermissionError(f"File locked after ffmpeg conversion: {tmp_out.name}")
        else:
            # ✅ 可流式格式：用 stdin/stdout 管道
            cmd = [
                "ffmpeg", "-f", input_format, "-i", "pipe:0",
                "-ac", str(ac), "-ar", str(sr),
                "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
                "-loglevel", "error"
            ]
            proc = subprocess.run(cmd, input=data, capture_output=True)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg stream conversion failed ({input_format})\n{proc.stderr.decode()}")
            pcm_data = proc.stdout

        if not pcm_data:
            raise RuntimeError(f"No PCM data produced for {input_format}")
        return pcm_data
    finally:
        # --- 清理临时文件 ---
        for path in cleanup_paths:
            for _ in range(10):
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                    break
                except PermissionError:
                    time.sleep(0.05)
            else:
                print(f"⚠️ Warning: could not remove temporary file {path}")

def binary_vad_segments(
    data: bytes,
    input_format: Optional[str] = None,
    sr: int = 16000,
    frame_ms: int = 20,
    vad_mode: int = 3,
    padding_frames: int = 8,
    start_prop: float = 0.6,
    end_prop: float = 0.9,
    return_pcm: bool = True,
    feed_chunk_frames: int = 50,
    se_model=None
) -> List[Tuple[bytes, float, float]]:
    """
    使用 VADStreamProcessor（流式）对一次性二进制音频做 VAD 分段。
    行为与原 binary_vad_segments 保持兼容（返回 pcm bytes + start/end 秒）。
    feed_chunk_frames: 每次传入 accept 的帧数（默认 50 帧），可根据内存/延迟调整。
    """
    # 1) 转换为 PCM int16 mono bytes
    pcm_bytes = convert_audio_to_pcm(data, input_format=input_format, sr=sr, ac=1)
    pcm_bytes = se_model.enhance(pcm_bytes)

    # 2) 创建流式 VAD 处理器（参数与函数参数一致）
    vad_proc = VADStreamProcessor(sample_rate=sr, frame_ms=frame_ms, vad_mode=vad_mode,
                                  padding_frames=padding_frames, start_prop=start_prop, end_prop=end_prop)

    # 3) 分块 feed 给 accept（accept 会处理 leftover）
    frame_bytes = vad_proc.frame_bytes
    chunk_size = frame_bytes * max(1, int(feed_chunk_frames))
    segments = []

    offset = 0
    data_len = len(pcm_bytes)
    while offset < data_len:
        end = min(offset + chunk_size, data_len)
        chunk = pcm_bytes[offset:end]
        out = vad_proc.accept(chunk, is_final=False)
        if out:
            # out 是 list of (segment_bytes, start_time, end_time)
            segments.extend(out)
        offset = end

    # 4) flush 剩余（流结束）
    final_out = vad_proc.accept(b"", is_final=True)
    if final_out:
        segments.extend(final_out)

    # 5) 返回（保留原行为）
    if not return_pcm:
        return []
    return segments



class VADStreamProcessor:
    """
    流式 webrtcvad 封装：
    - 接收原始 PCM bytes (int16 little-endian, 16kHz 单声道)
    - 按帧判断 speech，并输出已合并的 voiced-segment bytes
    使用方法：对每个流片段调用 .accept(raw_bytes, is_final=False)
    当返回非 None 时，得到一个已准备好送 ASR 的 segment
    """
    def __init__(self, sample_rate=16000, frame_ms=20, vad_mode=3, padding_frames=8, start_prop=0.6, end_prop=0.9):
        assert frame_ms in (10,20,30)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_bytes = int(sample_rate * (frame_ms/1000.0) * 2)  # 2 bytes per sample int16
        self.padding_frames = padding_frames  # hang-over frames for end detection
        self.start_prop = start_prop
        self.end_prop = end_prop

        # internal buffers
        self._ring = deque(maxlen=padding_frames)
        self._triggered = False
        self._voiced_frames = []
        self._timestamp = 0.0
        self._frame_duration = frame_ms / 1000.0
        self._leftover = b""  # bytes leftover smaller than frame_bytes

    def _frame_iter(self, data: bytes):
        """
        把 data 分帧，返回 iterator of (timestamp, frame)
        """
        data = self._leftover + data
        offset = 0
        while offset + self.frame_bytes <= len(data):
            chunk = data[offset: offset + self.frame_bytes]
            frame_timestamp = self._timestamp
            self._timestamp += self._frame_duration
            yield frame_timestamp, chunk
            offset += self.frame_bytes
        # leftover keep for next call
        self._leftover = data[offset:]

    def accept(self, data: bytes, is_final: bool = False):
        """
        传入原始 bytes（可以是任意长度），返回列表 of (segment_bytes, start_time, end_time)
        行为修改：只要本次调用中有声音帧（timestamp >= 本次调用开始时刻），就会把本次调用中出现的声音合成一个片段返回。
        """
        out = []
        # 记录本次调用开始时的时间戳，用于区分本次调用中新加入的帧
        call_start_ts = self._timestamp
        for ts, frame in self._frame_iter(data):
            # 判断当前帧是否为语音
            try:
                is_speech = self.vad.is_speech(frame, sample_rate=self.sample_rate)
            except Exception:
                # 如果 frame 长度不对或 webrtcvad 抛错，视为静音
                is_speech = False

            if not self._triggered:
                # 未触发：维护 ring，可能会触发 start
                self._ring.append((ts, frame, is_speech))
                num_voiced = len([1 for _, _, s in self._ring if s])
                if num_voiced > int(self.start_prop * self._ring.maxlen):
                    # start speech: flush ring 到 voiced_frames（保持时间戳）
                    self._triggered = True
                    for r in self._ring:
                        self._voiced_frames.append((r[0], r[1]))
                    self._ring.clear()
            else:
                # 已在语音段中：继续追加到 voiced_frames 并维护 ring（用于 end 判断）
                self._voiced_frames.append((ts, frame))
                self._ring.append((ts, frame, is_speech))
                num_unvoiced = len([1 for _, _, s in self._ring if not s])
                if num_unvoiced > int(self.end_prop * self._ring.maxlen):
                    self._triggered = False
                    self._voiced_frames = []
                    self._ring.clear()

        # is_final flush （保留原行为）
        if is_final and self._voiced_frames:
            start_time = self._voiced_frames[0][0]
            end_time = self._voiced_frames[-1][0] + self._frame_duration
            segment = b"".join([f for _, f in self._voiced_frames])
            out.append((segment, start_time, end_time))
            self._triggered = False
            self._voiced_frames = []
            self._ring.clear()
            return out

        # 新增逻辑：如果本次调用期间（timestamp >= call_start_ts）有新增的 voiced_frames，
        # 将本次调用新增部分合成一个即时返回段（用于低延迟实时识别）。
        if self._voiced_frames:
            # 找出本次调用中 timestamp >= call_start_ts 的帧
            new_frames = [(t, f) for (t, f) in self._voiced_frames if t >= call_start_ts]
            if new_frames:
                start_time = new_frames[0][0]
                end_time = new_frames[-1][0] + self._frame_duration
                segment = b"".join([f for _, f in new_frames])
                # 注意：这里我们不清除 _voiced_frames 中的内容（保持流连续性），
                # 因此后续调用仍会保留上下文；但同时本次调用会即时返回这一段供 ASR 使用。
                out.append((segment, start_time, end_time))

        return out


class Model:
    """
    在进程初始化时就加载到GPU中
    防止某些设备饥饿
    """
    def __init__(self, device, model):
        self._cache = {}
        self._lock = threading.Lock()
        self._model = AutoModel(model=model, device=device, disable_update=True)
        # 每个模型实例维护一个 VAD 处理器（参数可调整）
        self._vad_proc = VADStreamProcessor(sample_rate=16000, frame_ms=20, vad_mode=3, padding_frames=8)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache.clear()
        self._lock.release()
        return False

    def __enter__(self):
        self._lock.acquire(blocking=False)
        return self

    def lock(self):
        return self._lock.acquire(blocking=False)

    def audio_recognition(self, data: bytes, input_format: str = None) -> str:
        """
        识别二进制流
        :param data:  音频二进制流
        :param input_format: 音频格式
        :return:
        """
        _bytes = convert_audio_to_pcm(data, input_format)
        res = self._model.generate(_bytes)
        return res[0]["text"]

    def audio_stream_recognition(self, data: bytes, is_final: bool = False) -> str:
        """
        模型要求输入 numpy float32， [-1.0, 1.0]之间的一维数组。
        bytearray -> int16 -> float32 是为了确保采样点数正确
        32768.0 则是int16的最大值
        使用float32是深度学习框架tensorflow的输入要求
        :param data:
        :param is_final:
        :return:
        """
        np_asr_float32 = numpy.frombuffer(data, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        res = self._model.generate(input=np_asr_float32, cache=self._cache, is_final=is_final,
                                   chunk_size=[0, asr_config.n_chunk_frame, asr_config.n_chunk_feature],
                             encoder_chunk_look_back=asr_config.encoder_chunk_size,
                             decoder_chunk_look_back=asr_config.decoder_chunk_size)
        return res[0]["text"]

# 流式接收带 VAD 的接口
    def audio_stream_recognition_with_vad(self, data: bytes, is_final: bool = False, enforce_float=True):
        """
        data: 原始 PCM int16 bytes（16k, 16-bit, mono）
        is_final: 标记当前 chunk 是流的结束
        返回：list of result texts（可能为空）
        说明：
          调用vad获取到有人声的语音部分再调用模型进行文字识别
        """
        prob,data = self._rn.denoise_chunk(audio_data)
        results = []
        # 获取由 VAD 输出的完整 segment 列表
        segments = self._vad_proc.accept(data, is_final=is_final)
        for seg_bytes, start_t, end_t in segments:
            # seg_bytes 是 int16 PCM bytes；转换为 float32 [-1,1]
            np_asr_float32 = numpy.frombuffer(seg_bytes, dtype=numpy.int16).astype(numpy.float32) / 32768.0
            # 调用现有 generate（保留 cache 与 chunk 参数）
            res = self._model.generate(input=np_asr_float32, cache=self._cache, is_final=is_final,
                                       chunk_size=[0, asr_config.n_chunk_frame, asr_config.n_chunk_feature],
                                       encoder_chunk_look_back=asr_config.encoder_chunk_size,
                                       decoder_chunk_look_back=asr_config.decoder_chunk_size)
            # 可能一次生成多个 hypothesis，依你现有用法取第一个
            if res and isinstance(res, list):
                results.append(res[0].get("text", ""))
            else:
                results.append("")
        return results

# 二进制文件调用vad优化版
    def batch_recognition_with_vad(self, data: bytes,
                                   input_format: Optional[str] = None,
                                   run_asr_per_segment: bool = True,
                                   sr: int = 16000,
                                   vad_kwargs: dict = None):
        """
            data: 原始 二进制数据
            返回：list of result texts（可能为空）
            说明：
              将语音文件转化成标准格式，调用vad获取到有人声的语音部分再调用模型进行文字识别
        """
        vad_kwargs = vad_kwargs or {}
        # 确保把 sr 透传给 binary_vad_segments
        segments = binary_vad_segments(data, input_format=input_format, sr=sr, se_model =self.se_model, **vad_kwargs)
        results = []
        for seg_bytes, s, e in segments:
            text = ""
            if run_asr_per_segment:
                try:
                    np_float = numpy.frombuffer(seg_bytes, dtype=numpy.int16).astype(numpy.float32) / 32768.0
                    res = self._model.generate(input=np_float, cache=self._cache, is_final=True,
                                               chunk_size=[0, asr_config.n_chunk_frame, asr_config.n_chunk_feature],
                                               encoder_chunk_look_back=asr_config.encoder_chunk_size,
                                               decoder_chunk_look_back=asr_config.decoder_chunk_size)
                    if res and isinstance(res, list):
                        text = res[0].get("text", "")
                except Exception as exc:
                    print("batch_recognition_with_vad ASR error:", exc)
            results.append((text, s, e))
        return results


class ModelList:
    def __init__(self, ):
        self._models = []
        self._idx = 0
        self.n_gpu = 0

    def load_model(self, model, devices):
        for device in devices:
            self._models.append(Model(device, model))
        self.n_gpu = len(devices)

    def get_model(self, timeout: float = 5):
        def _available_gpu():
            for i in range(self.n_gpu):
                idx = (self._idx + i) % self.n_gpu
                model = self._models[idx]
                # 以非阻塞的方式获取锁，占用GPU；在其它地方释放锁
                if model.lock():
                    return model
            return None  # 所有服务都忙
        _model = _available_gpu()
        if _model is None:  # 5s之内查询500次
            t1 = time.time()
            while time.time() - t1 < timeout:
                _model = _available_gpu()
                if _model is not None:
                    return _model
                time.sleep(0.01)
            if _model is None:
                raise RuntimeError(f"All ASR models are busy, please try again later.")
        return _model


class ParaModelASR:
    """
        AutoModel本身不具有并发能力，每个GPU加载并绑定一个模型，调用时通过路由分发。
        使用uvicorn运行多个程序就能拓展并发数

    """
    def __init__(self):
        self.zh_model: ModelList = ModelList()
        self.stream_model: ModelList = ModelList()

    def load_model(self, devices: str) -> None:
        """
        根据指定的GPU加载模型，给每个模型上锁，防止多个任务调同一个模型导致报错
        :return:
        """
        devices = [int(dev) for dev in devices.split(",")]
        self.zh_model.load_model(asr_config.zh_model, devices)
        self.stream_model.load_model(asr_config.zh_stream_model, devices)

    def recognize(self, stream: bool):
        if stream:
            return self.stream_model.get_model()
        return self.zh_model.get_model()


asr_model = ParaModelASR()

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Test batch_recognition_with_vad function")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (e.g., cpu, cuda:0)")
    parser.add_argument("--model-path", default=asr_config.zh_model, help="Path to the ASR model")

    args = parser.parse_args()

    # 初始化模型
    model = Model(args.device, args.model_path)

    # 读取音频文件
    with open(args.audio_file, "rb") as f:
        audio_data = f.read()

    # 调用 batch_recognition_with_vad 进行识别
    results = model.batch_recognition_with_vad(audio_data)

    # 打印结果
    for i, (text, start, end) in enumerate(results):
        print(f"Segment {i+1}: [{start:.2f}s - {end:.2f}s] {text}")

