"""
加载模型，并提供语音识别能力
"""
import threading
import time
import tempfile
import os
import numpy
import torch
import subprocess
from funasr import AutoModel

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


class Model:
    """
    在进程初始化时就加载到GPU中
    防止某些设备饥饿
    """
    def __init__(self, device, model):
        self._cache = {}
        self._lock = threading.Lock()
        self._model = AutoModel(model=model, device=device, disable_update=True)

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
        res = self._model.generate(data)
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

    def load_model(self) -> None:
        """
        根据指定的GPU加载模型，给每个模型上锁，防止多个任务调同一个模型导致报错
        :return:
        """
        devices = list(range(torch.cuda.device_count()))
        self.zh_model.load_model(asr_config.zh_model, devices)
        self.stream_model.load_model(asr_config.zh_stream_model, devices)

    def recognize(self, stream: bool):
        if stream:
            return self.stream_model.get_model()
        return self.zh_model.get_model()


asr_model = ParaModelASR()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test batch_recognition_with_vad function")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--device", default="cpu", help="Device to run the model on (e.g., cpu, cuda:0)")
    parser.add_argument("--model-path", default=asr_config.zh_model, help="Path to the ASR model")
    args = parser.parse_args()
    # 初始化模型
    model = Model(args.device, args.model_path)
