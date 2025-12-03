import os
import subprocess
import tempfile
import time

import webrtcvad


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

class AudioVAD:
    """
    声音活动检测
      前端会将静音内容通过websocket传递到后端 这是后台要做VAD的主要原因

      使用 webrtcvad
      内部做了很多对声音流的处理算子
      核心接口就一个，判断是否有人说话
      数据长度要求 10ms / 20ms / 30ms，其它长度 没效果

    """
    def __init__(self, vad_mode: int = 2):
        self.vad = webrtcvad.Vad(vad_mode)

    def stream_vad(self, audio: bytes, frame_ms: int = 30) -> bool:
        """
        传过来的二进制流单长度基本在合理范围内
        也就是audio片段不会过长也不会短  30ms 16k/Hz
        判断这个片段中是否有人说话

        :return:
        """
        # 二进制位数  位数超过bit直接切片 audio[:bits]，不足的补0即可
        bits = int(16000 * frame_ms * 1e-3 * 2)
        # 判断位数不足bits就补0
        if len(audio) < bits:
            audio = audio + (bits - len(audio)) * b'\x00'
        else:
            audio = audio[:bits]
        try:
            is_speech = self.vad.is_speech(audio, sample_rate=16000)
        except Exception:
            # 如果 frame 长度不对或 webrtcvad 抛错，视为静音
            is_speech = False
        return is_speech

    def split2join(self,
                   audio: bytes,
                   frame_ms: int = 30,
                   input_format=None) -> bytes:
        """
        将音频按照长度时间长度切片
        将识别出来有人说话的片段拼接到一起合并得到新的音频流
        :return:

        Args:
            sr:
        """
        # 根据bits长度将audio切片后处理
        bits = int(16000 * frame_ms * 1e-3 * 2)

        # 1) 转换为 PCM int16 mono bytes
        pcm_bytes = convert_audio_to_pcm(audio, input_format=input_format)

        # 2) 切片并获取分段合成无静音片段
        offset = 0
        data_len = len(pcm_bytes)
        out = b""  # 创建与pcm_bytes相同类型的bytes变量
        while offset < data_len:
            end = min(offset + bits, data_len)
            chunk = pcm_bytes[offset:end]
            is_speech = self.stream_vad(chunk)
            if is_speech:
                # 如果is_speech为真，则将这个片段拼接到一起
                out += chunk
            offset = end

        return  out
