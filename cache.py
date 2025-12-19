"""
  缓存websocket二进制流 并且要求根据时间戳排序
"""
import bisect
import struct
from typing import List, Tuple, Callable
from config import asr_stream_config, IS_DEBUG
import numpy


class AudioCache:
    """
    两层流式缓存，要求能读取能写入更要能删除：
      一： 接收二进制流，(时间戳,二进制流)
      二： 基于安全时间删除，并排序好，添加到待识别区域

    如果开启debug，保存接收到的原始流式音频文件
    """
    SEQUENCE_TIME = 0.3     # 考虑到vad接受的片段长度
    STOP_TIME = 0.6     # ASR模型接受的片段长度

    def __init__(self):
        self._meta_cache: List[Tuple[float, bytes]] = []
        self._raw_buffer = bytearray()
        self._last_time = 0
        self._stop_time = 0
        self.final = False
        if IS_DEBUG:
            self.pcm_buffer = bytearray()

    def append_meta(self, data: bytes, vad_call: Callable, **kwargs):
        """
        解析二进制数据，自定义协议
        [8字节: 时间戳（double）][4字节: 音频长度][N字节: 音频数据]
        struct.unpack(">dI", data[:12])  将解析得到一个 float，一个Int 元组
        data[12:]后面的内容则是纯粹的音频流
        当输入的音频长度为0时 就是结束信号了

        插入数据，丢弃乱序数据,自动排序

        vad_call 行为：  传入一段音频流，判断这个音频流中是否有人说话的声音
        """
        timestamp, _ = struct.unpack(">dI", data[:12])
        print(timestamp, _)
        print(numpy.frombuffer(data, dtype=numpy.int16).astype(numpy.float32) / 32768.0)
        if IS_DEBUG:
            self.pcm_buffer.extend(data[12:])
        if timestamp < self._last_time:
            return
        if _ == 0:
            bits = self._hand_frame(vad_call, **kwargs)
            if bits is not None:
                self._raw_buffer.extend(bits)
            self.final = True
            return
        bisect.insort(self._meta_cache, (timestamp, data[12:]))
        if timestamp - self._last_time >= self.SEQUENCE_TIME:
            bits = self._hand_frame(vad_call, **kwargs)
            # 如果是静音片段，计算超时时间
            if bits is not None:
                self._raw_buffer.extend(bits)
            else:
                self._stop_time = timestamp - self._last_time
            self._last_time = self._meta_cache[-1][0]
            self._meta_cache.clear()

    def __iter__(self):
        """
        中间停顿型的静音，会导致停顿前的部分字符无法识别。
        :return:
        """
        while len(self._raw_buffer) >= asr_stream_config.chunk_size_bits:
            yield self._raw_buffer[:asr_stream_config.chunk_size_bits], False
            self._raw_buffer = self._raw_buffer[asr_stream_config.chunk_size_bits:]
        if self._stop_time >= self.STOP_TIME:
            yield self._raw_buffer + b'\x00' * (asr_stream_config.chunk_size_bits - len(self._raw_buffer)), False
        if self.final:
            if asr_stream_config.debug:
                self.debug_save()
            yield self._raw_buffer + b'\x00' * (asr_stream_config.chunk_size_bits - len(self._raw_buffer)), True


    def _hand_frame(self, call: Callable, **kwargs):
        _pcm = bytearray()
        for meta in self._meta_cache:
            _pcm.extend(meta[1])
        if call(_pcm, **kwargs):
            return _pcm
        return None

    def debug_save(self):
        from datetime import datetime
        import wave
        print("debug 保存流式音频文件")
        with wave.open(f"recorded_audios/received_audio-{datetime.now().strftime('%Y%m%d%H%M%S')}.wav", "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(self.pcm_buffer)
