import os
import subprocess
import tempfile
import time

import webrtcvad


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
