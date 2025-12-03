

class AudioVAD:
    """
    声音活动检测
      前端会将静音内容通过websocket传递到后端 这是后台要做VAD的主要原因

      使用 webrtcvad
      内部做了很多对声音流的处理算子
      核心接口就一个，判断是否有人说话
      数据长度要求 10ms / 20ms / 30ms，其它长度 没效果

    """
    def __init__(self, frame_ms: int = 30):
        ...

    def stream_vad(self, audio: bytes, frame_ms: int = 30) -> bool:
        """
        传过来的二进制流单长度基本在合理范围内
        也就是audio片段不会过长也不会短  30ms 16k/Hz
        判断这个片段中是否有人说话

        :return:
        """
        # 二进制位数  位数超过bit直接切片 audio[:bits]，不足的补0即可
        bits = int(16000 * frame_ms * 1e-3 * 2)

    def split2join(self, audio: bytes, frame_ms: int = 30) -> bytes:
        """
        将音频按照长度时间长度切片
        将识别出来有人说话的片段拼接到一起合并得到新的音频流

        :param audio:
        :return:
        """
        # 根据bits长度将audio切片后处理
        bits = int(16000 * frame_ms * 1e-3 * 2)
