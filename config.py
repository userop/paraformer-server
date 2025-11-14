from yaml import safe_load


class ParaformerConfig:
    """
    配置信息只读，防止篡改
    """
    def __init__(self):
        with open('model.yaml', 'r', encoding='utf-8') as f:
            config = safe_load(f)
        self._zh_model = config["paraformer-zh"]["model_path"]
        self._zh_stream_model = config["paraformer-zh-streaming"]["model_path"]
        self._encoder_chunk_size = config["paraformer-zh-streaming"]["encoder_chunk_size"]
        self._decoder_chunk_size = config["paraformer-zh-streaming"]["decoder_chunk_size"]
        self._n_chunk_frame = config["paraformer-zh-streaming"]["n_chunk_frame"]
        self._n_chunk_feature = config["paraformer-zh-streaming"]["n_chunk_feature"]

    @property
    def zh_model(self):
        return self._zh_model

    @property
    def zh_stream_model(self):
        return self._zh_stream_model

    @property
    def chunk_size_bits(self):
        """
        获取模型想要的字节切片长度
        采样 60ms @16kHz
        2个字节对应1个采样点

        每次识别，输入帧数  （推荐10帧 也就是600ms）
        :return:
        """
        return int(16000 * 60 * 1e-3 * 2 * self._n_chunk_frame)

    @property
    def encoder_chunk_size(self):
        return self._encoder_chunk_size

    @property
    def decoder_chunk_size(self):
        return self._decoder_chunk_size

    @property
    def n_chunk_frame(self):
        return self._n_chunk_frame

    @property
    def n_chunk_feature(self):
        return self._n_chunk_feature



asr_config = ParaformerConfig()
