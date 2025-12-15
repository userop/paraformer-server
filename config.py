import os
from yaml import safe_load


class ParaformerConfig:
    """
    配置信息只读，防止篡改
    """
    def __init__(self):
        """
        先从环境变量中加载，再到后台配置文件中加载，如果配置文件也没有抛出异常
        """
        self._asr_model_path = os.environ.get("ASR_MODEL_PATH")
        self._asr_stream_model_path = os.environ.get("ASR_STREAM_MODEL_PATH")
        self._debug = os.environ.get("DEBUG")
        # 流式识别模型输入数据参数，输入处理不好会直接影响识别效果
        self._n_chunk_frame = os.environ.get("N_CHUNK_FRAME",10)
        self._n_chunk_feature = os.environ.get("N_CHUNK_FEATURE",5)
        self._encoder_chunk_size = os.environ.get("ENCODER_CHUNK_SIZE",4)
        self._decoder_chunk_size = os.environ.get("DECODER_CHUNK_SIZE",1)

        if os.path.exists("model.yaml"):
            with open('model.yaml', 'r', encoding='utf-8') as f:
                config = safe_load(f)
                if not self._asr_model_path:
                    self._asr_model_path = config["paraformer-zh"]["model_path"]
                if not self._asr_stream_model_path:
                    self._asr_stream_model_path = config["paraformer-zh-streaming"]["model_path"]
                if not self._debug:
                    self._debug = config.get("debug")

    @property
    def zh_model(self):
        return self._asr_model_path

    @property
    def zh_stream_model(self):
        return self._asr_stream_model_path

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

    @property
    def debug(self):
        return self._debug



asr_config = ParaformerConfig()
