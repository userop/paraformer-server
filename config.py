"""
在程序初始化时
"""

import os

N_MODEL_SINGLE_GPU = float(os.environ.get("N_MODEL_SINGLE_GPU", 5))
IS_DEBUG = os.environ.get("DEBUG", False)


class ParaformerConfig:
    cls_name = 'Paraformer'
    model_name: str


class FunAudioLLMConfig:
    cls_name = 'FunAudioLLM'
    model_name: str


class ParaformerStreamConfig:
    cls_name = 'ParaformerStream'
    model_name = os.environ.get("ASR_STREAM_MODEL_NAME")
    n_chunk_frame = os.environ.get("N_CHUNK_FRAME", 10)
    n_chunk_feature = os.environ.get("N_CHUNK_FEATURE", 5)
    encoder_chunk_size = os.environ.get("ENCODER_CHUNK_SIZE",4)
    decoder_chunk_size = os.environ.get("DECODER_CHUNK_SIZE",1)
    chunk_size_bits = int(16000 * 60 * 1e-3 * 2 * n_chunk_frame)


ASR_LIST = [ParaformerConfig, FunAudioLLMConfig]
ASR_STREAM_LIST = [ParaformerStreamConfig]

# ASR openai
def _get_asr_config():
    loader = os.environ.get("ASR_MODEL_LOAD")
    cls, model_name = loader.split(';', 1)
    for Conf in ASR_LIST:
        if Conf.cls_name == cls:
            Conf.model_name = model_name
            return Conf
    raise Exception(f"{cls}未匹配可用加载器: {[conf.cls_name for conf in ASR_LIST]}")


# Websocket ASR stream
# ASR openai
def _get_stream_asr_config():
    loader = os.environ.get("ASR_STREAM_MODEL_LOAD")
    cls, model_name = loader.split(';', 1)
    for Conf in ASR_STREAM_LIST:
        if Conf.cls_name == cls:
            Conf.model_name = model_name
            return Conf
    raise Exception(f"{cls}未匹配可用加载器: {[conf.cls_name for conf in ASR_STREAM_LIST]}")


asr_config = _get_asr_config()
asr_stream_config = _get_stream_asr_config()

