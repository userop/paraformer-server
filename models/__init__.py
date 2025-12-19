from importlib import import_module
from models.paraformer import Paraformer, ParaformerStream
from models.fun_audiollm import FunAudioLLM


def lazy_load_model(device, config):
    try:
        module = import_module('models')
        loader = getattr(module, config.cls_name)
    except ImportError:
        raise RuntimeError(".")
    return loader(device, f"/app/serve/models/{config.model_name}")
