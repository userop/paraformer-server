"""
加载模型，并提供语音识别能力
"""
import time
import torch
from config import asr_config, asr_stream_config, N_MODEL_SINGLE_GPU
from models import lazy_load_model


class ModelList:
    def __init__(self, ):
        self._models = []
        self._idx = 0
        self.n_gpu = 0

    def load_model(self, config, single_gb: float = 2.0):
        devices = list(range(torch.cuda.device_count()))
        for device in devices:
            free, total = torch.cuda.mem_get_info(device)
            free /= 1024 ** 3
            n_single = min(N_MODEL_SINGLE_GPU, free // single_gb)
            if n_single < 1:
                print("指定GPU没有足够内存加载语音模型，请换张卡或者有足够内存时(>=2GB)再尝试")
            else:
                print(f"将在cuda:{device}中加载{n_single}个实例")
                self._models.extend([lazy_load_model(device, config) for _ in range(int(n_single))])
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
        self.zh_model.load_model(asr_config)
        self.stream_model.load_model(asr_stream_config)

    def recognize(self, stream: bool):
        if stream:
            return self.stream_model.get_model()
        return self.zh_model.get_model()


asr_model = ParaModelASR()
