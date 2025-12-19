import os
import tempfile
import threading

from funasr import AutoModel


class BaseModel:
    """
    每个模型基本都不是线程安全的
    每个模型实例在工作时，当成一个上下文来维护
    """
    def __init__(self, device, model, **kwargs):
        self._cache = {}
        self._lock = threading.Lock()
        print(kwargs, model)
        self._model = AutoModel(model=model, device=device, disable_update=True, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache.clear()
        self._lock.release()
        return False

    def __enter__(self):
        self._lock.acquire(blocking=False)
        return self

    def lock(self):
        return self._lock.acquire(blocking=False)

    def audio_stream_recognition(self, data: bytes, is_final: bool = False) -> str:
        """
        流式识别，二进制音频流，返回识别片段
        流式模型和整段文件
        :param data:
        :param is_final:
        :return:
        """


class TempFileModel(BaseModel):
    def __init__(self, device, model, **kwargs):
        super(TempFileModel, self).__init__(device, model, **kwargs)
        self.temp_file = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache.clear()
        self._lock.release()
        if self.temp_file is not None:
            os.remove(self.temp_file.name)
            self.temp_file = None
        return False

    def __enter__(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        self._lock.acquire(blocking=False)
        return self

    def audio_recognition(self) -> str:
        """识别二进制音频数据，返回识别文本"""
        raise NotImplementedError



