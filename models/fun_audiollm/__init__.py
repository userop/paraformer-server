import os
from models.base import TempFileModel


class FunAudioLLM(TempFileModel):
    def __init__(self, device, model):
        super().__init__(device, model, trust_remote_code=True,
                         remote_code=f"{os.path.dirname(__file__)}/model.py")

    def audio_recognition(self) -> str:
        if os.path.getsize(self.temp_file.name) == 0:
            return "文件为空或异常格式，转录PCM失败"
        res = self._model.generate(input=[self.temp_file.name], cache=self._cache, batch_size=1)[0]['text']
        if not res:
            res = "未识别到有效人声"
        return res
