import os
import numpy

from models.base import BaseModel, TempFileModel
from config import asr_config


class Paraformer(TempFileModel):
    def __init__(self, device, model):
        super().__init__(device, model)

    def audio_recognition(self) -> str:
        if os.path.getsize(self.temp_file.name) == 0:
            return "文件为空或异常格式，转录PCM失败"
        # 读取文件内容，ASR
        with open(self.temp_file.name, 'rb') as reader:
            data = reader.read()
        res = self._model.generate(data)[0]['text']
        if not res:
            res = "未识别到有效人声"
        return res


class ParaformerStream(BaseModel):
    def __init__(self, device, model):
        super().__init__(device, model)

    def audio_stream_recognition(self, data: bytes, is_final: bool = False) -> str:
        """
        模型要求输入 numpy float32， [-1.0, 1.0]之间的一维数组。
        bytearray -> int16 -> float32 是为了确保采样点数正确
        32768.0 则是int16的最大值
        使用float32是深度学习框架tensorflow的输入要求
        :param data:
        :param is_final:
        :return:
        """
        np_asr_float32 = numpy.frombuffer(data, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        res = self._model.generate(input=np_asr_float32, cache=self._cache, is_final=is_final,
                                   chunk_size=[0, asr_config.n_chunk_frame, asr_config.n_chunk_feature],
                             encoder_chunk_look_back=asr_config.encoder_chunk_size,
                             decoder_chunk_look_back=asr_config.decoder_chunk_size)
        return res[0]["text"]
