from .vad_interface import VADInterface

import torch
import numpy as np

class SileroVAD(VADInterface):
    def __init__(self,  **kwargs):
        # 加载 Silero VAD 模型
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=False,
        )
        # 获取需要的工具方法
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils

        self.sampling_rate = sampling_rate = 16000 # 采样率

    async def detect_activity(self, client):
        """
        使用Silero VAD模型进行语音活动检测。
        :param client: 传入的客户端对象，应该包含音频数据
        :return: 语音段落的时间戳列表
        """
        frames = np.frombuffer(client.scratch_buffer, dtype=np.int16)
        # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
        frames = frames / (1 << 15)
        audio_tensor = torch.tensor(frames.astype(np.float32))
        # 获取语音时间戳
        vad_results = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            return_seconds=True,
            sampling_rate=self.sampling_rate,
            threshold=0.5,
            min_speech_duration_ms=500,
            max_speech_duration_s=float('inf'),
            min_silence_duration_ms=200,
            speech_pad_ms=30
        )

        # 返回语音时间段（以秒为单位）
        vad_segments = [{"start": segment["start"], "end": segment["end"], "confidence": 1.0} for segment in vad_results]
        return vad_segments
