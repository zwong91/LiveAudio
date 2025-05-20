import os
import time
import logging
import torch
import numpy as np
from transformers import pipeline
from .asr_interface import ASRInterface

class WhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_name = kwargs.get("model_name", "openai/whisper-large-v3-turbo")
        language = kwargs.get("language", "zh")  # 可选语言参数，例如 "zh"、"en" 等
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": "zh"}
        )

    async def transcribe(self, client):
        start_time = time.time()
        # 转换音频格式
        samples = np.frombuffer(client.scratch_buffer, dtype=np.int16)
        float_samples = samples.astype(np.float32) / 32768.0

        result = self.asr_pipeline(float_samples, return_timestamps=True)

        end_time = time.time()
        asr_duration = end_time - start_time
        logging.info(f"ASR pipeline processing time: {asr_duration:.4f} seconds, asr text: {result['text']}")
        to_return = {
            "target_lang": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            "language_probability": None,
            "text": result["text"].strip(),
            "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return
