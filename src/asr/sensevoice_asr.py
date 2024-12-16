import os
import torch

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

from src.utils.audio_utils import save_audio_to_file

from .asr_interface import ASRInterface

class SenseVoiceASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = kwargs.get("model_name", "iic/SenseVoiceSmall")
        print("loading ASR model...")
        self.asr_pipeline = AutoModel(
            model=model_name,
            trust_remote_code=True, 
            device=device
        )

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        if client.config["language"] is not None:
            to_return = self.asr_pipeline.generate(
                input=file_path, cache={},
                generate_kwargs={"language": client.config["language"]},
                use_itn=False, batch_size=64
            )[0]["text"]
        else:
            to_return = self.asr_pipeline.generate(
                input=file_path, cache={},
                language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False, batch_size=64
            )[0]["text"]

        os.remove(file_path)

        text = rich_transcription_postprocess(to_return.strip())
        to_return = {
            "language": "zh-cn",
            "language_probability": None,
            "text": text,
            "words": "UNSUPPORTED_BY_HUGGINGFACE_SENSEVOICE",
        }
        return to_return
