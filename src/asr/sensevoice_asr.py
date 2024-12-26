import os
import torch
import pysbd
import langid
import re
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

    def has_sentence_boundary(self, text, language):
        if language == "zh":
            return bool(re.search(r"[。！？]", text))  # 中文标点符号
        else:
            return bool(re.search(r"[.!?]", text))  # 英文等语言标点符号

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        if client.config["language"] is not None:
            text = self.asr_pipeline.generate(
                input=file_path, cache={},
                generate_kwargs={"language": client.config["language"]},
                use_itn=False, batch_size=64
            )[0]["text"].strip()
        else:
            text = self.asr_pipeline.generate(
                input=file_path, cache={},
                language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False, batch_size=64
            )[0]["text"].strip()

        os.remove(file_path)

        trans_text = rich_transcription_postprocess(text)
        # print(f"stt transcribe text: {trans_text}")
        # pysbd_language = langid.classify(trans_text)[0].strip()
        # seg = pysbd.Segmenter(language=pysbd_language, clean=False)

        # sentence_finished = False
        # first = True

        # n = 2  # number of times to see if the sentence ends
        # while not sentence_finished and n > 0:
        #     first = False
        #     if len(seg.segment(trans_text)) > 1 or self.has_sentence_boundary(trans_text, pysbd_language):
        #         sentence_finished = True
        #         print(f"Sentence boundary detected for language '{langid_detected_language}': {text}")
        #     else:
        #         n -= 1
        #         print(f"No sentence boundary detected. Tries left: {n}")
        
        # sentence = trans_text if sentence_finished else ""
        to_return = {
            "language": "zh-cn",
            "language_probability": None,
            "text": trans_text,
            "words": "UNSUPPORTED_BY_HUGGINGFACE_SENSEVOICE",
        }
        return to_return
