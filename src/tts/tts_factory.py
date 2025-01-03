from .edge_tts import EdgeTTS
from .xtts import XTTS_v2
from .cosyvoice import CosyVoice_v2

class TTSFactory:
    @staticmethod
    def create_tts_pipeline(tts_type, **kwargs):
        if tts_type == "edge":
            return EdgeTTS(**kwargs)
        elif tts_type == "xtts-v2":
            return XTTS_v2(**kwargs)
        elif tts_type == "cosy-v2":
            return CosyVoice_v2(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
