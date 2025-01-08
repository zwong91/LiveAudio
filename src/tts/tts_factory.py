from .edge_tts import EdgeTTS
from .gtts import GTTS
from .openai import OpenAITTS
from .xtts import XTTS_v2

class TTSFactory:
    @staticmethod
    def create_tts_pipeline(tts_type, **kwargs):
        if tts_type == "edge":
            return EdgeTTS(**kwargs)
        elif tts_type == "gtts":
            return GTTS(**kwargs)
        elif tts_type == "openai":
            return OpenAITTS(**kwargs)
        elif tts_type == "xtts-v2":
            return XTTS_v2(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
