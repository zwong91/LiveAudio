from .edge_tts import EdgeTTS
from .gtts import GTTS
from .elevenlabs_tts import ElevenlabTTS
from .kokoro import Kokoro

class TTSFactory:
    @staticmethod
    def create_tts_pipeline(tts_type, **kwargs):
        if tts_type == "edge":
            return EdgeTTS(**kwargs)
        elif tts_type == "gtts":
            return GTTS(**kwargs)
        elif tts_type == "elevenlabs":
            return ElevenlabTTS(**kwargs)
        elif tts_type == "xtts-v2":
            return XTTS_v2(**kwargs)
        elif tts_type == "kokoro":
            return Kokoro(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
