from .edge_tts import EdgeTTS
from .gtts import GTTS
from .system_tts import SystemTTS
from .elevenlabs_tts import ElevenlabTTS
from .xtts import XTTS_v2

class TTSFactory:
    @staticmethod
    def create_tts_pipeline(tts_type, **kwargs):
        if tts_type == "edge":
            return EdgeTTS(**kwargs)
        elif tts_type == "gtts":
            return GTTS(**kwargs)
        elif tts_type == "elevenlabs":
            return ElevenlabTTS(**kwargs)
        elif tts_type == "system":
            return SystemTTS(**kwargs)
        elif tts_type == "xtts-v2":
            return XTTS_v2(**kwargs)
        else:
            raise ValueError(f"Unknown TTS pipeline type: {tts_type}")
