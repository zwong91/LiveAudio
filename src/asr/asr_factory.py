from .faster_whisper_asr import FasterWhisperASR
from .whisper_asr import WhisperASR
from .sensevoice_asr import SenseVoiceASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(asr_type, **kwargs):
        if asr_type == "whisper":
            return WhisperASR(**kwargs)
        elif asr_type == "faster_whisper":
            return FasterWhisperASR(**kwargs)
        elif asr_type == "sensevoice":
            return SenseVoiceASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {asr_type}")
