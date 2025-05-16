from .whisper_asr import WhisperASR

class ASRFactory:
    @staticmethod
    def create_asr_pipeline(asr_type, **kwargs):
        if asr_type == "whisper":
            return WhisperASR(**kwargs)
        else:
            raise ValueError(f"Unknown ASR pipeline type: {asr_type}")
