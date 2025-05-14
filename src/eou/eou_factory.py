from .eou_detector import  EOUDetector

class EOUFactory:
    @staticmethod
    def create_eou_pipeline(turn_type, **kwargs):
        if turn_type == "livekit":
            return EOUDetector(**kwargs)
        else:
            raise ValueError(f"Unknown Turn pipeline type: {turn_type}")
