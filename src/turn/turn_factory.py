from .lk_turn import LKTurn
from .smart_turn import SmartTurn

class TurnFactory:
    @staticmethod
    def create_turn_pipeline(turn_type, **kwargs):
        if turn_type == "livekit":
            return LKTurn(**kwargs)
        elif turn_type == "pipecat":
            return SmartTurn(**kwargs)
        else:
            raise ValueError(f"Unknown Turn pipeline type: {turn_type}")
