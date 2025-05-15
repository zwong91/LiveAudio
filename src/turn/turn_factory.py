from .lk_turn import LKTurn
from .pipecat_turn import PIPETurn

class TurnFactory:
    @staticmethod
    def create_turn_pipeline(turn_type, **kwargs):
        if turn_type == "livekit":
            return LKTurn(**kwargs)
        elif turn_type == "pipecat":
            return PIPETurn(**kwargs)
        else:
            raise ValueError(f"Unknown Turn pipeline type: {turn_type}")
