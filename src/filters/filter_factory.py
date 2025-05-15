from .koala_filter import KoalaFilter
from .krisp_filter import KrispFilter
from .noisereduce_filter import NoiseReduceFilter

class FilterFactory:
    @staticmethod
    def create_filter_pipeline(filter_type, **kwargs):
        if filter_type == "koala":
            return KoalaFilter(**kwargs)
        elif filter_type == "krisp":
            return KrispFilter(**kwargs)
        elif filter_type == "noisereduce":
            return NoiseReduceFilter(**kwargs)
        else:
            raise ValueError(f"Unknown Filter pipeline type: {filter_type}")
