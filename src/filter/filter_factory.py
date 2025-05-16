from .noisereduce_filter import NoisereduceFilter
from .deepnet_filter import DeepNetFilter

class FilterFactory:
    @staticmethod
    def create_filter_pipeline(filter_type, **kwargs):
        if filter_type == "noisereduce":
            return NoisereduceFilter(**kwargs)
        elif filter_type == "deepfilter":
            return DeepNetFilter(**kwargs)
        else:
            raise ValueError(f"Unknown Filter pipeline type: {filter_type}")
