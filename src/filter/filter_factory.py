from .noisereduce_filter import NoisereduceFilter

class FilterFactory:
    @staticmethod
    def create_filter_pipeline(filter_type, **kwargs):
        if filter_type == "noisereduce":
            return NoisereduceFilter(**kwargs)
        else:
            raise ValueError(f"Unknown Filter pipeline type: {filter_type}")
