from .openai_llm import OpenAILLM
from .workflow import WorkflowLLM
class LLMFactory:
    @staticmethod
    def create_llm_pipeline(engine_type, **kwargs):
        if engine_type == "openai":
            return OpenAILLM(**kwargs)
        elif engine_type == "workflow":
            return WorkflowLLM(**kwargs)
        else:
            raise ValueError(f"Unknown LLM pipeline type: {engine_type}")
