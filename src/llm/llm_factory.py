from .gpt_llm import OpenAILLM
from .hf_llm import HFLLM
from .ollama_llm import OllamaLLM
from .workflow import WorkflowLLM
class LLMFactory:
    @staticmethod
    def create_llm_pipeline(engine_type, **kwargs):
        if engine_type == "openai":
            return OpenAILLM(**kwargs)
        elif engine_type == "hf":
            return HFLLM(**kwargs)
        elif engine_type == "ollama":
            return OllamaLLM(**kwargs)
        elif engine_type == "dify":
            return WorkflowLLM(**kwargs)
        else:
            raise ValueError(f"Unknown LLM pipeline type: {engine_type}")
