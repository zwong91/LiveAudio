from typing import List, Tuple, Dict, AsyncGenerator

class LLMInterface:
    async def generate_stream(self, messages: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        """
        流式生成回复

        Args:
            messages: 对话历史
            query: 用户输入
            simultaneous: 是否同声传译模式
            target_lang: 目标语言

        Yields:
            str: 生成的文本片段
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
