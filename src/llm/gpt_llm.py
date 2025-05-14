from .llm_interface import LLMInterface
from typing import List, Dict, AsyncGenerator
import os
import time
import asyncio
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .prompt import translation_prompt, chat_prompt

# Load environment variables
load_dotenv(override=True)

MODEL = os.getenv('MODEL')
BASE_URL = os.getenv('BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class OpenAILLM(LLMInterface):
    def __init__(
        self,
        model: str = "gpt-4-mini",
        tools=None,
        tool_choice=NOT_GIVEN,
        functions=None,
    ):
        self.model = MODEL
        self.aclient = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL
        )
        self.tools = tools
        self.tool_choice = tool_choice
        self.functions = functions

    def _prepare_messages(self, history: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str):
        """准备消息上下文"""
        if history is None:
            history = []

        # 获取相关知识库内容
        history.append({"role": "user", "content": query})

        template = translation_prompt if simultaneous else chat_prompt
        system_prompt = template.replace("{{target_lang}}", target_lang or "")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        return messages, history

    async def generate_stream(self, history: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        """流式生成回复，支持中断"""
        try:
            messages, updated_history = self._prepare_messages(history, query, simultaneous, target_lang)

            start_time = time.time()
            stream = await self.aclient.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except asyncio.CancelledError:
            print("OpenAI generation cancelled")
            raise
        finally:
            end_time = time.time()
            print(f"openai llm time: {end_time - start_time:.4f} seconds")
