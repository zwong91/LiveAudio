from .llm_interface import LLMInterface
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import os
import time
import json
import random

from dotenv import load_dotenv
# Load environment variables
load_dotenv(override=True)

import torch
import asyncio

#from sentence_transformers import SentenceTransformer, util

from openai import AsyncOpenAI
aclient = AsyncOpenAI()
aclient.api_key = 'ollama'
aclient.base_url = "http://localhost:11434/v1/"

from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    # required but ignored
    api_key='ollama',
)

from ollama import AsyncClient

from .prompt import translation_prompt, chat_prompt

class OllamaLLM(LLMInterface):
    def __init__(self, model: str = "qwen3:0.6b"):
        self.model = model
        self._request_id = None
        self._current_task = None

    def _prepare_messages(self, history: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str):
        """准备消息上下文"""
        if history is None:
            history = []

        query = query + "\n\n" + f"always use {target_lang} answer" if target_lang else query
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
            stream = await AsyncClient().chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    'num_predict': 256,
                    'temperature': 1,
                },
            )

            async for chunk in stream:
                if chunk["message"]["content"] is not None:
                    yield chunk["message"]["content"]

        except asyncio.CancelledError:
            print("LLM generation cancelled")
            raise
        finally:
            end_time = time.time()
            print(f"ollama llm time: {end_time - start_time:.4f} seconds")
