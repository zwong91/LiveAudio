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

from ollama import AsyncClient

from .prompt import translation_prompt, chat_prompt

class OllamaLLM(LLMInterface):
    def __init__(self, model: str = "qwen3:0.6b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._request_id = None
        self._client = AsyncClient(host=base_url)
        self._max_retries = 3
        self._retry_delay = 1

    async def _get_client(self):
        """Get a working client with retries"""
        for attempt in range(self._max_retries):
            try:
                # Quick health check
                await self._client.health()
                return self._client
            except Exception as e:
                if attempt == self._max_retries - 1:
                    raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}")
                await asyncio.sleep(self._retry_delay * (attempt + 1))
                self._client = AsyncClient(host=self.base_url)

    async def generate_stream(
        self,
        history: List[Dict[str, str]],
        query: str,
        simultaneous: bool,
        target_lang: str
    ) -> AsyncGenerator[str, None]:
        """Stream response with connection retry logic"""
        messages, updated_history = self._prepare_messages(
            history, query, simultaneous, target_lang
        )

        start_time = time.time()
        try:
            client = await self._get_client()
            stream = await client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    'num_predict': 256,
                    'temperature': 1,
                }
            )

            async for chunk in stream:
                if chunk["message"]["content"] is not None:
                    yield chunk["message"]["content"]

        except asyncio.CancelledError:
            print("LLM generation cancelled")
            raise
        except Exception as e:
            print(f"Error in LLM generation: {str(e)}")
            raise
        finally:
            end_time = time.time()
            print(f"ollama llm time: {end_time - start_time:.4f} seconds")
