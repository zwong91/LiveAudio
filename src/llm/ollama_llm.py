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

from sentence_transformers import SentenceTransformer, util
from ..prompts.sys_prompt import translation_prompt, chat_prompt

from ollama import AsyncClient


class OllamaLLM(LLMInterface):
    def __init__(self, model: str = "gemma3:27b", base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url
        self._request_id = None
        self._client = AsyncClient(host=base_url, headers={'x-api-key': 'ollama'})
        self._max_retries = 3
        self._retry_delay = 1

        # Initialize embedding model and vault content
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vault_content = []
        vault_path = os.path.join(os.path.abspath(os.getcwd()), "vault.txt")
        if os.path.exists(vault_path):
            with open(vault_path, "r", encoding="utf-8") as vault_file:
                self.vault_content = vault_file.readlines()
        self.vault_embeddings = self.embedding_model.encode(self.vault_content, convert_to_tensor=True) if self.vault_content else []

    def get_relevant_context(self, user_input, vault_embeddings, top_k=3):
        """获取知识库中最相关的上下文"""
        if len(vault_embeddings) == 0:
            return []

        input_embedding = self.embedding_model.encode([user_input], convert_to_tensor=True)
        cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

        print(f"Length of vault_content: {len(self.vault_content)}")
        print(f"Top indices: {top_indices}")
        relevant_context = [self.vault_content[idx].strip() for idx in top_indices]
        return relevant_context

    def _prepare_messages(self, history: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str):
        """准备消息上下文"""
        if history is None:
            history = []

        # # 获取相关知识库内容
        # relevant_context = self.get_relevant_context(query, self.vault_embeddings)
        # if relevant_context:
        #     query = "\n".join(relevant_context) + "\n\n" + query

        #query = query + "\n\n" + f"always use {target_lang} answer" if target_lang else query
        history.append({"role": "user", "content": query})

        template = translation_prompt if simultaneous else chat_prompt
        system_prompt = template.replace("{{target_lang}}", target_lang or "")

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        return messages, history

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
            stream = await self._client.chat(
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
