from .llm_interface import LLMInterface
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import os
import time
import json
import random

from dotenv import load_dotenv
load_dotenv(override=True)

import torch
import asyncio
from sentence_transformers import SentenceTransformer, util
from llama_cpp import Llama
from ..prompts.sys_prompt import translation_prompt, chat_prompt

class LlamaLLM(LLMInterface):
    def __init__(
        self,
        model_path="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        device="cuda",
        temperature=0.7,
        chat_format=None,
    ):
        self.model = Llama.from_pretrained(
            repo_id=model_path,
            filename="*q8_0.gguf",
            n_ctx=4096,
            n_gpu_layers=-1 if device == "cuda" else 0,
            verbose=False,
            chat_format=chat_format,
        )
        self.temperature = temperature

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

    async def generate_stream(self, messages: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        """流式生成回复，支持中断"""
        try:
            start_time = time.time()
            stream = self.model.create_chat_completion(
                messages,
                stream=True,
                temperature=self.temperature
            )

            for output in stream:
                if output["choices"][0]["delta"].get("content"):
                    yield output["choices"][0]["delta"]["content"]
                    # 让出控制权，允许检查中断
                    await asyncio.sleep(0)

                # 更新知识库
                with open("vault.txt", "a", encoding="utf-8") as vault_file:
                    vault_file.write(query + "\n")
                self.vault_content.append(query)
                self.vault_embeddings = self.embedding_model.encode(self.vault_content, convert_to_tensor=True)

        except asyncio.CancelledError:
            print("Llama generation cancelled")
            raise
        finally:
            end_time = time.time()
            print(f"llama llm time: {end_time - start_time:.4f} seconds")
