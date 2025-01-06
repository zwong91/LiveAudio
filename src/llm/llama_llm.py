from .llm_interface import LLMInterface
from typing import List, Optional, Tuple, Dict
import os
import time
import json
import random

from dotenv import load_dotenv
# Load environment variables
load_dotenv(override=True)

import torch
import asyncio

from llama_cpp import Llama

from .prompt import translation_prompt, chat_prompt

class LlamaLLM(LLMInterface):
    def __init__(
        self, 
        model_path="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        device="cuda",
        sys_prompt="",
        chat_format=None,
        temperature=0.7,
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

        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # # Load initial content from vault.txt
        # self.vault_content = []
        # vault_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio")), "vault.txt")
        # if os.path.exists(vault_path):
        #     with open(vault_path, "r", encoding="utf-8") as vault_file:
        #         self.vault_content = vault_file.readlines()
        # self.vault_embeddings = self.embedding_model.encode(self.vault_content, convert_to_tensor=True) if self.vault_content else []
    
    def get_relevant_context(self, user_input, vault_embeddings, top_k=3):
        """
        Retrieves the top-k most relevant context from the vault based on the user input.
        Local RAG embedding search
        """
        if len(vault_embeddings) == 0: # Check if the tensor has any elements
            return []
        # Encode the user input
        input_embedding = self.embedding_model.encode([user_input], convert_to_tensor=True)
        # Compute cosine similarity between the input and vault embeddings
        cos_scores = util.cos_sim(input_embedding, vault_embeddings)[0]
        # Adjust top_k if it's greater than the number of available scores
        top_k = min(top_k, len(cos_scores))
        # Sort the scores and get the top-k indices
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        print(f"Length of vault_content: {len(self.vault_content)}")
        print(f"Top indices: {top_indices}")
        # Get the corresponding context from the vault
        relevant_context = [self.vault_content[idx].strip() for idx in top_indices]
        return relevant_context

    async def generate(
        self,
        query,
        simultaneous,
        lang_tag: str,
        stream,
        max_length=100,
    ):
        query += f"\n\nalways use {lang_tag} answer"
        template = translation_prompt if simultaneous else chat_prompt
        system_prompt = template.replace("{{target_lang}}", lang_tag)
        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query})
        out = self.model.create_chat_completion(
            messages, stream=True, temperature=self.temperature
        )
        response_text = ""
        for o in out:
            if "content" in o["choices"][0]["delta"].keys():
                text = o["choices"][0]["delta"]["content"]
                response_text += text
                yield text
            if o["choices"][0]["finish_reason"] is not None:
                break

    async def generate_response(self, history: List[Dict[str, str]], query: str, simultaneous: bool, lang_tag: str, stream:  bool, max_tokens: int = 64) -> Tuple[str, List[Dict[str, str]]]:
        start_time = time.time()

        out = self.generate(history, query, simultaneous, lang_tag, stream)
        response = ""
        async for text in out:
            # which stores the transcription if interruption occurred. stop generating
            # if not interrupt_queue.empty():
            #     print("interruption detected LLM")
            #     break
            # TODO: text output queue where the result is accumulated
            response += text

        history.append({"role": "assistant", "content": response})
        history = history[-20:]

        end_time = time.time()
        print(f"llama llm time: {end_time - start_time:.4f} seconds")
        return response, history 
