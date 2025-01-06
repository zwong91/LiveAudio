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

#from sentence_transformers import SentenceTransformer, util

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    # required but ignored
    api_key='ollama',
)

from ollama import AsyncClient

from .prompt import translation_prompt, chat_prompt
class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model: str = "qwen2.5:3b",
    ):
        # Ollama should be installed and running
        #curl -fsSL https://ollama.com/install.sh | sh
        #ollama.pull(model)
        self.model = model

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

    async def generate(self, history: List[Dict[str, str]], vault_input: str, simultaneous: bool, max_length: int = 128) -> Tuple[str, List[Dict[str, str]]]:
        # with open("vault.txt", "a", encoding="utf-8") as vault_file:
        #     print("Wrote to info.")
        #     vault_file.write(vault_input + "\n")
        # vault_content = open("vault.txt", "r", encoding="utf-8").readlines()
        # vault_embeddings = self.embedding_model.encode(vault_content)
        # print(f"Length of vault_content: {len(vault_content)}")

        # relevant_context = self.get_relevant_context(vault_input, self.vault_embeddings)
        query = vault_input
        # if relevant_context:
        #     query = "\n".join(relevant_context) + "\n\n" + vault_input

        print(f"query: {query}")

        if history is None:
            history = []
        history.append({"role": "user", "content": query})
        system_prompt = translation_prompt if simultaneous else chat_prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
        stream = await AsyncClient().chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                'num_predict': 128,
                'num_predict': 128,
                'temperature': 1,
            },
        )

        async for chunk in stream:
            if chunk["message"]["content"] is not None:
                yield chunk["message"]["content"]


    async def generate_response(self, history: List[Dict[str, str]], query: str, simultaneous: bool, stream:  bool, max_tokens: int = 128) -> Tuple[str, List[Dict[str, str]]]:
        start_time = time.time()

        if history is None:
            history = []
        history.append({"role": "user", "content": query})
        system_prompt = translation_prompt if simultaneous else chat_prompt
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=128,
            temperature=1,
        )

        role = response.choices[0].message.role
        response_content = response.choices[0].message.content

        history.append({"role": "assistant", "content": response_content})
        history = history[-10:]

        end_time = time.time()
        print(f"ollama llm time: {end_time - start_time:.4f} seconds")
        return response_content, history     
