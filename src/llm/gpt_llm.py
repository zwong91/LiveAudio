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

# 初始化模型
from openai._types import NOT_GIVEN
from openai import AsyncOpenAI
aclient = AsyncOpenAI()
MODEL = os.getenv('MODEL')
BASE_URL = os.getenv('BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from .prompt import translation_prompt, chat_prompt

class OpenAILLM(LLMInterface):
    def __init__(
        self, 
        model: str = "gpt-4o-mini",
        tools=None,
        tool_choice=NOT_GIVEN,
        tool_utterances=None,
        functions=None,
    ):
        self.model = MODEL
        aclient.api_key = OPENAI_API_KEY
        #aclient.base_url = "https://xyz-api.jongun2038.win/v1/"
        aclient.base_url = BASE_URL
        self.tools = tools
        self.tool_choice = tool_choice
        self.tool_utterances = tool_utterances
        self.functions = functions

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

    async def generate(self, history: List[Dict[str, str]], vault_input: str, simultaneous: bool, stream_mode: bool, max_lengths: int = 128) -> Tuple[str, List[Dict[str, str]]]:
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
        system_prompt = translation_prompt if simultaneous else chat_prompt
        if history is None:
            history = []
        history.append({"role": "user", "content": query})
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        self.messages.extend(history)

        finished = False
        while not finished:
            func_call = dict()
            function_call_detected = False
            stream = await aclient.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=max_lengths,
                temperature=1,
                stream=stream_mode,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )

            async for chunk in stream:
                finish_reason = chunk.choices[0].finish_reason
                if chunk.choices[0].delta.tool_calls is not None:
                    function_call_detected = True
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function.name:
                        func_call["name"] = tool_call.function.name
                        func_call["id"] = tool_call.id
                        func_call["arguments"] = ""
                        # Choose a utterance for the tool at random and output it for the tts
                        yield random.choice(
                            self.tool_utterances[func_call["name"]]
                        ) + " . "  # the period is to make
                        # it say immediately
                    if tool_call.function.arguments:
                        func_call["arguments"] += tool_call.function.arguments
                if function_call_detected and finish_reason == "tool_calls":
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": func_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": func_call["name"],
                                        "arguments": func_call["arguments"],
                                    },
                                }
                            ],
                        }
                    )
                    # run the function
                    function_response = self.functions[func_call["name"]](
                        **json.loads(func_call["arguments"])
                    )
                    self.messages.append(
                        {
                            "tool_call_id": func_call["id"],
                            "role": "tool",
                            "name": func_call["name"],
                            "content": function_response,
                        }
                    )
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                if finish_reason == "stop":
                    finished = True

    async def generate_response(self, history: List[Dict[str, str]], query: str, simultaneous: bool, stream:  bool, max_lengths: int = 32) -> Tuple[str, List[Dict[str, str]]]:
        start_time = time.time()

        out = self.generate(history, query, simultaneous, stream, max_lengths)
        response = ""
        async for text in out:
            # which stores the transcription if interruption occurred. stop generating
            # if not interrupt_queue.empty():
            #     print("interruption detected LLM")
            #     break
            # TODO: text output queue where the result is accumulated
            response += text

        # remove the tool utterances from the response
        # for tool in self.tool_utterances:
        #     response = response.replace(self.tool_utterances[tool], '')
        history.append({"role": "assistant", "content": response})
        history = history[-20:]

        end_time = time.time()
        print(f"openai llm time: {end_time - start_time:.4f} seconds")
        return response, history       
