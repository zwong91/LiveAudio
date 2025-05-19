from .llm_interface import LLMInterface
from typing import List, Dict, AsyncGenerator
import os
import time
import asyncio
import uuid
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

MAX_TOKENS = 8192
MAX_NEW_TOKENS = 2048
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. You can hear and speak. You are chatting with a user over voice. Your voice and personality should be warm and engaging, with a lively and playful tone, full of charm and energy. The content of your responses should be conversational, nonjudgmental, and friendly.

Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class HFLLM(LLMInterface):
    MODEL_ID = "gaunernst/gemma-3-4b-it-int4-awq"

    def __init__(self):
        engine_args = AsyncEngineArgs(
            model=self.MODEL_ID,
            max_model_len=MAX_TOKENS,
            enable_prefix_caching=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def _truncate_messages(self, max_history=10):
        """保持消息历史在合理长度"""
        if len(self.messages) > max_history + 1:  # +1 是为了保留system消息
            # 保留system消息和最近的对话
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            recent_messages = self.messages[-(max_history-len(system_messages)):]
            self.messages = system_messages + [msg for msg in recent_messages if msg["role"] != "system"]

    async def generate_stream(self, messages: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        full_response_text = ""
        previously_yielded_text_len = 0
        request_id = uuid.uuid4().hex
        try:
            start_time = time.time()

            sampling_param = SamplingParams(max_tokens=MAX_NEW_TOKENS, skip_special_tokens=True,)

            # Construct the prompt using the current conversation history
            # self.messages already contains system prompt and previous turns
            conversation_history = self.messages + [dict(role="user", content=query)]

            prompt = self.tokenizer.apply_chat_template(
                conversation_history,
                tokenize=False,
                add_generation_prompt=True,
            )

            results_generator =  self.engine.generate(prompt, sampling_param, request_id)

            async for request_output in results_generator:
                text_outputs = [output.text for output in request_output.outputs]
                if text_outputs and len(text_outputs) > 0:
                    # Get the first output text (assuming there's only one generation)
                    text = text_outputs[0]

                    # Extract the new token(s) since the last yield
                    new_text = text[previously_yielded_text_len:]
                    previously_yielded_text_len = len(text)

                    # Append to full response
                    full_response_text += new_text

                    # Yield the new piece of text
                    yield new_text

        except asyncio.CancelledError:
            # If the stream is cancelled, abort the request on the vLLM engine side.
            await self.engine.abort(request_id)
            raise

        finally:
            # Update the message history with the user's query and the full assistant response
            self.messages.append(dict(role="user", content=query))
            self.messages.append(dict(role="assistant", content=full_response_text))
            # 添加历史记录管理，防止历史过长
            self._truncate_messages()
            print(f"huggingface llm time: {time.time() - start_time:.4f}s")
