from .llm_interface import LLMInterface
from typing import List, Dict, AsyncGenerator
import os
import time
import asyncio
import uuid
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.utils import RequestOutput

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

    async def generate_stream(self, messages: List[Dict[str, str]], query: str, simultaneous: bool, target_lang: str) -> AsyncGenerator[str, None]:
        text = ""
        request_id = uuid.uuid4().hex
        try:
            start_time = time.time()

            sampling_param = SamplingParams(max_tokens=MAX_NEW_TOKENS)
            prompt = self.tokenizer.apply_chat_template(
                self.messages + [dict(role="user", content=query)],
                tokenize=False,
                add_generation_prompt=True,
            )

            self.engine.add_request(request_id, prompt, sampling_param)
            async for chunk in self._get_streaming_results(request_id):
                yield chunk
                text += chunk

        except asyncio.CancelledError:
            await self.engine.abort(request_id)
            raise

        finally:
            self.messages.append(dict(role="user", content=query))
            self.messages.append(dict(role="assistant", content=text))
            print(f"llm time: {time.time() - start_time:.4f}s")

    async def _get_streaming_results(self, request_id: str) -> AsyncGenerator[str, None]:
        cursor = 0
        while True:
            result = await self.engine.get_results(request_id)
            if result is not None:
                output = result.outputs[0].text
                new_output = output[cursor:]
                if new_output:
                    yield new_output
                    cursor = len(output)
                if result.finished:
                    break
            else:
                await asyncio.sleep(0.01)
