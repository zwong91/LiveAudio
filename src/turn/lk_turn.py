"""
End-of-turn detection Python implementation
Original source: LiveKit Agents Project
License: Apache License 2.0
"""

import json
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, snapshot_download
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import logging
from .turn_interface import TurnInterface

# Constants
HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
MODEL_REVISION = "v0.1.1-intl"
MAX_HISTORY_TURNS = 6
MAX_HISTORY_TOKENS = 256

class LKTurn(TurnInterface):
    def __init__(
        self,
        model_path=None,
        # if set, overrides the per-language threshold tuned for accuracy.
        # not recommended unless you're confident in the impact.
        unlikely_threshold: float | None = None,
    ):
        try:
            start_time = time.time()

            # Download or load model
            local_path = hf_hub_download(
                repo_id=HG_MODEL,
                filename=ONNX_FILENAME,
                subfolder="onnx",
                revision=MODEL_REVISION,
                local_files_only=True,
            )

            config_fname = hf_hub_download(
                repo_id=HG_MODEL,
                filename="languages.json",
                revision=MODEL_REVISION,
                local_files_only=True,
            )
            with open(config_fname) as f:
                self.languages = json.load(f)

            self.unlikely_threshold = unlikely_threshold
            # Initialize session and tokenizer
            self.session = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])
            self.tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL,
                revision=MODEL_REVISION,
                local_files_only=True,
                truncation_side="left",
            )
            logging.info(f"Loaded LKTurn model from {local_path}")
            logging.info(f"Using tokenizer: {self.tokenizer.name_or_path}")
            logging.info(f"Model inputs: {self.session.get_inputs()}")
            for input in self.session.get_inputs():
                print("Input name:", input.name)
                print("Input shape:", input.shape)
                print("Input type:", input.type)
            logging.info(f"Model outputs: {self.session.get_outputs()}")

            logging.info(f"LKTurn initialization took: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"LKTurn initialization error: {e}")
            raise

    def format_chat_ctx(self, chat_ctx):
        """Format the chat context for model input."""
        new_chat_ctx = []
        for msg in chat_ctx:
            content = msg["content"]
            if not content:
                continue

            msg["content"] = content
            new_chat_ctx.append(msg)

        convo_text = self.tokenizer.apply_chat_template(
            new_chat_ctx,
            add_generation_prompt=False,
            add_special_tokens=False,
            tokenize=False,
        )

        # remove the EOU token from current utterance
        ix = convo_text.rfind("<|im_end|>")
        text = convo_text[:ix]
        return text

    def unlikely_threshold(self, language: str | None) -> float | None:
        if language is None:
            return None

        lang = language.lower()
        # try the full language code first
        lang_data = self.languages.get(lang)

        # try the base language if the full language code is not found
        if lang_data is None and "-" in lang:
            base_lang = lang.split("-")[0]
            lang_data = self.languages.get(base_lang)

        if not lang_data:
            logging.warning(f"Language {language} not supported by EOU model")
            return None
        # if a custom threshold is provided, use it
        if self.unlikely_threshold is not None:
            return self.unlikely_threshold
        else:
            return lang_data["threshold"]

    def supports_language(self, language: str | None) -> bool:
        return self.unlikely_threshold(language) is not None


    async def predict_endpoint(self, context: Optional[List[Dict[str, str]]], last_language: str, audio: Optional[bytearray])-> Dict[str, Any]:
        """
        Predict whether the current turn is complete.

        Args:
            chat_ctx (list): List of chat messages

        Returns:
            float: Probability of end of turn
        """
        if context is not None and not isinstance(context, list):
            raise ValueError("context must be a list of messages")

        if not self.supports_language(last_language):
            logging.debug("Turn detector does not support language %s", last_language)

        unlikely_threshold = self.unlikely_threshold(last_language)
        if unlikely_threshold is None:
            return {
                "prediction": 0,
                "probability": 0.0,
            }
        start_time = time.perf_counter()
        formatted_text = self.format_chat_ctx(context[-MAX_HISTORY_TURNS:])

        inputs = self.tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )

        outputs = self.session.run(None, {"input_ids": inputs["input_ids"].astype("int64")})
        print(f"Model outputs: {outputs}")
        eou_probability = outputs[0][0]
        end_time = time.perf_counter()

        print(f"End of turn probability: {float(eou_probability):.4f}, duration: {end_time - start_time:.4f} seconds")

        prediction = 1 if float(eou_probability) >= unlikely_threshold else 0
        return {
            "prediction": prediction,
            "probability": float(eou_probability),
        }
