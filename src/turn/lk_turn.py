"""
End-of-turn detection Python implementation
Original source: LiveKit Agents Project
License: Apache License 2.0
"""

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
ONNX_FILENAME = "model.onnx"
MODEL_REVISION = "v1.2.1"
MAX_HISTORY = 4
MAX_HISTORY_TOKENS = 512
UNLIKELY_THRESHOLD = 0.15

class LKTurn(TurnInterface):
    def __init__(self, model_path=None):
        """
        Initialize the ONNX model and tokenizer.

        Args:
            model_path (str, optional): Local model path. If None, download from HuggingFace.
        """
        try:
            start_time = time.time()

            # Download or load model
            model_file = hf_hub_download(
                repo_id=HG_MODEL,
                filename=ONNX_FILENAME,
                subfolder="onnx",
                revision=MODEL_REVISION
            ) if model_path is None else str(Path(model_path) / ONNX_FILENAME)

            # Initialize session and tokenizer
            self.session = ort.InferenceSession(model_file)
            self.tokenizer = AutoTokenizer.from_pretrained(
                HG_MODEL if model_path is None else model_path,
                revision=MODEL_REVISION,
                truncation_side="left",
                trust_remote_code=True
            )

            self.eou_index = self.tokenizer.encode("<|im_end|>")[0]
            for output in self.session.get_outputs():
                print("Output name:", output.name)
            logging.info(f"LKTurn initialization took: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"LKTurn initialization error: {e}")
            raise

    def normalize(self, text):
        """
        Normalize the input text by removing punctuation and standardizing whitespace.
        """
        PUNCS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

        stripped = ''.join(char for char in text if char not in PUNCS)
        return ' '.join(stripped.lower().split())

    def format_chat_context(self, chat_context):
        """Format the chat context for model input."""
        # Normalize and filter empty messages
        normalized_context = [
            msg for msg in [
                {**msg, 'content': self.normalize(msg['content'])}
                for msg in chat_context
            ]
            if msg['content']
        ]

        # Apply chat template
        convo_text = self.tokenizer.apply_chat_template(
            normalized_context,
            add_generation_prompt=True,
            add_special_tokens=False,
            tokenize=False
        )

        # Remove the EOU token from current utterance
        eou_token = "<|im_end|>"
        ix = convo_text.rfind(eou_token)
        return convo_text[:ix] if ix != -1 else convo_text

    async def predict_endpoint(self, context: Optional[List[Dict[str, str]]], audio: Optional[bytearray])-> Dict[str, Any]:
        """
        Predict whether the current turn is complete.

        Args:
            chat_context (list): List of chat messages

        Returns:
            float: Probability of end of turn
        """
        if context is not None and not isinstance(context, list):
            raise ValueError("context must be a list of messages")

        formatted_text = self.format_chat_context(context[-MAX_HISTORY:])

        inputs = self.tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )

        input_dict = {"input_ids": np.array(inputs["input_ids"], dtype=np.int64)}

        outputs = self.session.run(None, {"input_ids": inputs["input_ids"]})
        completion_prob = outputs[0][0]  # Extract probability

        print(f"End of turn probability: {completion_prob:.4f}")
        prediction = 1 if completion_prob >= UNLIKELY_THRESHOLD else 0
        return {
            "prediction": prediction,
            "probability": completion_prob,
        }
