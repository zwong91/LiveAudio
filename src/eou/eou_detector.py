"""
End-of-turn detection Python implementation
Original source: LiveKit Agents Project
License: Apache License 2.0
"""

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, snapshot_download
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import logging

# Constants
HG_MODEL = "livekit/turn-detector"
ONNX_FILENAME = "model_q8.onnx"
MODEL_REVISION = "v1.2.1"
MAX_HISTORY = 4
MAX_HISTORY_TOKENS = 512
UNLIKELY_THRESHOLD = 0.15

class EOUDetector:
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
            logging.info(f"EOUDetector initialization took: {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logging.error(f"EOUDetector initialization error: {e}")
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

        # Handle end of utterance token
        eou_token = "<|im_end|>"
        last_eou_index = convo_text.rfind(eou_token)
        return convo_text[:last_eou_index] if last_eou_index >= 0 else convo_text

    def softmax(self, logits):
        """Compute softmax probabilities for logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def predict_end_of_turn(self, chat_context):
        """
        Predict whether the current turn is complete.

        Args:
            chat_context (list): List of chat messages

        Returns:
            float: Probability of end of turn
        """
        formatted_text = self.format_chat_context(chat_context)

        inputs = self.tokenizer(
            formatted_text,
            add_special_tokens=False,
            return_tensors="np",
            max_length=MAX_HISTORY_TOKENS,
            truncation=True,
        )

        input_dict = {"input_ids": np.array(inputs["input_ids"], dtype=np.int64)}

        # Run inference
        output = self.session.run(["logits"], input_dict)

        # Process output
        logits = output[0]
        last_token_logits = logits[0, -1]
        probs = self.softmax(last_token_logits)

        return float(probs[self.eou_index])

    def is_turn_complete(self, chat_context, threshold=UNLIKELY_THRESHOLD):
        """
        Check if the current turn is complete.

        Args:
            chat_context (list): List of chat messages
            threshold (float): Probability threshold for end of turn

        Returns:
            bool: True if turn is complete, False otherwise
        """
        prob = self.predict_end_of_turn(chat_context[-MAX_HISTORY:])
        return prob >= threshold
