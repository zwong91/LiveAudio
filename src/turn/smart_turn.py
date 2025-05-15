#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from typing import Any, Dict, List, Optional

import numpy as np
import logging

from .turn_interface import TurnInterface

try:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2BertForSequenceClassification
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class SmartTurn(TurnInterface):
    def __init__(self, *, smart_turn_model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        if not smart_turn_model_path:
            # Define the path to the pretrained model on Hugging Face
            smart_turn_model_path = "pipecat-ai/smart-turn"

        logging.debug("Loading Local Smart Turn model...")
        # Load the pretrained model for sequence classification
        self._turn_model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            smart_turn_model_path
        )
        # Load the corresponding feature extractor for preprocessing audio
        self._turn_processor = AutoFeatureExtractor.from_pretrained(smart_turn_model_path)
        # Set device to GPU if available, else CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move model to selected device and set it to evaluation mode
        self._turn_model = self._turn_model.to(self._device)
        self._turn_model.eval()
        logging.debug("Loaded Local Smart Turn")

    async def predict_endpoint(self, context: Optional[List[Dict[str, str]]], buffer: Optional[bytearray]) -> Dict[str, Any]:

        # Check input type
        if buffer is not None and not isinstance(buffer, bytearray):
            raise ValueError("Input buffer must be of type bytearray")

        audio_int16 = np.frombuffer(buffer, np.int16)
        audio_array = audio_int16.astype(np.float32) / 32768.0

        inputs = self._turn_processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=800,  # Maximum length as specified in training
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move input tensors to the same device as the model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = self._turn_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            completion_prob = probabilities[0, 1].item()  # Probability of class 1 (Complete)
            prediction = 1 if completion_prob > 0.5 else 0

        print(f"End of turn probability: {completion_prob:.4f}")
        return {
            "prediction": prediction,
            "probability": completion_prob,
        }
