#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import numpy as np
import logging

from .filter_interface import FilterInterface

try:
    import noisereduce as nr
except ModuleNotFoundError as e:
    logging.error(f"Exception: {e}")
    raise Exception(f"Missing module: {e}")


class NoisereduceFilter(FilterInterface):
    def __init__(self) -> None:
        self._filtering = True
        self._sample_rate = 0


    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering:
            return audio

        data = np.frombuffer(audio, dtype=np.int16)

        # Add a small epsilon to avoid division by zero.
        epsilon = 1e-10
        data = data.astype(np.float32) + epsilon

        # Noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=self._sample_rate)
        audio = np.clip(reduced_noise, -32768, 32767).astype(np.int16).tobytes()

        return audio
