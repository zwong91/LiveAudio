import numpy as np
import torch
import logging
from .filter_interface import FilterInterface

logger = logging.getLogger(__name__)

try:
    from df.enhance import enhance, init_df, load_audio
except ModuleNotFoundError as e:
    logger.error(f"Missing deepfilter: {e}")
    raise Exception("Please install deepfilterlib: pip install deepfilternet")


class DeepNetFilter(FilterInterface):
    def __init__(self, **kwargs):
        self._filtering = True

        try:
            # Initialize model
            self.model, self.df_state, _ = init_df()
        except Exception as e:
            logger.error(f"Failed to load DeepNetFilter: {e}")
            self._filtering = False

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering:
            return audio

        try:
            # Convert to float32 numpy array
            data = np.frombuffer(audio, dtype=np.int16)
            audio_float = data.astype(np.float32) / 32768.0

            audio_tensor = torch.from_numpy(audio_float).float()
            audio_tensor = audio_tensor.unsqueeze(0)
            # Process audio
            enhanced = enhance(
                self.model,
                self.df_state,
                audio_tensor,
                atten_lim_db=6, #噪声衰减限制（attenuation limit）
            )

            # Convert back to int16
            processed = (enhanced.squeeze(0).numpy() * 32768.0).astype(np.int16)
            return processed.tobytes()

        except Exception as e:
            logger.error(f"DeepNetFilter processing failed: {e}")
            return audio
