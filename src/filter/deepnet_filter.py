import numpy as np
import logging
from .filter_interface import FilterInterface

logger = logging.getLogger(__name__)

try:
    from deepfilternet.enhance import enhance, init_df, load_audio
except ModuleNotFoundError as e:
    logger.error(f"Missing deepfilterlib: {e}")
    raise Exception("Please install deepfilterlib: pip install deepfilternet")


class DeepNetFilter(FilterInterface):
    def __init__(self, **kwargs):
        self._sample_rate = 16000
        self._filtering = True

        try:
            # Initialize model
            self.df_state = init_df()
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

            # Process audio
            enhanced = enhance(
                audio_float,
                self.df_state,
                sr=self._sample_rate,
                atten_lim_db=6,
            )

            # Convert back to int16
            processed = (enhanced * 32768.0).astype(np.int16)
            return processed.tobytes()

        except Exception as e:
            logger.error(f"DeepNetFilter processing failed: {e}")
            return audio
