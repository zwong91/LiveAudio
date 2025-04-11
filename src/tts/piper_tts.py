import io
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator
import os
import langid

from piper import PiperVoice
from .tts_interface import TTSInterface

class PiperTTS(TTSInterface):
    def __init__(self, model_path: str = "voice-en-us-ryan-low"):
        self.voice = PiperVoice.load(model_path)

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 22050,
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> Tuple[str]:
        start_time = time.time()
        v_speed = 1.0 if speed is None else float(speed)
        v_speed = max(0.5, min(2.0, v_speed))

        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"
        wav_data = self.voice.synthesize(text, speed=v_speed)

        with open(output_path, "wb") as f:
            f.write(wav_data)

        end_time = time.time()
        print(f"PiperTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        return output_path

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()

        chunk_size = 1024 * 4  # 4KB chunks
        total_data = self.voice.synthesize(text)

        position = 0
        first_chunk = True

        while position < len(total_data):
            chunk = total_data[position:position + chunk_size]
            if first_chunk:
                print(f"First chunk time: {time.time() - start_time:.2f} seconds")
                first_chunk = False
            yield chunk
            position += chunk_size
