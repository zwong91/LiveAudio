import io
from io import BytesIO
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator

import os
from .tts_interface import TTSInterface

import soundfile as sf
import langid

from pydub import AudioSegment
from dotenv import load_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs

load_dotenv()

class ElevenlabTTS(TTSInterface):
    def __init__(self, model_id="eleven_flash_v2_5", voice_id="hkfHEbBvdQFNX4uWHqRF"):
        self.model_id = model_id
        self.voice_id = voice_id

        api_key = os.getenv("ELEVENLABS_API_KEY")
        self.api_key = api_key
        self.client = ElevenLabs(
            api_key=api_key,
        )

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 44100, #elevenlabs (44100)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> Tuple[str]:
        """使用 elevenlabs 库将文本转语音"""
        pass

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        first_chunk = True
        # response = await self.client.voices.get_all()
        # print(response.voices)

        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="pcm_16000",
        )

        for chunk in audio_stream:
            if isinstance(chunk, bytes):
                if first_chunk:
                    time_to_first_chunk = time.time() - start_time
                    print(f"Time to first chunk: {time_to_first_chunk:.4f}s")
                    first_chunk = False

                yield chunk

        print(f"ElevenLabs TTS time: {time.time() - start_time:.4f}s")
