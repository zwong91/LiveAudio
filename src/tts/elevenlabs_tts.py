import io
from io import BytesIO
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator

import os
from .tts_interface import TTSInterface

from src.utils.audio_utils import wave_header_chunk
import langid

from pydub import AudioSegment

import tempfile
import requests

class ElevenlabTTS(TTSInterface):
    def __init__(self, model_id="eleven_turbo_v2", voice_id="IKne3meq5aSn9XLyUdCD", api_key=""):
        self.model_id = model_id
        self.voice_id = voice_id
        if api_key == "":
            load_dotenv()
            api_key = os.getenv("ELEVENLABS_API_KEY")
        self.api_key = api_key

        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


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
        language = langid.classify(text)[0].strip()
        if language == 'zh':
            language = 'zh-CN'

        #1. send talking audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.talking_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        temp_file_path = tempfile.gettempdir()
        file_path = os.path.join(temp_file_path, f"audio_{uuid4().hex[:8]}.wav")

        #2. stream synthesize audio
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}?optimize_streaming_latency=4"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": f"{self.api_key}",
        }

        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }

        response = requests.post(url, json=data, headers=headers)
        audio: AudioSegment = AudioSegment.from_file(
            io.BytesIO(response.content), format="mp3"
        )
        #samples = np.array(audio.get_array_of_samples())
        # 处理音频，重采样到16kHz，单声道，16bit
        audio_resampled = (
            audio.set_frame_rate(16000)
                .set_channels(1)
                .set_sample_width(2)  # 16bit sample_width 16/8=2  16k-mono-mp3
        )
        pcm_data_16K = audio_resampled.raw_data
        yield pcm_data_16K

        #3. send silent audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield pcm_data_16K
