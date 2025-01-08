import io
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator
import os
from .tts_interface import TTSInterface

from src.utils.audio_utils import wave_header_chunk

import openai

from dotenv import load_dotenv
# Load environment variables
load_dotenv(override=True)

from pydub import AudioSegment

BASE_URL = os.getenv('BASE_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

voices_list = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer"
]

class OpenAITTS(TTSInterface):
    def __init__(self, voice: str = 'alloy', model: str = "tts-1", speed: float | None = None):
        self.voice = voice
        self.speed = speed
        self.model = model
        self.aclient = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 22050, #openai (22050)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
        """使用 openai_tts 库将文本转语音"""
        pass

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        #1. send talking audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.talking_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        extra_args = {"speed": self.speed} if self.speed is not None else {}
        
        #2. stream synthesize audio
        first_chunk = True
        async with self.aclient.with_streaming_response.audio.speech.create(
            model=self.model,
            input=text,
            voice=self.voice,
            response_format="mp3",
            **extra_args,
        ) as resp:
            async for chunk in resp.iter_bytes():
                if first_chunk:
                    first_chunk = False
                    print(f"First chunk Time elapsed: {time.time() - start_time:.2f} seconds")         
                # 使用 BytesIO 来读取音频数据
                with io.BytesIO(chunk) as f:
                    f.seek(0)
                    audio: AudioSegment = AudioSegment.from_file(f)
                    # 处理音频，重采样到16kHz，单声道，16bit
                    audio_resampled = (
                        audio.set_frame_rate(16000)
                            .set_channels(1)
                            .set_sample_width(2)  # 16bit sample_width (16/8=2)
                    )
                    pcm_data_16K = audio_resampled.raw_data
                    # 使用 wave_header_chunk 发送处理后的数据
                    yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
                    
        #3. send silent audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000) 