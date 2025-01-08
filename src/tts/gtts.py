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

from gtts import gTTS
import gtts.lang

class GTTS(TTSInterface):
    def __init__(self, tld: str = "com", speed: float | None = 1.0):
        self.lang = "mul"
        self.tld = tld
        self.chunk_length = 100
        self.crossfade_length = 10
        self.speed = speed
        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 22050, #google (22050)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
        """使用 gtts 库将文本转语音"""
        pass

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
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

        voices = []
        languages = gtts.lang.tts_langs()
        tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]

        for lang in languages.keys():
            for tld in tlds:
                print(f"GTTSVoice supported language: {lang}, tld: {tld}")

        file_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        print(f"Detected language: {language}")
        #2. Generate audio with gTTS
        with io.BytesIO() as f:
            tts = gTTS(text=text, lang=language, tld=self.tld, slow=False)   
            # Create an in-memory buffer for audio data
            audio_data = BytesIO()
            # Write the generated audio to the buffer
            tts.write_to_fp(audio_data)
            # Reset buffer to the start
            audio_data.seek(0)
            # Return the audio data
            yield audio_data

            # tts.write_to_fp(f)
            # f.seek(0)

            # audio = AudioSegment.from_file(f, format="mp3")

            # if self.speed != 1.0:
            #     audio = audio.speedup(
            #         playback_speed=self.speed,
            #         chunk_size=self.chunk_length,
            #         crossfade=self.crossfade_length,
            #     )

            # audio.export(file_path, format="wav")

        #3. send silent audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000) 