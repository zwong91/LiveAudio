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

        voices = []
        languages = gtts.lang.tts_langs()
        tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]

        # for lang in languages.keys():
        #     for tld in tlds:
        #         print(f"GTTSVoice supported language: {lang}, tld: {tld}")

        file_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        #2. Generate audio with gTTS 
        for i, chunk in enumerate(gTTS(text=text, lang=language, tld=self.tld, slow=False).stream()):
            if i == 0:
                print(f"First chunk Time elapsed: {time.time() - start_time:.2f} seconds")            
            # 使用 BytesIO 来读取音频数据
            with io.BytesIO(chunk) as audio_io:
                audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
                # 处理音频，重采样到16kHz，单声道，16bit
                audio_resampled = (
                    audio.set_frame_rate(16000)
                        .set_channels(1)
                        .set_sample_width(2)  # 16bit sample_width (16/8=2)
                )
                pcm_data_16K = audio_resampled.raw_data
                # 使用 wave_header_chunk 发送处理后的数据
                yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)


        # #3. send silent audio
        # if not simultaneous:
        #     audio = AudioSegment.from_wav(self.silence_wav)
        #     # 重采样为 16kHz，单声道，16-bit
        #     audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        #     pcm_data_16K = audio_resampled.raw_data
        #     yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)