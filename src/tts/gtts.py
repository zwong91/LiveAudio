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


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 22050, #google (22050)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> Tuple[str]:
        """使用 gtts 库将文本转语音"""
        pass

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        language = langid.classify(text)[0].strip()
        if language == 'zh':
            language = 'zh-CN'

        voices = []
        languages = gtts.lang.tts_langs()
        tlds = ["com", "com.au", "co.uk", "us", "ca", "co.in", "ie", "co.za"]

        # for lang in languages.keys():
        #     for tld in tlds:
        #         print(f"GTTSVoice supported language: {lang}, tld: {tld}")

        file_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        #stream synthesize audio
        # with io.BytesIO() as f:
        #     for i, chunk in enumerate(gTTS(text=text, lang=language, tld=self.tld, slow=False).stream()):
        #         f.write(chunk)
        #     # 将 BytesIO 中的数据重置指针，并加载为 AudioSegment
        #     f.seek(0)
        #     audio: AudioSegment = AudioSegment.from_mp3(f)
        #     if self.speed != 1.0:
        #         audio = audio.speedup(
        #             playback_speed=self.speed,
        #             chunk_size=self.chunk_length,
        #             crossfade=self.crossfade_length,
        #         )
        #     #audio.export(file_path, format="wav")

        #     # 处理音频，重采样到16kHz，单声道，16bit
        #     audio_resampled = (
        #         audio.set_frame_rate(16000)
        #             .set_channels(1)
        #             .set_sample_width(2)  # 16bit sample_width 16/8=2  16k-mono-mp3
        #     )
        #     pcm_data_16K = audio_resampled.raw_data
        #     yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        #Generate audio with gTTS stream
        for i, chunk in enumerate(gTTS(text=text, lang=language, tld=self.tld, slow=False).stream()):
            if i == 0:
                print(f"First chunk Time elapsed: {time.time() - start_time:.2f} seconds")
            # 使用 BytesIO 来读取音频数据
            with io.BytesIO(chunk) as audio_io:
                audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
                # 处理音频，重采样到22050Hz，单声道，16bit
                audio_resampled = (
                    audio.set_frame_rate(16000)
                        .set_channels(1)
                        .set_sample_width(2)  # 16bit sample_width (16/8=2)
                )
                pcm_data_16K = audio_resampled.raw_data
                # 发送处理后的数据
                yield pcm_data_16K
