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
import pyttsx3

class SystemTTS(TTSInterface):
    def __init__(self, voice: str = "Zira", print_installed_voices: bool = False):
        self.engine = pyttsx3.init()
        installed_voices = self.engine.getProperty("voices")
        if voice is not None:
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
                    self.engine.setProperty("voice", installed_voice.id)

        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 22050, #system (22050)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
        """使用 pyttsx3 库将文本转语音"""
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
        """VOICE"""
        voices = self.engine.getProperty('voices')      #getting details of current voice
        #engine.setProperty('voice', voices[0].id)      #changing index, changes voices. o for male
        self.engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

        self.engine.save_to_file(text, file_path)
        self.engine.runAndWait()

        # Get media info of the file
        info = mediainfo(file_path)

        # Check if the file format is AIFF and convert to WAV if necessary
        if info["format_name"] == "aiff":
            audio = AudioSegment.from_file(self.file_path, format="aiff")
            audio.export(self.file_path, format="wav")

        audio: AudioSegment = AudioSegment.from_file(file_path, format="wav")
        # 处理音频，重采样到16kHz，单声道，16bit
        audio_resampled = (
            audio.set_frame_rate(16000)
                .set_channels(1)
                .set_sample_width(2)  # 16bit sample_width 16/8=2  16k-mono-mp3
        )
        pcm_data_16K = audio_resampled.raw_data
        yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        #3. send silent audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)