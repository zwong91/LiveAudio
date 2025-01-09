import io
from io import BytesIO
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator

import os
from .tts_interface import TTSInterface

from src.utils.audio_utils import wave_header_chunk
import langid

import torch
from pydub import AudioSegment
from scipy.io.wavfile import write

class ParlerTTS(TTSInterface):
    def __init__(self, model_id="parler-tts/parler_tts_mini_v0.1", tts_description=None, temperature=1.0,):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        from transformers.modeling_outputs import BaseModelOutput

        if tts_description is None:
            tts_description = (
                "A female speaker with a slightly low-pitched voice delivers her words quite "
                "expressively, in a very confined sounding environment with clear audio quality."
            )
            
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(
            device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.tts_description = tts_description
        input_ids = self.tokenizer(tts_description, return_tensors="pt").input_ids.to(
            device
        )
        self.desc_tensor = BaseModelOutput(
            last_hidden_state=self.model.text_encoder(
                input_ids=input_ids
            ).last_hidden_state
        )
        self.temperature = temperature

        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 44100, #self.model.config.sampling_rate
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
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
        prompt_input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        generation = self.model.generate(
            encoder_outputs=self.desc_tensor,
            prompt_input_ids=prompt_input_ids,
            temperature=self.temperature,
        )
        audio_arr = generation.cpu().numpy().squeeze()
        sample_rate = self.model.config.sampling_rate

        # 创建一个 BytesIO 对象存储 WAV 文件
        wav_buffer = io.BytesIO()
        write(wav_buffer, sample_rate, audio_arr.astype("int16"))
        wav_buffer.seek(0)

        audio: AudioSegment = AudioSegment.from_file(
            wav_buffer, format="wav"
        )
        #samples = np.array(audio.get_array_of_samples())
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