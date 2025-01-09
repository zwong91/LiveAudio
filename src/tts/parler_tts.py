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
import tempfile
import pyaudio
from pydub import AudioSegment
from scipy.io.wavfile import write

from threading import Thread
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

class ParlerTTS(TTSInterface):
    def __init__(self, model_id="parler-tts/parler-tts-mini-v1", voice_prompt=None, temperature=1.0):
        # Jon、Lea、Gary、Jenna、Mike、Laura ... speaker
        if voice_prompt is None:
            voice_prompt = (
                "Laura's voice is fast-paced with a slight rasp, recorded up close with no background noise."
                "Her words come out quickly, creating a sense of urgency and intensity."
            )
            # voice_prompt = (
            #     "John's voice is deep and smooth, with a calm and reassuring tone. He speaks slowly and clearly, with a slight southern accent."
            # )            
        
        self.play_steps_in_s = 0.5
        self.voice_parameters = {}
        self.buffer_duration_s = 1.0
        self.print_time_to_first_token = False
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(
            device, dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.voice_prompt = voice_prompt
        self.temperature = temperature

        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")


    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 44100, # Using 32-bit float, mono, 44100 Hz self.model.config.sampling_rate
            "sample_width": pyaudio.paFloat32,
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
        frame_rate = self.model.audio_encoder.config.frame_rate
        sampling_rate = self.model.audio_encoder.config.sampling_rate

        play_steps = int(frame_rate * self.play_steps_in_s)
        streamer = ParlerTTSStreamer(self.model, device=self.device, play_steps=play_steps)

        inputs = self.tokenizer(self.voice_prompt, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "prompt_input_ids": prompt.input_ids,
            "attention_mask": inputs.attention_mask,
            "prompt_attention_mask": prompt.attention_mask,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 1.0,
            "min_new_tokens": 10,
            **self.voice_parameters,  # Merge with any additional voice parameters
        }

        # Initialize variables for buffering
        audio_buffer = []
        buffer_length_s = 0.0
        generation_completed = False

        # Start the audio generation (blocking call)
        def generate_audio():
            self.model.generate(**generation_kwargs)

        # Start the generation in a separate thread
        generation_thread = Thread(target=generate_audio)
        generation_thread.start()

        # Process the streamer in the main thread
        while not generation_completed:
            try:
                new_audio = next(streamer)
                if new_audio.shape[0] == 0:
                    # Streamer signaled completion
                    generation_completed = True
                    break

                audio_chunk = new_audio
                audio_buffer.append(audio_chunk)
                buffer_length_s += new_audio.shape[0] / sampling_rate

                if buffer_length_s >= self.buffer_duration_s:
                    # Buffering complete, start streaming
                    break
            except StopIteration:
                # No more audio data
                generation_completed = True
                break

        # Queue the buffered audio chunks
        for buffered_chunk in audio_buffer:
            with io.BytesIO(buffered_chunk.tobytes()) as audio_io:
                audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
                # 处理音频，重采样到16KHz，单声道，16bit
                audio_resampled = (
                    audio.set_frame_rate(16000)
                        .set_channels(1)
                        .set_sample_width(2)  # 16bit sample_width (16/8=2)
                )
                pcm_data_16K = audio_resampled.raw_data
                # 使用 wave_header_chunk 发送处理后的数据
                yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        # Continue streaming the rest of the audio
        first_token = False
        while not generation_completed:
            try:
                new_audio = next(streamer)
                if new_audio.shape[0] == 0:
                    # Streamer signaled completion
                    generation_completed = True
                    break
                audio_chunk = new_audio
                if not first_token and self.print_time_to_first_token:
                    end_time = time.time()
                    print(f"Time to first token: {end_time - start_time:.2f} s")
                with io.BytesIO(audio_chunk.tobytes()) as audio_io:
                    audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
                    # 处理音频，重采样到16KHz，单声道，16bit
                    audio_resampled = (
                        audio.set_frame_rate(16000)
                            .set_channels(1)
                            .set_sample_width(2)  # 16bit sample_width (16/8=2)
                    )
                    pcm_data_16K = audio_resampled.raw_data
                    # 使用 wave_header_chunk 发送处理后的数据
                    yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
                first_token = True
            except StopIteration:
                generation_completed = True
                break

        # Ensure the generation thread has completed
        generation_thread.join()

        #3. send silent audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)