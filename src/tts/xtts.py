import torch
import torchaudio
import asyncio
import os
from io import BytesIO
import sys
import time
import logging
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

import numpy as np

import langid

import glob

sys.path.insert(1, "../vc")

import edge_tts
from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

from src.xtts.TTS.utils.generic_utils import get_user_data_dir
from src.xtts.TTS.utils.manage import ModelManager

class XTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to(device)

    async def text_to_speech(self, text: str, vc_uid: str, gen_file: bool) -> Tuple[bytes, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()
        language, _ = langid.classify(text)
        source_wav = f"/tmp/audio_{uuid4().hex[:8]}.wav"    
        communicate = edge_tts.Communicate(text=text, voice=self.voice)
        await communicate.save(source_wav)

        # 使用 os.path 确保路径正确拼接
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符
        target_wav = target_wav_files[0] if target_wav_files else os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "liuyifei.wav")
        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        print(f"Target wav files:{target_wav}, Detected language: {language}, tts text: {text}")
        tts_task = asyncio.create_task(
            asyncio.to_thread(
                self.tts.voice_conversion_to_file,
                source_wav=source_wav,
                target_wav=target_wav,
                file_path=output_path
            )
        )
        await tts_task

        # 将生成的音频文件读入内存缓冲区
        with open(output_path, 'rb') as f:
            audio_buffer.write(f.read())

        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        end_time = time.time()
        print(f"XTTS text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, output_path

class XTTS_v2(TTSInterface):
    def __init__(self, voice: str = 'liuyifei'):
        device = "cuda"
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "liuyifei.wav")
        # print("Loading model...")
        # config = XttsConfig()
        # config.load_json("XTTS-v2/config.json")
        # self.model = Xtts.init_from_config(config)
        # self.model.load_checkpoint(config, checkpoint_dir="XTTS-v2", use_deepspeed=True)
        # self.model.to(device)
        
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logging.info("⏳Downloading model")
        ModelManager().download_model(model_name)
        model_path = os.path.join(
            get_user_data_dir("tts"), model_name.replace("/", "--")
        )

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=True)
        self.model.to(device)

        print("Computing speaker latents...")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[target_wav])
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding

    async def text_to_speech(self, text: str, vc_uid: str, gen_file: bool) -> Tuple[bytes, str]: 
        start_time = time.time()
        language, _ = langid.classify(text)
        if language == 'zh':
            language = 'zh-cn'
        # 构造目标路径，获取匹配的 .wav 文件
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符

        if not target_wav_files:
            target_wav_files = [os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "dayang*.wav")]
            print(f"No WAV files found matching pattern, use default: {target_wav_files}")

        print("Computing speaker latents...")

        # 调用模型函数，传递匹配的文件列表
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=target_wav_files)
        print(f"Target wav files:{target_wav_files}, Detected language: {language}, tts text: {text}")

        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            # Streaming
            stream_chunk_size=256,
            overlap_wav_len=1024,
            # GPT inference
            temperature=0.01,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=3,
            top_p=0.97,
            do_sample=True,
            speed=1.0,
            enable_text_splitting=True,
        )
        wav_chunks = []
        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"
        for i, chunk in enumerate(chunks):
            wav_chunks.append(chunk)

        wav = torch.cat(wav_chunks, dim=0)
        wav_audio = wav.squeeze().unsqueeze(0).cpu()

        # Saving to a file on disk
        if gen_file:
            torchaudio.save(output_path, wav_audio, 22050, format="wav")

        with torch.no_grad():
            # Use torchaudio to save the tensor to a buffer (or file)
            # Using a buffer to save the audio data as bytes
            buffer = BytesIO()
            torchaudio.save(buffer, wav_audio, 22050, format="wav")  # Adjust sample rate if needed
            audio_data = buffer.getvalue()

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return audio_data, output_path
