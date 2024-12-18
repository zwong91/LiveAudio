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
import base64

sys.path.insert(1, "../vc")

import edge_tts
from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

from src.xtts.TTS.utils.generic_utils import get_user_data_dir
from src.xtts.TTS.utils.manage import ModelManager

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
        
        # 缓存 gpt_cond_latent 和 speaker_embedding
        self.latent_cache = {}

    def get_cached_latents(self, vc_uid: str, target_wav_files: list):
        if isinstance(target_wav_files, str):
            target_wav_files = [target_wav_files]
        # 使用 vc_uid 和 文件MD5 作为缓存的键
        # 获取文件名并进行 Base64 编码
        filenames = ''.join(os.path.basename(f) for f in target_wav_files)
        bs64 = base64.b64encode(filenames.encode('utf-8')).decode('utf-8')
        cache_key = f"{vc_uid}_{bs64}"
        if cache_key in self.latent_cache:
            print(f"Cache hit for {vc_uid} with encoded file names {bs64}")
            return self.latent_cache[cache_key]
        else:
            print(f"Cache miss for {vc_uid} with encoded file names {bs64}")
            # 计算并返回新的 latents 和 speaker_embedding
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=target_wav_files)
            # 缓存结果
            self.latent_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
            return gpt_cond_latent, speaker_embedding

    async def text_to_speech(self, text: str, vc_uid: str) -> Tuple[str]: 
        start_time = time.time()
        language, _ = langid.classify(text)
        if language == 'zh':
            language = 'zh-cn'
        # 构造目标路径，获取匹配的 .wav 文件
        supported_extensions = ["wav", "m4a", "flac", "mp3"]

        # 初始化匹配的文件列表
        target_wav_files = []

        # 遍历支持的扩展名进行匹配
        for ext in supported_extensions:
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), f"{vc_uid}*.{ext}")
            target_wav_files.extend(glob.glob(target_wav_pattern))

        if not target_wav_files:
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "dayang.wav")
            target_wav_files = glob.glob(target_wav_pattern)
            print(f"No WAV files found matching pattern, use default: {target_wav_files}")
        else:
             target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../tts-tools/output/vc_uvr5_result")), f"vocal_{vc_uid}*10.wav")
             pure_target_wav_files = glob.glob(target_wav_pattern)  # vc_uvr5_result
             if pure_target_wav_files:
                    target_wav_files = pure_target_wav_files
             else:
                target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../tts-tools/output/uvr5_opt")), f"{vc_uid}*.wav_main_vocal.wav")
                pure_target_wav_files = glob.glob(target_wav_pattern)  # uvr5_opt 没有echo
                if pure_target_wav_files:
                    target_wav_files = pure_target_wav_files

        print("Computing speaker latents...")

        # 调用模型函数，传递匹配的文件列表
        gpt_cond_latent, speaker_embedding = self.get_cached_latents(vc_uid, target_wav_files)
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
        torchaudio.save(output_path, wav_audio, 22050, format="wav")

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return output_path


    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    async def text_to_speech_stream(self, text: str, vc_uid: str) -> Tuple[bytes]: 
        start_time = time.time()
        language, _ = langid.classify(text)
        if language == 'zh':
            language = 'zh-cn'
        # 构造目标路径，获取匹配的 .wav 文件
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符

        if not target_wav_files:
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "dayang.wav")
            target_wav_files = glob.glob(target_wav_pattern)
            print(f"No WAV files found matching pattern, use default: {target_wav_files}")
        else:
             target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../tts-tools/output/vc_uvr5_result")), f"vocal_{vc_uid}*.wav")
             pure_target_wav_files = glob.glob(target_wav_pattern)  # vc_uvr5_result
             if pure_target_wav_files:
                    target_wav_files = pure_target_wav_files
             else:
                target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../tts-tools/output/uvr5_opt")), f"{vc_uid}*.wav_main_vocal.wav")
                pure_target_wav_files = glob.glob(target_wav_pattern)  # uvr5_opt 没有echo
                if pure_target_wav_files:
                    target_wav_files = pure_target_wav_files

        print("Computing speaker latents...")

        # 调用模型函数，传递匹配的文件列表
        gpt_cond_latent, speaker_embedding = self.get_cached_latents(vc_uid, target_wav_files)
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

        # for i, chunk in enumerate(chunks):
        #     processed_chunk = self.wav_postprocess(chunk)
        #     processed_bytes = processed_chunk.tobytes()
        #     print(f"XTTS-v2 音频chunk大小: {len(processed_bytes)} 字节")
        #     yield processed_bytes
        wav_chunks = []
        for i, chunk in enumerate(chunks):
            wav_chunks.append(chunk)

        wav = torch.cat(wav_chunks, dim=0)
        wav_audio = wav.squeeze().unsqueeze(0)

        with torch.no_grad():
            # Use torchaudio to save the tensor to a buffer (or file)
            # Using a buffer to save the audio data as bytes
            buffer = BytesIO()
            torchaudio.save(buffer, wav_audio, 22050, format="wav")  # Adjust sample rate if needed
            buffer.seek(0)
            audio_data = buffer.read()

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return audio_data
