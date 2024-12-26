import torch
import torchaudio
import asyncio
import os
from typing import AsyncGenerator
import sys
import time
import io
import logging
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

import numpy as np

import langid
import glob
import base64

sys.path.insert(1, "../vc")

from src.xtts.TTS.api import TTS
from src.xtts.TTS.tts.configs.xtts_config import XttsConfig    
from src.xtts.TTS.tts.models.xtts import Xtts

from src.xtts.TTS.utils.generic_utils import get_user_data_dir
from src.xtts.TTS.utils.manage import ModelManager

from src.utils.audio_utils import postprocess_tts_wave_int16, convertSampleRateTo16khz, wave_header_chunk

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
        
        model_million_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logging.debug(f"{model_million_params}M parameters")

        self.supported_languages = config.languages
        self.config = config
        print("Computing speaker latents...")
        t_latent = time.time()
        ## note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=[target_wav], gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
        latent_calculation_time = time.time() - t_latent
        print(f"Embedding speaker latents computed in {latent_calculation_time:.4f} seconds")
        self.gpt_cond_latent = gpt_cond_latent
        self.speaker_embedding = speaker_embedding
        
        # 缓存 gpt_cond_latent 和 speaker_embedding
        self.latent_cache = {}

    def get_cached_latents(self, vc_uid: str, target_wav_files: list):
        if isinstance(target_wav_files, str):
            target_wav_files = [target_wav_files]
        # 使用 vc_uid 和 文件MD5 作为缓存的键
        # 获取文件名并进行 Base64 编码
        last_filename = os.path.basename(target_wav_files[-1])
        bs64 = base64.b64encode(last_filename.encode('utf-8')).decode('utf-8')
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

    def get_stream_info(self) -> dict:
        return {
            "format": 1, # PYAUDIO_PAFLOAT32
            "channels": 1,
            "sample_rate": self.config.audio.output_sample_rate,
            "sample_width": 4,
            "np_dtype": np.float32,
        }

    async def text_to_speech(self, text: str, vc_uid: str) -> Tuple[str]: 
        start_time = time.time()
        language = langid.classify(text)[0].strip()
        if language == 'zh':
            language = 'zh-cn'

        if language not in self.supported_languages:
            print(f"Language you put {language} in is not in our Supported Languages, please choose from {self.supported_languages}")
        
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
        
        t0 = time.time()
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
        real_time_factor= (time.time() - t0) / wav.shape[0] * 24000
        print(f"wav.shape {wav.shape}, Real-time factor (RTF): {real_time_factor}")
        wav_audio = wav.squeeze().unsqueeze(0).cpu()

        # Saving to a file on disk
        torchaudio.save(output_path, wav_audio, 22050, format="wav")

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")
        return output_path


    async def text_to_speech_stream(self, text: str, vc_uid: str) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        language = langid.classify(text)[0].strip()
        if language == 'zh':
            language = 'zh-cn'
        if language not in self.supported_languages:
            print(f"Language you put {language} in is not in our Supported Languages, please choose from {self.supported_languages}")

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

        t0 = time.time()
        wav_chunks = []
        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            # Streaming
            stream_chunk_size=80,
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

        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0} s")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            wav_chunks.append(chunk)
            processed_bytes = postprocess_tts_wave_int16(chunk)
            pcm_data_16K = convertSampleRateTo16khz(processed_bytes, self.config.audio.output_sample_rate)
            print(f"XTTS-v2 audio chunk size: {len(pcm_data_16K)} 字节")
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
            
        wav = torch.cat(wav_chunks, dim=0)
        real_time_factor= (time.time() - t0) / wav.shape[0] * 24000
        print(f"wav.shape {wav.shape}, Real-time factor (RTF): {real_time_factor}")

