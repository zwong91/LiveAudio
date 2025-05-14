import torch
import torchaudio
import asyncio
import os
from typing import Optional, AsyncGenerator
import sys
import time
import io
import logging
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

from pydub import AudioSegment
import numpy as np

import langid
import glob
import base64

sys.path.insert(1, "../assets")

# coqui-tts 0.26.0
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from trainer.io import get_user_data_dir
from TTS.utils.manage import ModelManager

from src.utils.audio_utils import postprocess_tts_wave_int16, convertSampleRateTo16khz, wave_header_chunk

class XTTS_v2(TTSInterface):
    def __init__(self, voice: str = 'liuyifei'):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "assets")), "liuyifei.wav")

        self.ov = TTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2").to(device)
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
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=target_wav_files,
                gpt_cond_len=140,      #音频秒数
                gpt_cond_chunk_len=6,  #音频被分割成块,音频块大小（秒）
                max_ref_length=120)    #decoder最大参考音频秒数

            self.latent_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
            return gpt_cond_latent, speaker_embedding

    def get_stream_info(self) -> dict:
        return {
            "format": 1, # PYAUDIO_PAFLOAT32
            "channels": 1,
            "sample_rate": self.config.audio.output_sample_rate,  #coqui (24000)
            "sample_width": 4,
            "np_dtype": np.float32,
        }

    def normalize_language_code(self, language: str) -> str:
        # 语言代码映射
        LANG_MAP = {
            'en': 'en-newest',
            'zh': 'zh',
            'ko': 'kr',
            'ja': 'jp',
            'fr': 'fr',
            'es': 'es'
        }
        return LANG_MAP.get(language.lower(), 'en-newest')  # 默认返回EN

    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> Tuple[str]:
        """ Coqui TTS engine's inability to handle multiple synthesis requests in parallel
        voice clone worked: a 22050 Hz mono 16bit WAV file containing a short (~5-30 sec) sample
        Args:
            text (str): _description_
            vc_uid (str): _description_
            speed (Optional[float], optional): _description_. Defaults to None.

        Returns:
            Tuple[str]: _description_
        """
        start_time = time.time()

        v_speed = 1.0 if speed is None else float(speed)
        v_speed = max(0.5, min(2.0, v_speed))

        language = langid.classify(text)[0].strip()
        ov_ses_lang = self.normalize_language_code(language)
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
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "assets")), f"{vc_uid}*.{ext}")
            target_wav_files.extend(glob.glob(target_wav_pattern))

        if not target_wav_files:
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "assets")), "dayang.wav")
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
        print(f"Target wav files:{target_wav_files}, Detected language: {language}, tts text: {text}, speed: {v_speed}")

        t0 = time.time()
        out = self.model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            # 克隆声音最佳参数
            temperature=0.4,      # 降低随机性，保持声音特征
            length_penalty=1.2,    # 保持自然长度
            repetition_penalty=5.0, # 适度控制重复
            do_sample=False,       # 关闭采样提高稳定性
            top_k=50,             # 限制采样范围
            top_p=0.85,           # 核采样阈值
            speed=v_speed,          # 默认是1 normal speed
            enable_text_splitting=True
        )
        src_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        inference_time = time.time() - t0
        print(f"Time to generate audio: {inference_time} seconds")
        real_time_factor= (time.time() - t0) / out['wav'].shape[-1] * 24000
        print(f"Real-time factor (RTF): {real_time_factor}")

        torchaudio.save(src_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        end_time = time.time()
        print(f"XTTSv2 text_to_speech time: {end_time - start_time:.4f} seconds")

        # TODO: assets voice conversion OpenVoiceV2
        #t0 = time.time()
        #save_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        # self.ov.voice_conversion_to_file(
        #     source_wav=source_wav,
        #     target_wav=target_wav_files,
        #     file_path=save_path
        # )
        #print(f"OpenVoice v2 voice conversion time: {time.time() - t0:.4f} seconds")
        return src_path

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        language = langid.classify(text)[0].strip()
        if language == 'zh':
            language = 'zh-cn'
        if language not in self.supported_languages:
            print(f"Language you put {language} in is not in our Supported Languages, please choose from {self.supported_languages}")

        # 构造目标路径，获取匹配的 .wav 文件
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "assets")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符

        if not target_wav_files:
            target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "assets")), "dayang.wav")
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
        generated_seconds = 0.0
        wav_chunks = []
        chunks = self.model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            # Streaming reduce it to get faster response, but degrade quality
            stream_chunk_size=40,
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

        #stream synthesize audio
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0} s")
            wav_chunks.append(chunk)
            processed_bytes = postprocess_tts_wave_int16(chunk)
            chunk_duration = len(processed_bytes) / (
                4 * 24000
            )  # 4 bytes per sample, 24000 Hz
            generated_seconds += chunk_duration
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}, chunk duration: {chunk_duration}")
            pcm_data_16K = convertSampleRateTo16khz(processed_bytes, self.config.audio.output_sample_rate)
            # such as chunk size 9600, (a.k.a 24K*20ms*2)
            print(f"XTTS-v2 audio chunk size: {len(pcm_data_16K)} 字节")
            yield pcm_data_16K

        wav = torch.cat(wav_chunks, dim=0)
        #real_time_factor= (time.time() - t0) / generated_seconds
        real_time_factor= (time.time() - t0) / wav.shape[0] * 24000 ## 4 bytes per sample, 24000 Hz
        print(f"wav.shape {wav.shape}, Real-time factor (RTF): {real_time_factor}")
