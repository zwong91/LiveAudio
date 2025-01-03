import torch
import torchaudio
import asyncio
import os
from typing import AsyncGenerator
import sys
import time
import io
import re
import logging
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface

import numpy as np

import langid
import glob
import base64

import sys
sys.path.append('../synthesize/cosyvoice/cosyvoice')
sys.path.append('../synthesize/cosyvoice/third_party/AcademiCodec')
sys.path.append('../synthesize/cosyvoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

from src.utils.audio_utils import postprocess_tts_wave_int16, convertSampleRateTo16khz, wave_header_chunk

class CosyVoice_v2(TTSInterface):
    def __init__(self, voice: str = '中文女'):
        device = "cuda"
        # 使用 os.path 确保路径正确拼接
        target_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "liuyifei.wav")
        model_name = "iic/CosyVoice2-0.5B"
        self.cosyvoice = CosyVoice2(model_name, load_jit=False, load_trt=False, fp16=False)

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

        print(f"Target wav files:{target_wav_files}, Detected language: {language}, tts text: {text}")
        
        t0 = time.time()
        pattern = r"生成风格:\s*([^\n;]+)[;\n]+播报内容:\s*(.+)"
        match = re.search(pattern, text)
        if match:
            style = match.group(1).strip()
            content = match.group(2).strip()
            tts_text = f"{style}<endofprompt>{content}"
            print(f"生成风格: {style}")
            print(f"播报内容: {content}")
        else:
            print("No match found")
            tts_text = text

        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        prompt_speech_16k = load_wav(target_wav_files[0], 16000)
        for i, j in enumerate(self.cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=False)):
            torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], self.cosyvoice.sample_rate)

        wav = torch.cat(wav_chunks, dim=0)
        real_time_factor= (time.time() - t0) / wav.shape[0] * 24000
        print(f"wav.shape {wav.shape}, Real-time factor (RTF): {real_time_factor}")
        wav_audio = wav.squeeze().unsqueeze(0).cpu()

        end_time = time.time()
        print(f"CosyVoice v2 text_to_speech time: {end_time - start_time:.4f} seconds")
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

        print(f"Target wav files:{target_wav_files}, Detected language: {language}, tts text: {text}")

        t0 = time.time()
        wav_chunks = []
        processed_tts_text = ""
        punctuation_pattern = r'([!?;。！？])'
        parts = re.split(punctuation_pattern, tts_text)
        if len(parts) > 2 and parts[-1]:
            tts_text = "".join(parts[:-1])
        processed_tts_text += tts_text
        print(f"cur_tts_text: {tts_text}")
        pattern = r"生成风格:\s*([^\n;]+)[;\n]+播报内容:\s*(.+)"
        match = re.search(pattern, text)
        if match:
            style = match.group(1).strip()
            content = match.group(2).strip()
            tts_text = f"{style}<endofprompt>{content}"
            print(f"生成风格: {style}")
            print(f"播报内容: {content}")
        else:
            print("No match found")
            tts_text = text

        text_list = [tts_text]
        for i in text_list:
            output_generator = cosyvoice.inference_sft(i, speaker_name)
            for output in output_generator:
                yield (22500, output['tts_speech'].numpy().flatten())
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                print(f"Time to first chunck: {time.time() - t0} s")
            print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
            wav_chunks.append(chunk)
            processed_bytes = postprocess_tts_wave_int16(chunk)
            pcm_data_16K = convertSampleRateTo16khz(processed_bytes, self.config.audio.output_sample_rate)
            # such as chunk size 9600, (a.k.a 24K*20ms*2)
            print(f"CosyVoice-v2 audio chunk size: {len(pcm_data_16K)} 字节")
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
            
        wav = torch.cat(wav_chunks, dim=0)
        real_time_factor= (time.time() - t0) / wav.shape[0] * 24000
        print(f"wav.shape {wav.shape}, Real-time factor (RTF): {real_time_factor}")

