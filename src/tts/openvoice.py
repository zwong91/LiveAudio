import torch
import torchaudio
import asyncio
import os
from io import BytesIO
import sys
import time
from uuid import uuid4
from typing import Tuple
from .tts_interface import TTSInterface
import numpy as np
import langid
import glob


sys.path.insert(1, "../vc")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "OpenVoice")))

from OpenVoice.openvoice import se_extractor
from OpenVoice.openvoice.api import ToneColorConverter
 
from melo.api import TTS


class OpenVoice_v2(TTSInterface):
    def __init__(self, voice: str = ''):
        self.voice = voice
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        ckpt_converter = 'checkpoints_v2/converter'
        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    async def text_to_speech(self, text: str, vc_uid: str, gen_file: bool) -> Tuple[bytes, str]:
        audio_buffer = BytesIO()
        """使用 x_tts 库将文本转语音"""
        start_time = time.time()
        language, _ = langid.classify(text)
        language = language.upper()
        src_path = f"/tmp/audio_{uuid4().hex[:8]}.wav"

        # 使用 os.path 确保路径正确拼接
        target_wav_pattern = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), f"{vc_uid}*.wav")
        target_wav_files = glob.glob(target_wav_pattern)  # 使用 glob 扩展通配符
        target_wav = target_wav_files[0] if target_wav_files else os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../rt-audio/vc")), "liuyifei.wav")
        output_path = f"/asset/audio_{uuid4().hex[:8]}.wav"

        print(f"Target wav files:{target_wav}, Detected language: {language}, tts text: {text}")
        
        reference_speaker = target_wav # This is the voice you want to clone
        target_se, audio_name = se_extractor.get_se(reference_speaker, self.tone_color_converter, vad=False)

        model = TTS(language=language, device=self.device)
        speaker_ids = model.hps.data.spk2id
    
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
    
            source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=self.device)
            model.tts_to_file(text, speaker_id, src_path, speed=1.0)

            # Run the tone color converter
            encode_message = "@MyShell"
            self.tone_color_converter.convert(
                audio_src_path=src_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
                message=encode_message)

        # 将生成的音频文件读入内存缓冲区
        with open(output_path, 'rb') as f:
            audio_buffer.write(f.read())

        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        end_time = time.time()
        print(f"OpenVice-V2 text_to_speech time: {end_time - start_time:.4f} seconds")

        return audio_data, output_path

