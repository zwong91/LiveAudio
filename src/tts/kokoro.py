from typing import AsyncGenerator, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import time
from uuid import uuid4
from io import BytesIO

from .tts_interface import TTSInterface
from src.utils.audio_utils import wave_header_chunk
from kokoro import KModel, KPipeline
from huggingface_hub import list_repo_files
from pydub import AudioSegment
from random import choice

class Kokoro(TTSInterface):
    REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
    SAMPLE_RATE = 24000
    N_ZEROS = 5000
    voice_files = list_repo_files(REPO_ID, repo_type="model")

    # 获取所有以 'zf_' 开头的女性语音模型文件
    female_voices = [file for file in voice_files if file.startswith("voices/zf_") and file.endswith(".pt")]

    # 获取所有以 'zm_' 开头的男性语音模型文件
    male_voices = [file for file in voice_files if file.startswith("voices/zm_") and file.endswith(".pt")]

    # 提取语音模型的名称（去除路径和扩展名）
    female_voice_names = [file.split("/")[-1].replace(".pt", "") for file in female_voices]
    male_voice_names = [file.split("/")[-1].replace(".pt", "") for file in male_voices]

    VOICES = {
        'female': female_voice_names,
        'male': male_voice_names
    }

    print(VOICES)
    ALL_VOICES = [v for voices in VOICES.values() for v in voices]


    def __init__(self, voice: str = "zf_002"):
        self.VOICE = voice if voice else choice(self.ALL_VOICES)
        print(f"Using voice: {self.VOICE}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        self.model = KModel(repo_id=self.REPO_ID).to(self.device).eval()
        self.en_pipeline = KPipeline(
            lang_code='a',
            repo_id=self.REPO_ID,
            model=False,
            device=self.device,
        )

        self.zh_pipeline = KPipeline(
            lang_code='z',
            repo_id=self.REPO_ID,
            model=self.model,
            device=self.device,
            en_callable=self._en_callable
        )

    def _en_callable(self, text: str) -> str:
        if text == 'Kokoro': return 'kˈOkəɹO'
        if text == 'Sol': return 'sˈOl'
        return next(self.en_pipeline(text)).phonemes

    def _speed_callable(self, len_ps: int) -> float:
        base_speed = 0.8
        if len_ps <= 83:
            return 1.1
        elif len_ps < 183:
            return (1 - (len_ps - 83) / 500) * 1.1
        return base_speed * 1.1

    def get_stream_info(self) -> Dict[str, int]:
        return {
            "sample_rate": 24000,
            "sample_width": 2,
            "channels": 1,
        }

    @torch.inference_mode()
    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> str:
        start_time = time.time()
        wavs = []
        path = Path("/asset")

        generator = self.zh_pipeline(text, voice=self.VOICE, speed=self._speed_callable)
        for i, (gs, ps, audio) in enumerate(generator):
            print(i)  # i => index
            print(gs) # gs => graphemes/text
            print(ps) # ps => phonemes
            wavs.append(audio) # 添加音频到列表
            sf.write(f'{i}.wav', audio, 24000)

        # 合并所有音频并保存
        final_path = path / f'HEARME_{vc_uid}_{uuid4().hex[:8]}.wav'
        final_audio = np.concatenate(wavs) if wavs else np.array([])
        sf.write(final_path, final_audio, self.SAMPLE_RATE)

        print(f"Kokoro TTS time: {time.time() - start_time:.4f}s")
        return str(final_path)

    @torch.inference_mode()
    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        first_chunk = True
        generator = self.zh_pipeline(text, voice=self.VOICE, speed=self._speed_callable)
        for data in generator:
            wav = data.audio
            # 转WAV字节流
            wav_io = BytesIO()
            sf.write(wav_io, wav, self.SAMPLE_RATE, format='WAV')
            wav_io.seek(0)
            # 重采样处理
            audio = AudioSegment.from_wav(wav_io)
            audio_resampled = (
                audio.set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)
            )

        if is_first_chunk:
            time_to_first_chunk = time.time() - start_time
            original_sample_rate_for_rtf = self.SAMPLE_RATE
            real_time_factor_first_chunk = time_to_first_chunk / (wav.shape[0] / original_sample_rate_for_rtf)

            print(f"==== 首次 Chunk 信息 ====")
            print(f"Time to get first chunk: {time_to_first_chunk:.4f}s")
            print(f"First chunk raw wav shape: {wav.shape}, sample rate: {original_sample_rate_for_rtf} Hz")
            print(f"First chunk length (seconds based on raw shape): {wav.shape[0] / original_sample_rate_for_rtf:.4f}s")
            print(f"Real-time factor (RTF) for first chunk: {real_time_factor_first_chunk:.4f}")
            print(f"Total elapsed time at first chunk: {time.time() - start_time:.4f}s")
            print(f"========================")

            is_first_chunk = False
        yield audio_resampled.raw_data

        print(f"Kokoro TTS time: {time.time() - start_time:.4f}s")
