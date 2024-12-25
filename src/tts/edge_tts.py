import io
import time
from uuid import uuid4
from typing import Tuple, AsyncGenerator
import edge_tts
from pydub import AudioSegment
from .tts_interface import TTSInterface


import langid

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
         
    async def get_voices(self, **kwargs):
        from edge_tts import VoicesManager

        voice_mg: VoicesManager = await VoicesManager.create()
        return voice_mg.find(**kwargs)

    async def save_submakers(self, vit_file: str):
        with open(vit_file, "w", encoding="utf-8") as file:
            file.write(self.submaker.generate_subs())

    def set_voice(self, voice: str):
        self.args.voice_name = voice

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 16000,
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str) -> Tuple[str]:
        start_time = time.time()
        audio_buffer = io.BytesIO()
        language, _ = langid.classify(text)
        if language == "zh":
            language = "zh-CN"
        """使用 edge_tts 库将文本转语音"""
        rate: int = 15
        pitch: int = 20
        volume: int = 110

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"
        output_path = f"/asset/audio_{uuid4().hex[:8]}.mp3"
        # 初始化 Communicate 对象，设置语音、语速、音调和音量参数
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
        )

        await communicate.save(output_path)
        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        # 返回原始文件名
        return output_path

    async def text_to_speech_stream(self, text: str, vc_uid: str) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        audio_buffer = io.BytesIO()
        language, _ = langid.classify(text)
        #TODO: choice zh voice
        if language == "zh":
            language = "zh-CN"

        rate: int = 15
        pitch: int = 20
        volume: int = 110

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        # 初始化 Communicate 对象，设置语音、语速、音调和音量参数
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
        )

        self.submaker = edge_tts.SubMaker()
        
        with io.BytesIO() as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    self.submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

            f.seek(0)
            audio: AudioSegment = AudioSegment.from_mp3(f)
            audio_resampled = (
                audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            )  # 16bit sample_width 16/8=2  16k-mono-mp3
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)