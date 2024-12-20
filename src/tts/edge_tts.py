import time
from uuid import uuid4
from typing import Tuple
import edge_tts
from io import BytesIO
from .tts_interface import TTSInterface


import langid

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
            
    async def text_to_speech(self, text: str, vc_uid: str) -> Tuple[str]:
        start_time = time.time()
        audio_buffer = BytesIO()
        language, _ = langid.classify(text)
        if language == "zh":
            language = "zh-CN"
        """使用 edge_tts 库将文本转语音"""
        rate: int = 20
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

    async def text_to_speech_stream(self, text: str, vc_uid: str) -> Tuple[bytes]:
        start_time = time.time()
        audio_buffer = BytesIO()
        language, _ = langid.classify(text)
        #TODO: choice zh voice
        if language == "zh":
            language = "zh-CN"
        """使用 edge_tts 库将文本转语音"""
        rate: int = 20
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

        # async for chunk in communicate.stream():
        #     if chunk["type"] == "audio":
        #         print(f"EdgeTTS 音频chunk大小: {len(chunk['data'])} 字节")
        #         yield chunk["data"]
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        print(f"音频数据大小: {len(audio_data)} 字节")
        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        # 返回音频数据的字节流
        return audio_data