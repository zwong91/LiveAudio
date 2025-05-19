import io
import time
from uuid import uuid4
from typing import Optional, Tuple, AsyncGenerator
import edge_tts
import os
from pydub import AudioSegment
from .tts_interface import TTSInterface

from src.utils.audio_utils import wave_header_chunk

import langid

language_list = [
    'en-US-JennyNeural', 'en-US-GuyNeural', 'en-US-AnaNeural', 'en-US-AriaNeural',
    'en-US-ChristopherNeural', 'en-US-EricNeural', 'en-US-MichelleNeural', 'en-US-RogerNeural',
    'zh-CN-XiaoxiaoNeural', 'zh-CN-YunyangNeural', 'zh-CN-YunxiNeural',
    'zh-CN-XiaoyiNeural', 'zh-CN-YunjianNeural', 'zh-CN-YunxiaNeural', 'zh-CN-liaoning-XiaobeiNeural',
    'zh-CN-shaanxi-XiaoniNeural', 'zh-HK-HiuMaanNeural', 'zh-HK-HiuGaaiNeural', 'zh-HK-WanLungNeural',
    'zh-TW-HsiaoChenNeural', 'zh-TW-HsiaoYuNeural', 'zh-TW-YunJheNeural'
]

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice

    async def get_voices(self, **kwargs):
        from edge_tts import VoicesManager

        voice_mg: VoicesManager = await VoicesManager.create()
        return voice_mg.find(**kwargs)

    """
    CHANNELS = 1
    RATE = 24000  # azure (16000) system (22050),
    """
    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 16000, #msedge (24000)
            "sample_width": 2,
            "channels": 1,
        }

    async def text_to_speech(self, text: str, vc_uid: str, speed: Optional[float] = None) -> Tuple[str]:
        """使用 edge_tts 库将文本转语音"""
        start_time = time.time()
        v_speed = 1.0 if speed is None else float(speed)
        v_speed = max(0.5, min(2.0, v_speed))
        audio_buffer = io.BytesIO()
        language, _ = langid.classify(text)
        if language == "zh":
            language = "zh-CN"

        voices = [voice for voice in language_list if voice.startswith(language)]
        voice = self.voice
        if voices:
            # 如果存在，取第一个语音
            print(f"Target wav files:{voices[0]}, Detected language: {language}, tts text: {text}")
            voice = voices[0]
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
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
            #proxy="http://127.0.0.1:7890"
        )

        await communicate.save(output_path)
        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        # 返回原始文件名
        return output_path

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        language, _ = langid.classify(text)
        #TODO: choice zh voice
        if language == "zh":
            language = "zh-CN"

        voices = [voice for voice in language_list if voice.startswith(language)]
        voice = self.voice
        if voices:
            # 如果存在，取第一个语音
            print(f"Target wav files:{voices[0]}, Detected language: {language}, tts text: {text}")
            voice = voices[0]

        rate: int = 15
        pitch: int = 20
        volume: int = 110

        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        volume_str = f"{volume:+d}%"

        # 初始化 Communicate 对象，设置语音、语速、音调和音量参数
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
            #proxy="http://127.0.0.1:7890"
        )
        #stream synthesize audio
        # with io.BytesIO() as f:
        #     async for chunk in communicate.stream():
        #         if chunk["type"] == "audio":
        #             f.write(chunk["data"])
        #     # 将 BytesIO 中的数据重置指针，并加载为 AudioSegment
        #     f.seek(0)
        #     audio: AudioSegment = AudioSegment.from_mp3(f)
        #     # 处理音频，重采样到16kHz，单声道，16bit
        #     audio_resampled = (
        #         audio.set_frame_rate(16000)
        #             .set_channels(1)
        #             .set_sample_width(2)  # 16bit sample_width 16/8=2  16k-mono-mp3
        #     )
        #     pcm_data_16K = audio_resampled.raw_data
        #     yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        # FIXME: ms-edge 浏览器也是有时候就是没有语音数据返回, ask microsoft.
        # 还有就是mp3 contain artifacts introduced chunk边界间隙伪影依赖上一个chunk, 最终的方式应该是直接yield chunk["data"]， 前端用mpv实时流播放器
        CHUNK_SIZE = 20 * 1024  # 假设每个块大约1024字节（根据实际格式调整）
        total_data = b""  # 用于存储接收到的音频数据
        is_first_chunk = True
        CHUNK_THRESHOLD = 120
        chunk_len = 5 if len(text) > CHUNK_THRESHOLD else 1
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                total_data += chunk["data"]
                # 如果接收到的数据达到一个完整的块大小
                if len(total_data) >= CHUNK_SIZE and chunk_len > 0:
                    # 使用 BytesIO 来读取音频数据
                    with io.BytesIO(total_data[:CHUNK_SIZE]) as audio_io:
                        audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
                        # 处理音频，重采样到16kHz，单声道，16bit
                        audio_resampled = (
                            audio.set_frame_rate(16000)
                                .set_channels(1)
                                .set_sample_width(2)  # 16bit sample_width (16/8=2)
                        )
                        pcm_data_16K = audio_resampled.raw_data
                        if is_first_chunk:
                            print(f"First chunk Time elapsed: {time.time() - start_time:.2f} seconds")
                            is_first_chunk = False
                            # 使用 wave_header_chunk 发送处理后的数据
                            #yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
                        # raw pcm data
                        yield pcm_data_16K

                    chunk_len -= 1
                    # 移除已经处理的音频数据, 并且向前overlapped
                    total_data = total_data[CHUNK_SIZE:]

        # 使用 BytesIO 来读取剩余的音频数据
        with io.BytesIO(total_data[:]) as audio_io:
            audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
            # 处理音频，重采样到16kHz，单声道，16bit
            audio_resampled = (
                audio.set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)  # 16bit sample_width (16/8=2)
            )
            pcm_data_16K = audio_resampled.raw_data
            yield pcm_data_16K
