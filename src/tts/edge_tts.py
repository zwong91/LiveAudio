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
    'es-MX-DaliaNeural', 'es-MX-JorgeNeural', 'ko-KR-SunHiNeural', 'ko-KR-InJoonNeural',
    'ja-JP-NanamiNeural', 'ja-JP-KeitaNeural', 'fr-FR-DeniseNeural', 'fr-FR-EloiseNeural',
    'fr-FR-HenriNeural', 'pt-BR-FranciscaNeural', 'pt-BR-AntonioNeural', 'id-ID-ArdiNeural',
    'id-ID-GadisNeural', 'he-IL-AvriNeural', 'he-IL-HilaNeural', 'it-IT-IsabellaNeural',
    'it-IT-DiegoNeural', 'it-IT-ElsaNeural', 'nl-NL-ColetteNeural', 'nl-NL-FennaNeural',
    'nl-NL-MaartenNeural', 'nb-NO-FinnNeural', 'sv-SE-SofieNeural', 'sv-SE-MattiasNeural',
    'ar-SA-HamedNeural', 'ar-SA-ZariyahNeural', 'el-GR-AthinaNeural', 'el-GR-NestorasNeural',
    'de-DE-KatjaNeural', 'de-DE-AmalaNeural', 'de-DE-ConradNeural', 'de-DE-KillianNeural',
    'ar-AE-FatimaNeural', 'ar-AE-HamdanNeural', 'ar-EG-SalmaNeural', 'ar-EG-ShakirNeural',
    'ar-IQ-BasselNeural', 'ar-IQ-RanaNeural', 'da-DK-ChristelNeural', 'da-DK-JeppeNeural',
    'de-AT-IngridNeural', 'de-AT-JonasNeural', 'de-CH-JanNeural', 'de-CH-LeniNeural',
    'en-AU-NatashaNeural', 'en-AU-WilliamNeural', 'en-CA-ClaraNeural', 'en-CA-LiamNeural',
    'en-GB-LibbyNeural', 'en-GB-MaisieNeural', 'en-GB-RyanNeural', 'en-GB-SoniaNeural',
    'en-GB-ThomasNeural', 'en-HK-SamNeural', 'en-HK-YanNeural', 'en-IN-NeerjaNeural',
    'en-IN-PrabhatNeural', 'en-SG-LunaNeural', 'en-SG-WayneNeural', 'es-AR-ElenaNeural',
    'es-AR-TomasNeural', 'es-ES-AlvaroNeural', 'es-ES-ElviraNeural', 'es-US-AlonsoNeural',
    'es-US-PalomaNeural', 'fr-CH-ArianeNeural', 'fr-CH-FabriceNeural', 'ga-IE-ColmNeural',
    'ga-IE-OrlaNeural', 'gl-ES-RoiNeural', 'gl-ES-SabelaNeural', 'hi-IN-MadhurNeural',
    'hi-IN-SwaraNeural', 'hr-HR-GabrijelaNeural', 'hr-HR-SreckoNeural', 'hu-HU-NoemiNeural',
    'hu-HU-TamasNeural', 'ru-RU-SvetlanaNeural', 'ru-RU-DmitryNeural', 'tr-TR-AhmetNeural',
    'tr-TR-EmelNeural', 'zh-CN-XiaoxiaoNeural', 'zh-CN-YunyangNeural', 'zh-CN-YunxiNeural',
    'zh-CN-XiaoyiNeural', 'zh-CN-YunjianNeural', 'zh-CN-YunxiaNeural', 'zh-CN-liaoning-XiaobeiNeural',
    'zh-CN-shaanxi-XiaoniNeural', 'zh-HK-HiuMaanNeural', 'zh-HK-HiuGaaiNeural', 'zh-HK-WanLungNeural',
    'zh-TW-HsiaoChenNeural', 'zh-TW-HsiaoYuNeural', 'zh-TW-YunJheNeural'
]

class EdgeTTS(TTSInterface):
    def __init__(self, voice: str = 'zh-CN-XiaoxiaoNeural'):
        self.voice = voice
        self.talking_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "talking.wav")
        self.silence_wav = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "vc")), "silence.wav")

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

    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
        """使用 edge_tts 库将文本转语音"""
        start_time = time.time()
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
        #1. send talking audio
        if not simultaneous:
            audio = AudioSegment.from_wav(self.talking_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        #2. stream synthesize audio
        with io.BytesIO() as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
            # 将 BytesIO 中的数据重置指针，并加载为 AudioSegment
            f.seek(0)
            audio: AudioSegment = AudioSegment.from_mp3(f)
            # 处理音频，重采样到16kHz，单声道，16bit
            audio_resampled = (
                audio.set_frame_rate(16000)
                    .set_channels(1)
                    .set_sample_width(2)  # 16bit sample_width 16/8=2  16k-mono-mp3
            )
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        # FIXME: ms-edge 浏览器也是有时候就是没有语音数据返回, ask microsoft. 还有就是mp3 chunk边界间隙依赖上一个chunk, 最终的方式应该是直接yield chunk["data"]， 前端用mpv实时流播放器
        # CHUNK_SIZE = 10 * 1024  # 假设每个块大约1024字节（根据实际格式调整）
        # total_data = b""  # 用于存储接收到的音频数据
        # for chunk in communicate.stream_sync():
        #     if chunk["type"] == "audio":
        #         total_data += chunk["data"]
                
        #         # 如果接收到的数据达到一个完整的块大小
        #         if len(total_data) >= CHUNK_SIZE:
        #             print(f"First chunk Time elapsed: {time.time() - start_time:.2f} seconds")
                    
        #             # 使用 BytesIO 来读取音频数据
        #             with io.BytesIO(total_data[:CHUNK_SIZE]) as audio_io:
        #                 audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
        #                 # 处理音频，重采样到16kHz，单声道，16bit
        #                 audio_resampled = (
        #                     audio.set_frame_rate(16000)
        #                         .set_channels(1)
        #                         .set_sample_width(2)  # 16bit sample_width (16/8=2)
        #                 )
        #                 pcm_data_16K = audio_resampled.raw_data
        #                 # 使用 wave_header_chunk 发送处理后的数据
        #                 yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
                    
        #             # 移除已经处理的音频数据, 并且向前overlapped
        #             total_data = total_data[CHUNK_SIZE:]

        # # 处理剩余的数据
        # if total_data:
        #     print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
        #     # 使用 BytesIO 来读取剩余的音频数据
        #     with io.BytesIO(total_data) as audio_io:
        #         audio: AudioSegment = AudioSegment.from_file(audio_io, format="mp3")
        #         # 处理音频，重采样到16kHz，单声道，16bit
        #         audio_resampled = (
        #             audio.set_frame_rate(16000)
        #                 .set_channels(1)
        #                 .set_sample_width(2)  # 16bit sample_width (16/8=2)
        #         )
        #         pcm_data_16K = audio_resampled.raw_data
                
        #         # 使用 wave_header_chunk 发送处理后的数据
        #         yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
     
        #3. send silent audio           
        if not simultaneous:
            audio = AudioSegment.from_wav(self.silence_wav)
            # 重采样为 16kHz，单声道，16-bit
            audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            pcm_data_16K = audio_resampled.raw_data
            yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)
                