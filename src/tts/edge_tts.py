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
         
    async def get_voices(self, **kwargs):
        from edge_tts import VoicesManager

        voice_mg: VoicesManager = await VoicesManager.create()
        return voice_mg.find(**kwargs)

    async def save_submakers(self, vit_file: str):
        with open(vit_file, "w", encoding="utf-8") as file:
            file.write(self.submaker.generate_subs())

    def get_stream_info(self) -> dict:
        return {
            "sample_rate": 16000,
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
        )

        await communicate.save(output_path)
        end_time = time.time()
        print(f"EdgeTTS text_to_speech time: {end_time - start_time:.4f} seconds")
        # 返回原始文件名
        return output_path

    async def text_to_speech_stream(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        start_time = time.time()
        audio_buffer = io.BytesIO()
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
        )

        self.submaker = edge_tts.SubMaker()
        
        audio = AudioSegment.from_wav(self.talking_wav)
        # 重采样为 16kHz，单声道，16-bit
        audio_resampled = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        pcm_data_16K = audio_resampled.raw_data
        yield wave_header_chunk(pcm_data_16K, 1, 2, 16000)

        with io.BytesIO() as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    self.submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

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