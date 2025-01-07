from typing import Optional, Tuple, AsyncGenerator

class TTSInterface:
    async def text_to_speech(self, text: str, vc_uid: str, target_lang: Optional[str] = None) -> Tuple[str]:
        """
        将文本转换为语音，并返回音频文件路径
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )

    async def text_to_speech_stream(self, text: str, vc_uid: str, simultaneous: bool) -> AsyncGenerator[bytes, None]:
        """
        将文本转换为语音，并返回音频流
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
