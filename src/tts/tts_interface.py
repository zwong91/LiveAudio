from typing import Tuple

class TTSInterface:
    async def text_to_speech(self, text: str, vc_uid: str, gen_file: bool) -> Tuple[bytes, str]:
        """
        将文本转换为语音，并返回音频文件路径及其文本
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
