from typing import List, Dict, Any, Optional
import numpy as np

class TurnInterface:
    async def predict_endpoint(
        self,
        context: Optional[List[Dict[str, str]]] = None,
        last_language: str = "zh",
        trans_len: str = 0,
        audio: Optional[bytearray] = None
    ) -> Dict[str, Any]:
        """检测当前轮次是否完成

        Args:
            context: 对话消息历史
                List of messages with "role" and "content" keys
            last_language: 上一轮的语言
                str: 语言代码, e.g., "zh", "en"
            audio: 音频数据
                bytearray: Raw PCM 16kHz audio data

        Returns:
            Dict[str, Any]: {
                "prediction": int,  # 1: complete, 0: incomplete
                "probability": float,  # confidence score
            }
        """
        if context is None and audio is None:
            raise ValueError("Either context or audio must be provided")
        raise NotImplementedError
