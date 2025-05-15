from typing import List, Dict, Any, Optional
import numpy as np

class TurnInterface:
    async def predict_endpoint(
        self,
        context: Optional[List[Dict[str, str]]] = None,
        audio: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """检测当前轮次是否完成

        Args:
            context: 对话消息历史
                List of messages with "role" and "content" keys
            audio: 音频数据
                bytes: Raw PCM 16kHz audio data

        Returns:
            Dict[str, Any]: {
                "prediction": int,  # 1: complete, 0: incomplete
                "probability": float,  # confidence score
                "status": str  # success or error
            }
        """
        if context is None and audio is None:
            raise ValueError("Either context or audio must be provided")
        raise NotImplementedError
