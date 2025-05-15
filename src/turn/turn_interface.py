from typing import List, Dict, Any, Union, Optional

class TurnInterface:
    async def predict_endpoint(
        self,
        context: Union[List[Dict[str, str]], bytes],
    ) -> Dict[str, Any]:
        """检测当前轮次是否完成

        Args:
            context: 对话历史或音频数据
                - List[Dict[str, str]]: 对话消息历史
                - bytes: 音频数据pcm16K

        Returns:
            Dict[str, Any]: 检测结果
             - prediction: 1 if turn is complete, else 0
             - probability: Confidence of the prediction
        """
        pass
