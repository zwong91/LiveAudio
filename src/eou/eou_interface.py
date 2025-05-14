from typing import List, Dict

class EOUInterface:
    async def detect(self, context: List[Dict[str, str]], threshold: float = 0.15) -> bool:
        """
        Detect if the current turn is complete.

        :param chat_context: List of chat messages with conversation history
        :param threshold: Probability threshold for end of turn detection
        :return: bool indicating if turn is complete
        """
        pass
