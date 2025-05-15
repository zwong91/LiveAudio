
class FilterInterface:
      async def filter(self, audio: bytes) -> bytes:
        """
        This method is intended to process audio data and return the filtered result.
        audio (bytes): The input audio data in PCM 16k format.
        bytes: The filtered audio data.
        """
        pass
