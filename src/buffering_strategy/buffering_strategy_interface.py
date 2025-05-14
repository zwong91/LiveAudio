class BufferingStrategyInterface:
    """
    An interface class for buffering strategies in audio processing systems.

    This class defines the structure for buffering strategies used in handling
    and processing audio data. It serves as a template for creating custom
    buffering strategies that fit specific requirements of an audio processing
    pipeline.

    Subclasses should implement the methods defined in this interface to ensure
    consistency and compatibility with the system's audio processing framework.

    Methods:
        process_audio: Process audio data. This method should be implemented
                       by subclasses.
    """

    async def process_audio(self, websocket, asr, vad, eou, llm, tts):
        """
        Process audio data using the provided WebSocket connection and various processing pipelines.

        This method is designed to be overridden in subclasses to implement specific
        buffering strategies for handling and processing audio data.

            websocket (WebSocket): The WebSocket connection for communication with clients.
            asr: The Automatic Speech Recognition (ASR) pipeline for transcribing speech.
            vad: The Voice Activity Detection (VAD) pipeline for detecting speech activity.
            eou: End-of-Utterance (EOU) detection mechanism for identifying the end of speech.
            llm: The Large Language Model (LLM) for processing or generating text based on transcriptions.
            tts: The Text-to-Speech (TTS) system for converting text to audio responses.

            NotImplementedError: If the method is not implemented in a subclass.

        Raises:
            NotImplementedError: If the method is not implemented in the
                                 subclass.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
