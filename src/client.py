# isort: skip_file

from src.buffering_strategy.buffering_strategy_factory import (
    BufferingStrategyFactory,
)


class Client:
    """
    Represents a client connected to the audio server.

    This class maintains the state for each connected client, including their
    unique identifier, audio buffer, configuration, and a counter for processed
    audio files.

    Attributes:
        client_id (str): A unique identifier for the client.
        buffer (bytearray): A buffer to store incoming audio data.
        config (dict): Configuration settings for the client, like chunk length
                       and offset.
        file_counter (int): Counter for the number of audio files processed.
        total_samples (int): Total number of audio samples received from this
                             client.
        sampling_rate (int): The sampling rate of the audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
    """

    def __init__(self, use_webrtc, client_id, sampling_rate, samples_width):
        self.use_webrtc = use_webrtc
        # client side endpoint
        self.endpoint = None
        self.client_id = client_id
        self.history = []
        self.speaker = None
        self.buffer = bytearray()        # Stores incoming audio data
        self.scratch_buffer = bytearray() # Used for processing chunks
        self.config = {
            "is_simultaneous": False,
            "target_lang": None,
            "processing_strategy": "silence_at_end_of_chunk",
            "processing_args": {
                "chunk_length_seconds": 0.5,
                "chunk_offset_seconds": 0.1,
            },
        }
        self.file_counter = 0
        self.total_samples = 0
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.buffering_strategy = (
            BufferingStrategyFactory.create_buffering_strategy(
                self.config["processing_strategy"],
                self,
                **self.config["processing_args"],
            )
        )
        self.vc_uid = "c9cf4e49"

    def update_config(self, config_data):
        self.config.update(config_data)
        self.buffering_strategy = (
            BufferingStrategyFactory.create_buffering_strategy(
                self.config["processing_strategy"],
                self,
                **self.config["processing_args"],
            )
        )

    def set_stream_sid(self, sid):
        self.sid = sid

    def stream_sid(self):
        return self.sid

    def append_audio_data(self, audio_data, vc_uid):
        self.buffer.extend(audio_data)
        self.total_samples += len(audio_data) / self.samples_width

        self.vc_uid = vc_uid

    def clear_buffer(self):
        self.buffer.clear()

    def increment_file_counter(self):
        self.file_counter += 1

    def get_file_name(self):
        return f"{self.client_id}_{self.file_counter}.wav"

    async def send_initial_conversation(self, endpoint, text, llm, tts):
        """
        Sends the initial conversation data to the specified endpoint.

        This method processes the provided text using the given LLM (Language Model)
        and TTS (Text-to-Speech) pipelines, and sends the data to the endpoint using
        the buffering strategy.

        Args:
            endpoint (str): The target endpoint to send the conversation data to.
            text (str): The initial text of the conversation to be processed.
            llm (Callable): The Language Model pipeline to process the text.
            tts (Callable): The Text-to-Speech pipeline to generate audio output.

        Returns:
            None

        """
        await self.buffering_strategy.send_initial_conversation(
            endpoint, self.use_webrtc, text, llm, tts
        )

    async def process_audio(self, endpoint, asr, vad, eou, llm, tts):
        """
        Process the audio data in the buffer using the provided ASR, VAD, EOU,
        LLM, and TTS pipelines.
        This method is responsible for handling the audio data
        """
        await self.buffering_strategy.process_audio(
            endpoint, self.use_webrtc, asr, vad, eou, llm, tts
        )
