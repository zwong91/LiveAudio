import asyncio
import json
import os
import time
import logging
from .buffering_strategy_interface import BufferingStrategyInterface

class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with
    silence detection.

    This class is responsible for handling audio chunks, detecting silence at
    the end of each chunk, and initiating the transcription process for the
    chunk.

    Attributes:
        client (Client): The client instance associated with this buffering
                         strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered
                                      for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering
                             strategy.
            **kwargs: Additional keyword arguments, including
                      'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client

        self.chunk_length_seconds = os.environ.get(
            "BUFFERING_CHUNK_LENGTH_SECONDS"
        )
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = kwargs.get("chunk_length_seconds")
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get(
            "BUFFERING_CHUNK_OFFSET_SECONDS"
        )
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds")
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)

        self.error_if_not_realtime = os.environ.get("ERROR_IF_NOT_REALTIME")
        if not self.error_if_not_realtime:
            self.error_if_not_realtime = kwargs.get(
                "error_if_not_realtime", False
            )

        self.processing_flag = False

    def process_audio(self, websocket, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            websocket: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        if len(self.client.buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                #FIXME: 这里直接丢弃？
                self.client.buffer.clear()
                logging.warning("Warning in realtime processing: tried processing a new chunk while the previous one was still being processed")
                return
            self.client.scratch_buffer += self.client.buffer
            self.client.buffer.clear()
            self.processing_flag = True
            # schedule the processing in a separate task
            asyncio.create_task(
                self.process_audio_async(websocket, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)
            )


    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending
                                   transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
            llm_pipeline: The language model pipeline.
            tts_pipeline: The text-to-speech pipeline.
        """
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)

        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        if vad_results[-1]["end"] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.client)
            #TODO: repeated deealing with the same data
            if transcription["text"] != "":
                tts_text, updated_history = await llm_pipeline.generate(
                    self.client.history, transcription["text"]
                )
                # async for chunk in tts_pipeline.text_to_speech(tts_text, "liuyifei", False):
                #     audio_data = chunk[0]  # 获取音频数据，chunk[0] 是音频数据
                #     await websocket.send_bytes(audio_data)
                speech_audio, _ = await tts_pipeline.text_to_speech(tts_text, "liuyifei", False)
                end = time.time()
                logging.debug(f"processing_time: {end - start}, text: {tts_text}")
                try:
                    await websocket.send_bytes(speech_audio)
                    #TODO: 异步等待 1 秒，防止音频重叠
                    #await asyncio.sleep(1)
                except Exception as e:
                    logging.error(f"Error sending WebSocket message: {e}")
                self.client.history = updated_history
                self.client.scratch_buffer.clear()
                self.client.increment_file_counter()

        self.processing_flag = False
