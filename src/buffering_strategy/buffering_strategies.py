import asyncio
import json
import os
import time
import logging
from .buffering_strategy_interface import BufferingStrategyInterface
import ormsgpack
from collections import deque
import io
from ..eou.eou_detector import EOUDetector

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
        # Initialize EOUDetector
        self.eou_detector = EOUDetector()

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

        self.interrupt_flag = False
        self.processing_flag = False
        self.processing_task = None

    def process_audio(self, endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            endpoint: The full connection for sending s2s.
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
                logging.debug("Warning in realtime processing: tried processing a new chunk while the previous one was still being processed")
                return

            # 处理累积的buffer
            self.client.scratch_buffer.extend(self.client.buffer)
            self.client.buffer.clear()
            self.processing_flag = True

            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(
                self.process_audio_async(endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)
            )

    async def _send_interrupt_signal(self, endpoint, use_webrtc):
        """
        Sends an audio chunk to the specified endpoint using either WebRTC or WebSocket.

        Args:
            endpoint: The endpoint object to send the chunk to.
            use_webrtc (bool): Whether to use WebRTC for sending.
            chunk: The audio chunk to send.
        """
        async def send_webrtc():
            endpoint.send(b"END_OF_AUDIO")

        async def send_websocket():
            await endpoint.send_bytes(b"END_OF_AUDIO")

        try:
            await (send_webrtc() if use_webrtc else send_websocket())
        except Exception as e:
            logging.error(f"Failed to send audio chunk:{e}")

    async def _send(self, endpoint, use_webrtc, chunk):
        """
        Sends an audio chunk to the specified endpoint using either WebRTC or WebSocket.

        Args:
            endpoint: The endpoint object to send the chunk to.
            use_webrtc (bool): Whether to use WebRTC for sending.
            chunk: The audio chunk to send.
        """
        async def send_webrtc():
            endpoint.send(chunk)

        async def send_websocket():
            await endpoint.send_bytes(chunk)

        try:
            await (send_webrtc() if use_webrtc else send_websocket())
        except Exception as e:
            logging.error(f"Failed to send audio chunk:{e}")

    def _update_client_state(self, updated_history):
        """Update client state after TTS process ends."""
        self.client.history = updated_history
        self.client.scratch_buffer.clear()
        self.client.increment_file_counter()

    def _prepare_messages(self, transcription_text):
        """准备对话消息历史

        Args:
            transcription_text (str): 转录的文本

        Returns:
            list: 包含历史消息的列表
        """
        if not hasattr(self.client, 'history'):
            self.client.history = []

        # 构造新消息
        user_message = {
            "role": "user",
            "content": transcription_text
        }

        # 复制历史并添加新消息
        messages = self.client.history.copy()
        messages.append(user_message)
        return messages

    async def process_audio_async(self, endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.
        """
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)
        logging.debug(f"vad vad_results: {vad_results}")

        if len(vad_results) == 0:
            self._clear_buffers()
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds

        # Only proceed with processing if VAD detects speech end
        if vad_results[-1]["end"] < last_segment_should_end_before:
            # Step 1: Transcribe audio
            transcription = await asr_pipeline.transcribe(self.client)
            if not transcription["text"]:
                self._clear_buffers()
                return

            # Step 2: Prepare messages and check if turn is complete
            messages = self._prepare_messages(transcription["text"])
            if not self.eou_detector.is_turn_complete(messages):
                logging.debug("EOUDetector indicates user hasn't finished speaking")
                self._partial_clear()  # 保留 scratch_buffer，因为用户还没说完
                return

            # Step 3: Generate and stream response
            try:
                tts_text, updated_history = await llm_pipeline.generate_response(
                    self.client.history,
                    transcription["text"],
                    self.client.config["is_simultaneous"],
                    self.client.config["target_lang"],
                    True
                )

                await self._stream_audio_response(endpoint, use_webrtc, tts_pipeline, tts_text)

            except Exception as e:
                logging.error(f"An error occurred during processing: {e}")
            finally:
                end = time.time()
                print(f"Total processing time: {end - start:.2f}s, text: {tts_text}")
                self._update_client_state(updated_history)

        self._clear_buffers()

    def _clear_buffers(self):
        """完全清理所有缓冲区和状态标志"""
        self.client.scratch_buffer.clear()
        self.client.buffer.clear()
        self.processing_flag = False
        self.interrupt_flag = False

    def _partial_clear(self):
        """只清理状态标志，保留 scratch_buffer"""
        self.client.buffer.clear()
        self.processing_flag = False
        self.interrupt_flag = False

    async def _stream_audio_response(self, endpoint, use_webrtc, tts_pipeline, text):
        """流式传输音频响应

        Args:
            endpoint: WebSocket/WebRTC endpoint
            use_webrtc (bool): 是否使用WebRTC
            tts_pipeline: TTS pipeline实例
            text (str): 要转换成语音的文本
        """
        try:
            async for chunk in tts_pipeline.text_to_speech_stream(
                text,
                self.client.vc_uid,
                self.client.config["is_simultaneous"]
            ):
                if not self.interrupt_flag:
                    await self._send(endpoint, use_webrtc, chunk)
                else:
                    raise StopAsyncIteration

        except StopAsyncIteration:
            logging.warning("TTS stream interrupted.")
