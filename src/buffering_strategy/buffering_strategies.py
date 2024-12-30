import asyncio
import json
import os
import time
import logging
from .buffering_strategy_interface import BufferingStrategyInterface
import ormsgpack
from collections import deque
import io

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

        self.interrupt_flag = False
        self.processing_flag = False
        self.processing_task = None

    def process_audio(self, channel, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            channel: The full connection for sending s2s.
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
                #self.interrupt_flag = True
                # FIXME: TO interrupt live-audio, start talking
                # asyncio.create_task(
                #     self._send_interrupt_signal(channel)
                # )
                logging.debug("Warning in realtime processing: tried processing a new chunk while the previous one was still being processed")
                return

            # 处理累积的buffer
            self.client.scratch_buffer.extend(self.client.buffer)
            self.client.buffer.clear()
            self.processing_flag = True

            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(
                self.process_audio_async(channel, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)
            )

    async def _send_interrupt_signal(self, channel, use_webrtc):
        """
        Sends an audio chunk to the specified channel using either WebRTC or WebSocket.
    
        Args:
            channel: The channel object to send the chunk to.
            use_webrtc (bool): Whether to use WebRTC for sending.
            chunk: The audio chunk to send.
        """
        async def send_webrtc():
            channel.send(b"END_OF_AUDIO")
    
        async def send_websocket():
            await channel.send_bytes(b"END_OF_AUDIO")
    
        try:
            await (send_webrtc() if use_webrtc else send_websocket())
        except Exception as e:
            logger.error(f"Failed to send audio chunk:{e}")

    async def _send(self, channel, use_webrtc, chunk):
        """
        Sends an audio chunk to the specified channel using either WebRTC or WebSocket.
    
        Args:
            channel: The channel object to send the chunk to.
            use_webrtc (bool): Whether to use WebRTC for sending.
            chunk: The audio chunk to send.
        """
        async def send_webrtc():
            channel.send(chunk)
    
        async def send_websocket():
            await channel.send_bytes(chunk)
    
        try:
            await (send_webrtc() if use_webrtc else send_websocket())
        except Exception as e:
            logger.error(f"Failed to send audio chunk:{e}")

    def _update_client_state(self, updated_history):
        """Update client state after TTS process ends."""
        self.client.history = updated_history
        self.client.scratch_buffer.clear()
        self.client.increment_file_counter()

    async def process_audio_async(self, channel, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and s2s of the audio data. It sends the
        s2s results through the channel connection.

        Args:
            channel (Websocket/WebRTC): The channel connection for sending
                                   s2s.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
            llm_pipeline: The language model pipeline.
            tts_pipeline: The text-to-speech pipeline.
        """
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.client)
        logging.debug(f"vad vad_results: {vad_results}")
        if len(vad_results) == 0:
            self.client.scratch_buffer.clear()
            self.client.buffer.clear()
            self.processing_flag = False
            self.interrupt_flag = False
            return

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        if vad_results[-1]["end"] < last_segment_should_end_before:
            # Step 1: Transcribe audio
            transcription = await asr_pipeline.transcribe(self.client)
            if transcription["text"] != "":
                # Step 2: Generate response
                tts_text, updated_history = await llm_pipeline.generate_response(
                    self.client.history, transcription["text"], True
                )
                # Step 3: Stream audio chunks
                try:
                    async for chunk in tts_pipeline.text_to_speech_stream(tts_text, self.client.vc_uid):
                        if not self.interrupt_flag:
                            await self._send(channel, use_webrtc, chunk)
                        else:
                            raise StopAsyncIteration

                except StopAsyncIteration:
                    logging.warning("TTS stream interrupted.")
                    # Send stop signal
                    # await self._send_interrupt_signal(channel)
                    
                except Exception as e:
                    logging.error(f"An error occurred during TTS: {e}")
                finally:
                    # Always clean up, no matter success or failure
                    end = time.time()
                    print(f"Total processing time: {end - start:.2f}s, text: {tts_text}")
                    self._update_client_state(updated_history)

        self.processing_flag = False
        self.interrupt_flag = False
