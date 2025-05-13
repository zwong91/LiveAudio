import asyncio
import logging
import os
import time
from collections import deque

from .buffering_strategy_interface import BufferingStrategyInterface
from ..eou.eou_detector import EOUDetector

logger = logging.getLogger(__name__)

# Constants for punctuation
CN_SENTENCE_END = set("。！？…")
EN_SENTENCE_END = set(".!?")
CLAUSE_BREAKS = set("，,；;")

class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    Buffering strategy: process audio chunks upon detecting silence at end of chunk.
    """

    def __init__(self, client, *, chunk_length_seconds=None, chunk_offset_seconds=None, error_if_not_realtime=False):
        self.client = client
        self.eou_detector = EOUDetector()

        self.chunk_length_seconds = float(
            os.getenv("BUFFERING_CHUNK_LENGTH_SECONDS", chunk_length_seconds)
        )
        self.chunk_offset_seconds = float(
            os.getenv("BUFFERING_CHUNK_OFFSET_SECONDS", chunk_offset_seconds)
        )
        self.error_if_not_realtime = bool(
            os.getenv("ERROR_IF_NOT_REALTIME", str(error_if_not_realtime)) == "True"
        )

        self.interrupt_flag = False
        self.processing_task = None
        self.llm_task = None

        self.interrupt_queue = asyncio.Queue()

    def process_audio(self, endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        if not self._ready_for_new_chunk():
            return
        if self.processing_task and not self.processing_task.done():
            asyncio.create_task(self._request_interrupt())
        else:
            self._start_processing(endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)

    def _ready_for_new_chunk(self) -> bool:
        bytes_ready = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        return len(self.client.buffer) > bytes_ready

    async def _request_interrupt(self):
        if self.interrupt_flag:
            return
        self.interrupt_flag = True
        await self.interrupt_queue.put(True)
        try:
            await asyncio.wait_for(self.processing_task, timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning("Interrupt timeout, cancelling task")
            self.processing_task.cancel()
        finally:
            self._reset_state()

    def _start_processing(self, endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        # swap buffers
        self.client.scratch_buffer.extend(self.client.buffer)
        self.client.buffer.clear()
        self.processing_task = asyncio.create_task(
            self._process_chunk(endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)
        )

    async def _process_chunk(self, endpoint, use_webrtc, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        start = time.time()
        try:
            if not await self._detect_silence(vad_pipeline):
                return

            text = await self._transcribe(asr_pipeline)
            if not text:
                return

            if not await self._await_end_of_utterance(text):
                return

            await self._respond(endpoint, use_webrtc, llm_pipeline, tts_pipeline, text)

        except Exception as e:
            logger.error("Processing error: %s", e)
        finally:
            logger.info(f"Chunk processed in {time.time() - start:.2f}s")
            self._reset_buffers()

    async def _detect_silence(self, vad_pipeline) -> bool:
        results = await vad_pipeline.detect_activity(self.client)
        if not results:
            return False
        last_end = results[-1]["end"]
        buf_duration = len(self.client.scratch_buffer) / (self.client.sampling_rate * self.client.samples_width)
        return last_end < buf_duration - self.chunk_offset_seconds

    async def _transcribe(self, asr_pipeline) -> str:
        result = await asr_pipeline.transcribe(self.client)
        text = result.get("text", "").strip()
        return text

    async def _await_end_of_utterance(self, text: str) -> bool:
        # clear on interrupt
        if not self.interrupt_queue.empty():
            await self.interrupt_queue.get()
            self.interrupt_flag = False
            return False

        # use detector
        return self.eou_detector.is_turn_complete([{"role": "user", "content": text}])

    async def _respond(self, endpoint, use_webrtc, llm_pipeline, tts_pipeline, text: str):
        buffer = []
        async for chunk in llm_pipeline.generate_stream(
            self.client.history,
            text,
            self.client.config.get("is_simultaneous"),
            self.client.config.get("target_lang")
        ):
            if self.interrupt_flag:
                break
            buffer.append(chunk)
            current = "".join(buffer)
            if self._is_sentence_end(current) or self._has_clause_break(current):
                await self._play_tts(endpoint, use_webrtc, tts_pipeline, current)
                buffer.clear()
        # remaining
        if buffer and not self.interrupt_flag:
            await self._play_tts(endpoint, use_webrtc, tts_pipeline, "".join(buffer))
        # update history
        if not self.interrupt_flag:
            self.client.history.append({"role": "assistant", "content": text})

    async def _play_tts(self, endpoint, use_webrtc, tts_pipeline, text: str):
        try:
            async for audio in tts_pipeline.text_to_speech_stream(
                text,
                self.client.vc_uid,
                self.client.config.get("is_simultaneous")
            ):
                if self.interrupt_flag:
                    return
                if use_webrtc:
                    endpoint.send(audio)
                else:
                    await endpoint.send_bytes(audio)
        except Exception:
            logger.exception("TTS error")

    def _reset_buffers(self):
        self.client.buffer.clear()
        self.client.scratch_buffer.clear()
        self.interrupt_flag = False
        self.processing_task = None
        self.llm_task = None

    def _reset_state(self):
        self._reset_buffers()
        while not self.interrupt_queue.empty():
            self.interrupt_queue.get_nowait()

    def _is_sentence_end(self, text: str) -> bool:
        if not text:
            return False
        return text[-1] in CN_SENTENCE_END.union(EN_SENTENCE_END)

    def _has_clause_break(self, text: str) -> bool:
        return len(text) >= 10 and any(ch in CLAUSE_BREAKS for ch in text)
