import asyncio
import json
import os
import time
import logging
from .buffering_strategy_interface import BufferingStrategyInterface
from collections import deque
from ..utils.misc import smart_split
from ..utils.audio_utils import pcm16k_to_ulaw
import base64

import langid

logger = logging.getLogger(__name__)

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
            client (Client): The client instance.
            **kwargs: Additional keyword arguments:
                - chunk_length_seconds (float): Length of each audio chunk
                - chunk_offset_seconds (float): Offset time for processing
                - allow_interruption (bool): 是否允许用户打断 AI 的发言
                - interrupt_on_speech_start (bool): 是否在检测到语音开始时就中断
                - interrupt_min_duration (float): 最小中断持续时间(秒)，防止误触发
        """
        self.client = client

        # 中断控制参数
        self.allow_interruption = kwargs.get("allow_interruption", True)
        self.interrupt_on_speech_start = kwargs.get("interrupt_on_speech_start", True)
        self.interrupt_min_duration = kwargs.get("interrupt_min_duration", 0.3)
        self.last_speech_start = 0

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

        # 语音处理状态
        self.processing_task = None


    async def send_initial_conversation(self, endpoint, use_webrtc, text, llm, tts):
        await self._stream_tts(
            endpoint,
            use_webrtc,
            tts,
            text
        )
        # 更新对话历史
        self.client.history.append({
            "role": "assistant",
            "content": text
        })

    def stop_processing_task(self):
        """停止 LLM 生成任务"""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            self.processing_task = None

    def _should_process_new_chunk(self):
        """判断是否需要处理新的音频块"""
        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )
        return len(self.client.buffer) > chunk_length_in_bytes

    async def process_audio(self, endpoint, use_webrtc, asr, vad, eou, llm, tts):
        """处理音频数据，管理任务状态"""
        # Trigger an interruption. Your use case might work better using input_audio_buffer speech_stopped
        # Interrupt handling/AI preemption 清除流缓冲区并发送 truncate
        # # 如果AI正在说话且检测到新语音，执行中断
        # if (not self.processing_task.done() and
        #     self._should_process_new_chunk()):
        #     # 停止当前任务
        #     self.stop_processing_task()
        #     # 清理旧数据
        #     self._clear_buffers()
        #     return

        if not self._should_process_new_chunk():
            return

        # 1. VAD 检测
        if not await self._handle_vad_detection(vad):
            return

        # 2. 语音转文字
        transcription = await self._transcribe_audio(asr)
        if not transcription:
            return

        # 3. Turn taking 检测
        if not await self._check_conversation_complete(eou, transcription["text"]):
            return

        # 开始处理新的音频块
        self.client.scratch_buffer.extend(self.client.buffer)
        self.client.buffer.clear()
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(
                self.process_audio_async(endpoint, use_webrtc, asr, vad, eou, llm, tts)
            )


    async def process_audio_async(self, endpoint, use_webrtc, asr, vad, eou, llm, tts):
        """异步处理音频并生成响应"""
        start = time.time()
        try:
            # 生成和播放响应
            await self._generate_and_play_response(
                endpoint, use_webrtc, llm, tts, transcription["text"]
            )

        except Exception as e:
            logging.error(f"Error in audio processing: {e}")
            raise
        finally:
            end = time.time()
            #print(f"Total processing time: {end - start:.2f}s")
            self._clear_buffers()

    async def _handle_vad_detection(self, vad):
        """处理 VAD 检测结果"""
        vad_results = await vad.detect_activity(self.client)
        if not vad_results:
            return False

        last_segment_should_end_before = (
            len(self.client.scratch_buffer)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds

        return vad_results[-1]["end"] < last_segment_should_end_before

    async def _transcribe_audio(self, asr):
        """转录音频"""
        transcription = await asr.transcribe(self.client)
        if not transcription["text"]:
            return None
        return transcription

    async def _check_conversation_complete(self, eou, text):
        """检查对话是否完成"""
        messages = self._prepare_messages(text)
        last_language, _ = langid.classify(text)
        result = await eou.predict_endpoint(messages, last_language, self.client.scratch_buffer)

        if not result["prediction"]:
            logger.debug(f"User hasn't finished speaking (prob: {result['probability']:.3f})")
            return False

        logger.info(f"Turn complete [len={len(text)}]: {text}")
        logger.debug(f"Turn probability: {result['probability']:.3f}")

        return True

    async def _generate_and_play_response(
        self,
        endpoint,
        use_webrtc,
        llm,
        tts,
        text: str
    ):
        """生成并播放 LLM 响应

        Args:
            endpoint: WebSocket endpoint
            use_webrtc: 是否使用 WebRTC
            llm: LLM 管道
            tts: TTS 管道
            text: 输入文本
        """
        response_buffer = []
        try:
            # 创建并开始 LLM 生成流
            stream = llm.generate_stream(
                self.client.history,
                text,
                self.client.config["is_simultaneous"],
                self.client.config["target_lang"]
            )

            buffer = ""
            seg_idx = 1  # 句子序号从 1 开始
            # 实时处理 LLM 输出
            async for delta in stream:
                response_buffer.append(delta)
                buffer += delta
                sentences = smart_split(buffer)

                # 只处理完整的句子，保留最后一段 incomplete 的
                complete = sentences[:-1]
                for sentence in complete:
                    logging.info(f"seg {seg_idx}: {sentence}\n")
                    await self._stream_tts(
                        endpoint,
                        use_webrtc,
                        tts,
                        sentence
                    )
                    seg_idx += 1

                # 保留最后一个不完整的片段
                buffer = sentences[-1] if sentences else buffer

            # 处理剩余文本
            if buffer:
                await self._stream_tts(
                    endpoint,
                    use_webrtc,
                    tts,
                    buffer
                )

            # 更新对话历史
            self.client.history.append({
                "role": "assistant",
                "content": "".join(response_buffer)
            })
            # 更新客户端状态
            self._update_client_state(self.client.history)

        except asyncio.CancelledError:
            logger.info("Response generation interrupted")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
        finally:
            return

    async def _stream_tts(
        self,
        endpoint,
        use_webrtc,
        tts,
        text: str
    ):
        """流式处理文本到语音转换

        Args:
            endpoint: WebSocket endpoint
            use_webrtc: 是否使用 WebRTC
            tts: TTS 管道
            text: 要转换的文本块
        """
        try:
            async for chunk in tts.text_to_speech_stream(
                text,
                self.client.vc_uid,
                self.client.config["is_simultaneous"]
            ):
                await self._send(endpoint, use_webrtc, chunk)
        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")
            # 继续处理，不中断整个流程
            pass

    async def _send(self, endpoint, use_webrtc: bool, chunk: bytes):
        """发送音频数据

        Args:
            endpoint: 目标端点
            use_webrtc: 是否使用 WebRTC
            chunk: 音频数据块
        """
        audio_payload = base64.b64encode(pcm16k_to_ulaw(chunk)).decode('utf-8')
        audio_delta = {
            "event": "media",
            "streamSid": self.client.stream_sid(),
            "media": {
                "payload": audio_payload
            }
        }
        try:
            if use_webrtc:
                endpoint.send(chunk)
            else:
                await endpoint.send_json(audio_delta)
        except Exception as e:
            logger.error(f"发送失败: {e}")
            raise

    def _update_client_state(self, updated_history):
        """Update client state after TTS process ends."""
        self.client.history = updated_history
        self.client.scratch_buffer.clear()
        self.client.increment_file_counter()

    def _prepare_messages(self, transcription_text: str) -> tuple[str, bool]:
        """准备要发送给 LLM 的消息

        Args:
            transcription_text: 转录文本

        Returns:
            tuple: (处理后的文本, 是否完整对话)
        """
        if not transcription_text:
            return False

        user_message = {
            "role": "user",
            "content": transcription_text
        }
        messages = getattr(self.client, 'history', []).copy()  # 复制现有历史
        messages.append(user_message)  # 添加新消息

        return messages

    def _clear_buffers(self):
        """清理所有缓冲区和状态"""
        # 清理音频缓冲
        self.client.scratch_buffer.clear()
