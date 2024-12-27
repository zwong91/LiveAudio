import asyncio
import logging
import threading
import time
from typing import Optional, Set
from av.frame import Frame
from av import AudioFrame, VideoFrame
import fractions

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from .utils.audio_utils import torchTensor2bytes

logging.basicConfig()
logger = logging.getLogger(__name__)

class ClientStreamTrack(MediaStreamTrack):
    """
    A media track that receives frames from a RTCClient.
    """

    def __init__(
            self, 
            kind, 
            track,
            vad_pipeline,
            asr_pipeline,
            llm_pipeline,
            tts_pipeline,
            peer_connection,
            datachannel, 
            signaling=None
    ):
        super().__init__()  # don't forget this!
        self.kind = kind
        self.track = track
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.llm_pipeline = llm_pipeline
        self.tts_pipeline = tts_pipeline
        self.peer_connection = peer_connection
        self.datachannel = datachannel
        # self.signaling = signaling
        
        self.sampling_rate = 16_000
        self.samples_width = 2
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sampling_rate,
        )
        self.buffer = torch.tensor(
            [],
            dtype=torch.float32,
        )
        self.history = []
        self.speaker = None
        self.scratch_buffer = bytearray()
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

    async def recv(self) -> Frame:
        frame = await self.track.recv()
        print(frame)
        frame = self.resampler.resample(frame)[0]
        frame_array = frame.to_ndarray()
        frame_array = frame_array[0].astype(np.float32)
        # print(frame_array)
        # s16 (signed integer 16-bit number) can store numbers in range -32 768...32 767.
        frame_array = torch.tensor(frame_array, dtype=torch.float32) / 32_767

        self.buffer = torch.cat(
            [
                self.buffer,
                frame_array,
            ]
        )
        print("Let's Speech to Speech!")
        self.process_audio(torchTensor2bytes(self.buffer))

        return frame
    
    def process_audio(self, buffer):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            self.peer_connection.addTrack: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.sampling_rate
            * self.samples_width
        )
        if len(buffer) > chunk_length_in_bytes:
            if self.processing_flag:
                #self.interrupt_flag = True
                # FIXME: TO interrupt live-audio, start talking
                logging.debug("Warning in realtime processing: tried processing a new chunk while the previous one was still being processed")
                return

            self.scratch_buffer.extend(buffer)
            self.buffer = torch.tensor(
                [],
                dtype=torch.float32,
            )
            self.processing_flag = True

            if self.processing_task is None or self.processing_task.done():
                self.processing_task = asyncio.create_task(
                self.process_audio_async(vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline)
            )

        # llm_text = llm_chat_v1(user_text=speech_rec_text)
        # result = f"xxx: {speech_rec_text}\n\n yyy:\n{llm_text}"
        # sf.write(
        #     "bot/experiments/aiortc_vad_stt_llm_tts/temp_speech.wav",
        #     data=self.non_silent_segments,
        #     samplerate=self.sampling_rate,
        # )
        # sample_rate, all_speech = speech_generation.generate(
        #     text=llm_text,
        # )
        # temp_audio_name = str(uuid.uuid4())
        # temp_audio_path = f"bot/experiments/aiortc_vad_stt_llm_tts/temp_audio_name.wav"
        # sf.write(
        #     temp_audio_path,
        #     data=all_speech,
        #     samplerate=sample_rate,
        # )
        # player = MediaPlayer(temp_audio_path)

        # self.peer_connection.addTrack(player.audio)
        # # player._start()
        # self.datachannel.send(result)
        
    def _update_client_state(self, updated_history):
        """Update client state after TTS process ends."""
        self.history = updated_history
        self.scratch_buffer.clear()
        self.increment_file_counter()

    async def process_audio_async(self, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            Peer Connection: The WebRTC connection for sending
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
                            await self.peer_connection.addTrack(chunk)
                        else:
                            raise StopAsyncIteration

                except StopAsyncIteration:
                    logging.warning("TTS stream interrupted.")
                    
                except Exception as e:
                    logging.error(f"An error occurred during TTS: {e}")
                finally:
                    # Always clean up, no matter success or failure
                    end = time.time()
                    print(f"Total processing time: {end - start:.2f}s, text: {tts_text}")
                    self._update_client_state(updated_history)

        self.processing_flag = False
        self.interrupt_flag = False

            
class RTCStreamTrack:
    def __init__(self, track, vad_pipeline, asr_pipeline, llm_pipeline, tts_pipeline, peer_connection, datachannel, signaling=None):

        # Examine streams
        self.__audio: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="audio", track=track, vad_pipeline=vad_pipeline, asr_pipeline=asr_pipeline, llm_pipeline=llm_pipeline, tts_pipeline=tts_pipeline, peer_connection=peer_connection, datachannel=datachannel)
        self.__video: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="video", track=track, vad_pipeline=vad_pipeline, asr_pipeline=asr_pipeline, llm_pipeline=llm_pipeline, tts_pipeline=tts_pipeline, peer_connection=peer_connection, datachannel=datachannel)

    @property
    def audio(self) -> MediaStreamTrack:
        """A MediaStreamTrack instance if the player provides audio."""
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """A MediaStreamTrack instance if the player provides video."""
        return self.__video

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"RTCStreamTrack {msg}", *args)
