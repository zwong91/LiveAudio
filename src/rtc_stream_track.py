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

logging.basicConfig()
logger = logging.getLogger(__name__)

class ClientStreamTrack(MediaStreamTrack):
    """
    A media track that receives frames from a RTCClient.
    """

    def __init__(self, kind, track, peer_connection, datachannel, signaling=None):
        super().__init__()  # don't forget this!
        self.kind = kind
        self.track = track        
        self.peer_connection = peer_connection
        self.datachannel = datachannel
        # self.signaling = signaling
        
        self.sampling_rate = 16_000
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sampling_rate,
        )
        self.buffer = torch.tensor(
            [],
            dtype=torch.float32,
        )

    async def recv(self) -> Frame:
        frame = await self.track.recv()
        # print(frame)
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

        if not speech_prob is None:
            is_speech = speech_prob >= 0.4
            if (
                np.mean(self.segments) <= 0.4
                and self.is_activated
                and len(self.segments) == self.segments_amount
            ):
                print("Let's Speech to text!")
                # asyncio.create_task(
                #     asyncio.to_thread(self.extract_text),
                # )

            # 新的数据
            self.buffer = frame_array

        return frame
    
    def extract_text(self):
        print("Extract text")

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
        
        pass

class RTCStreamTrack:
    def __init__(self, track, peer_connection, datachannel, signaling=None):

        # Examine streams
        self.__audio: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="audio", track=track, peer_connection=peer_connection, datachannel=datachannel)
        self.__video: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="video", track=track, peer_connection=peer_connection, datachannel=datachannel)

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
