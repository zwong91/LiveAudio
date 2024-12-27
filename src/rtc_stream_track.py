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

    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self._started = False
        self._timestamp = 0
        self._start_time = None

    async def recv(self) -> Frame:
        if not self._started:
            loop = asyncio.get_running_loop()
            self._player._start(self, loop)
            self._started = True

        try:
            frame = await self._queue.get()
            if frame is None:
                await asyncio.sleep(0)
                self.stop()
                raise MediaStreamError("End of stream")

            # Set timestamp and time_base
            if self._start_time is None:
                self._start_time = time.time()
                self._timestamp = 0
            else:
                if self.kind == 'audio':
                    self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                else:
                    self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)

            frame.pts = self._timestamp
            frame.time_base = AUDIO_TIME_BASE if self.kind == 'audio' else VIDEO_TIME_BASE

            return frame
        except asyncio.CancelledError:
            self.stop()
            raise
        except Exception as e:
            logger.error(f"Error while receiving frame: {e}")
            raise MediaStreamError("Error in receiving frame")

    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def player_worker_thread(quit_event, loop, container, audio_track, video_track):
    """Worker thread to render and process frames."""
    container.render(quit_event, loop, audio_track, video_track)


class RTCStreamTrack:
    def __init__(self, rtc_processor, format=None, options=None, timeout=None, loop=False, decode=True):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # Examine streams
        self.__started: Set[ClientStreamTrack] = set()
        self.__audio: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="audio")
        self.__video: Optional[ClientStreamTrack] = ClientStreamTrack(self, kind="video")

        self.__container = rtc_processor

    @property
    def audio(self) -> MediaStreamTrack:
        """A MediaStreamTrack instance if the player provides audio."""
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """A MediaStreamTrack instance if the player provides video."""
        return self.__video

    def _start(self, track: ClientStreamTrack, loop: asyncio.AbstractEventLoop) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    loop,
                    self.__container,
                    self.__audio,
                    self.__video
                ),
            )
            self.__thread.start()

    def _stop(self, track: ClientStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            # Clean up container if needed
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"RTCStreamTrack {msg}", *args)
