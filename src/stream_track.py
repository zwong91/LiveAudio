import asyncio
from typing import Optional

import av
from av.frame import Frame
import numpy as np

import logging
from aiortc import MediaStreamTrack, RTCDataChannel
from aiortc.contrib.media import MediaPlayer

class ClientStreamTrack(MediaStreamTrack):
    """
    A media track that receives frames from a RTCClient.
    """

    def __init__(
            self,
            track,
            kind,
            client,
            vad_pipeline,
            asr_pipeline,
            llm_pipeline,
            tts_pipeline,
            peer_connection=None,
            datachannel=None, 
            signaling=None
    ):
        super().__init__()  # don't forget this!
        self.kind = kind
        self.track = track
        self.client = client
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.llm_pipeline = llm_pipeline
        self.tts_pipeline = tts_pipeline
        self.peer_connection = peer_connection
        # server side channel
        self.channel = datachannel

        self.sampling_rate = 16_000
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=self.sampling_rate,
        )

    async def recv(self) -> Frame:
        frame = await self.track.recv()
        frame = self.resampler.resample(frame)[0]
        #frame.to_ndarray().flatten().astype(np.int16)
        frame_array = frame.to_ndarray()
        byte_stream = frame_array[0].astype(np.int16).tobytes()
        self.client.append_audio_data(byte_stream, "default")
        try:
            if self.channel is not None and self.channel.readyState == "open":
                self.client.process_audio(
                    self.channel, self.vad_pipeline, self.asr_pipeline, self.llm_pipeline, self.tts_pipeline
                )
        except Exception as e:
            logging.error(f"Processing error for {self.client.client_id}: {e}")

        return frame

    def playback(self):
        """ Playback Stream Track  
            add response player
        """
        self.track = MediaPlayer("ding.wav", format="wav", loop=False).audio
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(b"s-dingding")
