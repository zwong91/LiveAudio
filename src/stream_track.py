import asyncio
from typing import Optional

import av
from av.frame import Frame
import numpy as np

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
            peer_connection,
            datachannel, 
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
        self.dc = datachannel

        self.response_ready = False
        self.previous_response_silence = False
        
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
        frame_array = frame_array[0].astype(np.int16)
        self.client.append_audio_data(frame_array.tobytes(), "default")
        try:
            if self.dc.readyState == "open":
                self.client.process_audio(
                    self.dc, self.vad_pipeline, self.asr_pipeline, self.llm_pipeline, self.tts_pipeline
                )
        except Exception as e:
            logging.error(f"Processing error for {self.client.client_id}: {e}")

        return frame
    

        """ Playback Stream Track  
            add response player
        """
    def select_track(self):
        if self.response_ready:
            self.track = MediaPlayer("xtts2_out.wav", format="wav", loop=False).audio
        else:
            self.track = MediaPlayer("silence.wav", format="wav", loop=False).audio
        if self.dc.readyState == "open":
            if self.response_ready:
                self.dc.send("response")
                self.previous_response_silence = False
            else:
                if not self.previous_response_silence:
                    self.dc.send("silence")
                    self.previous_response_silence = True