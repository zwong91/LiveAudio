# musereal.py

import math
import fractions
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import torch.nn.functional as F
import glob
import pickle
import copy
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import multiprocessing as mp
import resampy
import av
from fractions import Fraction
import soundfile as sf
from tqdm import tqdm

import asyncio
from av import AudioFrame, VideoFrame
import gc

import signal

class BaseProcessor:
    def __init__(self, opt):
        self.fps = 30
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples per chunk (20ms * 16000 / 1000)

        self.speaking = False

        self.recording = False
        self.recordq_video = Queue()
        self.recordq_audio = Queue()

        self.curr_state = 0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def put_audio_file(self, filebytes):
        input_stream = BytesIO(filebytes)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk:
            self.put_audio_frame(stream[idx:idx + self.chunk])
            streamlen -= self.chunk
            idx += self.chunk

    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)  # [T*sample_rate,] float64
        print(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def is_speaking(self) -> bool:
        return self.speaking

    def init_customindex(self):
        self.curr_state = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def start_recording(self, path):
        """Start recording video"""
        if self.recording:
            return
        self.recording = True
        self.recordq_video.queue.clear()
        self.recordq_audio.queue.clear()
        self.container = av.open(path, mode="w")

        process_thread = Thread(target=self.record_frame, args=())
        process_thread.start()

    def record_frame(self):
        videostream = self.container.add_stream("libx264", rate=25)
        videostream.codec_context.time_base = Fraction(1, 25)
        audiostream = self.container.add_stream("aac")
        audiostream.codec_context.time_base = Fraction(1, 16000)
        init = True
        framenum = 0
        while self.recording:
            try:
                videoframe = self.recordq_video.get(block=True, timeout=1)
                videoframe.pts = framenum
                videoframe.dts = videoframe.pts
                if init:
                    videostream.width = videoframe.width
                    videostream.height = videoframe.height
                    init = False
                packet = videostream.encode(videoframe)
                if packet:
                    self.container.mux(packet)
                for _ in range(2):  # Assuming 2 audio frames per video frame
                    audioframe = self.recordq_audio.get(block=True, timeout=1)
                    audioframe.pts = int(round((framenum * 2) * 0.02 / audiostream.codec_context.time_base))
                    audioframe.time_base = Fraction(1, 16000)
                    packet = audiostream.encode(audioframe)
                    if packet:
                        self.container.mux(packet)
                framenum += 1
            except queue.Empty:
                print('record queue empty,')
                continue
            except Exception as e:
                print(f"Recording error: {e}")
                break
         # Flush and close streams
        packet = videostream.encode(None)
        if packet:
            self.container.mux(packet)
        packet = audiostream.encode(None)
        if packet:
            self.container.mux(packet)
        self.container.close()
        self.recordq_video.queue.clear()
        self.recordq_audio.queue.clear()
        print('record thread stop')

    def stop_recording(self):
        """Stop recording video"""
        if not self.recording:
            return
        self.recording = False

    def mirror_index(self, size, index):
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def get_audio_stream(self, audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx + self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  # Switch to silent state if audio ends
        return stream

    def set_curr_state(self, audiotype, reinit):
        print('set_curr_state:', audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def __mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


@torch.no_grad()
def inference(render_event, batch_size, latents_out_path, audio_feat_queue, audio_out_queue, res_frame_queue, quit_event):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
   
    # Cleanup GPU resources after inference is done
    torch.cuda.empty_cache()
    print("Inference process stopped and cleaned up GPU resources.")
    gc.collect()  # Garbage collection to free up any remaining memory


# import torch.multiprocessing as tmp

# # Ensure the 'fork' method is used
# tmp.set_start_method('fork')

class RTCProcessor(BaseProcessor):
    _audio_processor = None
    _diffusion_model = None

    @torch.no_grad()
    def __init__(self, opt):
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps
        self.video_path = ''
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)

        self.render_event = mp.Event()
        self.quit_event = mp.Event()  # New quit event to stop inference gracefully
        
        self.loop = asyncio.get_event_loop()
        
        # # Register signal handlers to clean up on reload or termination
        # signal.signal(signal.SIGINT, self.cleanup_signal_handler)
        # signal.signal(signal.SIGTERM, self.cleanup_signal_handler)
        
    def stop_inference(self):
        """Stop the inference process and clean up resources."""
        print("Stopping inference process...")
        self.quit_event.set()  # Signal the quit event to stop the loop
        if self.process.is_alive():
            self.process.join()    # Ensure the process is properly joined
        self.cleanup()
        
    def cleanup(self):
        """Clean up any remaining GPU resources, terminate subprocesses, and clear memory."""
        torch.cuda.empty_cache()  # Frees up the unused GPU memory
        gc.collect()  # Garbage collection to free up any remaining memory
        if self.process.is_alive():
            self.process.terminate()  # Ensure the process is terminated
        print("Cleaned up GPU resources and terminated processes.")

    def cleanup_signal_handler(self, signum, frame):
        """Signal handler to cleanup processes on Uvicorn reload or termination."""
        print(f"Received signal {signum}. Cleaning up before reload...")
        self.stop_inference()
        
    def __del__(self):
        """Ensure cleanup is performed when the object is deleted."""
        self.stop_inference()

    @torch.no_grad()
    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    @torch.no_grad()
    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):
        """Process frames and put them into the video and audio tracks."""
        pass

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        self.render_event.set()
        while not quit_event.is_set():
            self.asr.run_step()
            if video_track._queue.qsize() >= 1.5 * self.opt.batch_size:
                print('Sleeping, video queue size:', video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        self.render_event.clear()
        print('musereal render thread stop')

    @torch.no_grad()
    async def get_video_frame(self):
        """Retrieve the next video frame."""
        try:
            frame = await self.video_queue.get()
            return frame
        except Exception as e:
            print(f"Error getting video frame: {e}")
            return None

    @torch.no_grad()
    async def get_audio_frame(self):
        """Retrieve the next audio frame."""
        try:
            frame = await self.audio_queue.get()
            return frame
        except Exception as e:
            print(f"Error getting audio frame: {e}")
            return None
        
    def start_rendering(self):
        """Start the rendering process."""
        self.video_queue = asyncio.Queue(maxsize=100)
        self.audio_queue = asyncio.Queue(maxsize=200)
        self.process_thread = Thread(target=self.process_frames, args=(self.quit_event, self.loop, self.audio_queue, self.video_queue))
        self.process_thread.start()
        self.render_event.set()
        
    def stop_rendering(self):
        """Stop the rendering process."""
        self.quit_event.set()
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.render_event.clear()

    def put_audio_frame(self, audio_chunk):
        """Override to handle incoming audio frames."""
        super().put_audio_frame(audio_chunk)

    def put_audio_file(self, filebytes):
        """Override to handle audio files."""
        super().put_audio_file(filebytes)

    def set_curr_state(self, audiotype, reinit):
        super().set_curr_state(audiotype, reinit)