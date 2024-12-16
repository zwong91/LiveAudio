import os
from os import remove
import time

import webrtcvad

from .vad_interface import VADInterface

'''
WebRTC VAD 期望输入的音频帧长度为 10ms、20ms 或 30ms 的音频数据，
对于 16kHz 采样率，这对应于 160、320 或 480 个采样点。
'''

class WebRTCVAD(VADInterface):
    """
    WebRTCVad-based implementation of the VADInterface.
    """

    def __init__(self, **kwargs):
        """
        Initializes Webrtc's VAD pipeline.

        Args:
            model_name (str): The model name for WebRTCVAD.
        """
        idle_time = 0.5
        self.sample_rate = 16000
        self.frame_size = 360
        self.bytes_per_sample = 2
        self.idle_cut = (self.sample_rate * idle_time) / self.frame_size  # chunk audio if no voice for idle_time seconds
        self.last_voice_activity = 0
        self.vad = webrtcvad.Vad(3)  # 使用 WebRTC VAD，敏感度设置为 3


    def convert_buffer_size(self, audio_frame):
        while len(audio_frame) < (self.frame_size * self.bytes_per_sample):
            audio_frame = audio_frame + b'\x00'
        return audio_frame

    def voice_activity_detection(self, audio_frame):
        converted_frame = self.convert_buffer_size(audio_frame)
        is_speech = self.vad.is_speech(converted_frame, sample_rate=self.sample_rate)
        if is_speech:
            self.last_voice_activity = 0
            return "1"
        else:
            if self.last_voice_activity >= self.idle_cut:
                self.last_voice_activity = 0
                return "X"
            else:
                self.last_voice_activity += 1
                return "_"

    async def detect_activity(self, client):
        start_time = time.time()
        # 确保音频帧长度为 480 个采样点
        frame_size = 480 * 2  # 480 个采样点，每个采样点 2 个字节
        idx = 0
        while idx + frame_size <= len(client.scratch_buffer):
            chunk = client.scratch_buffer[idx: idx + frame_size]
            idx += frame_size
            vad_result = self.voice_activity_detection(chunk)
            #print("vad result: {}", vad_result)
            if vad_result == "1":
                # 语音活动检测到，继续累积数据
                continue
            elif vad_result == "X":
                # 语音活动结束，返回结果
                end_time = time.time()
                print(f"VAD spent time: {end_time - start_time:.4f} seconds")
                return [{"start": 0, "end": 0, "confidence": 1.0}]
        
        end_time = time.time()
        print(f"VAD spent time: {end_time - start_time:.4f} seconds")
        #语音尚未结束，继续等待数据
        return [{"start": 0, "end": 1e10, "confidence": 1.0}]
