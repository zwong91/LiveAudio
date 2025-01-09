import os
import wave
from scipy import signal
import aiofiles

from io import BytesIO
import asyncio

import torch
import numpy as np
import pyloudnorm as pyln

# 2^(16-1)
INT16_MAX_ABS_VALUE = 32768.0

def calculate_audio_volume(audio: bytes, sample_rate: int) -> float:
    audio_np = np.frombuffer(audio, dtype=np.int16)
    audio_float = audio_np.astype(np.float64)

    block_size = audio_np.size / sample_rate
    meter = pyln.Meter(sample_rate, block_size=block_size)
    loudness = meter.integrated_loudness(audio_float)

    # Loudness goes from -20 to 80 (more or less), where -20 is quiet and 80 is
    # loud. LUFS (Loudness Units relative to Full Scale)
    loudness = normalize_value(loudness, -20, 80)

    return loudness


def normalize_value(value, min_value, max_value):
    normalized = (value - min_value) / (max_value - min_value)
    normalized_clamped = max(0, min(1, normalized))
    return normalized_clamped


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def int2float(sound):
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound

async def save_audio_to_file(
    audio_data, file_name, audio_dir="audio_files", audio_format="wav",
    channles=1, sample_width=2, sample_rate=16000
):
    """
    Saves the audio data to a file asynchronously.

    :param audio_data: The audio data to save.
    :param file_name: The name of the file.
    :param audio_dir: Directory where audio files will be saved.
    :param audio_format: Format of the audio file.
    :return: Path to the saved audio file.
    """
    
    # Ensure directory exists
    os.makedirs(audio_dir, exist_ok=True)

    file_path = os.path.join(audio_dir, file_name)

    # Async write with aiofiles
    async with aiofiles.open(file_path, 'wb') as wav_file:
        # Writing audio data directly, but first need to convert it to a valid byte format.
        # Mono, 16-bit PCM, 16000 Hz
        with wave.open(file_path, 'wb') as wave_file:
            wave_file.setnchannels(channles)  #default mono audio
            wave_file.setsampwidth(sample_width)
            wave_file.setframerate(sample_rate)
            wave_file.writeframes(audio_data)
    
    return file_path

async def read_audio_file(file_path):
    with wave.open(file_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return frames


def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer



def bytes2NpArrayWith16(frames: bytes | bytearray):
    # Convert the buffer frames to a NumPy array
    audio_array = np.frombuffer(frames, dtype=np.int16)
    # Normalize the array to a [-1, 1] range
    float_data = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
    return float_data


def bytes2TorchTensorWith16(frames: bytes | bytearray):
    float_data = bytes2NpArrayWith16(frames)
    waveform_tensor = torch.tensor(float_data, dtype=torch.float32)
    # don't Stereo, just Mono, reshape(1,-1) (1(channel),size(time))
    if waveform_tensor.ndim == 1:
        # float_data= float_data.reshape(1, -1)
        waveform_tensor = waveform_tensor.reshape(1, -1)
    return waveform_tensor


def npArray2bytes(np_arr: np.ndarray) -> bytearray:
    # Convert a NumPy array to bytes
    bytes_obj = np_arr.tobytes()
    # bytes -> bytearray
    byte_arr = bytearray(bytes_obj)
    return byte_arr


def torchTensor2bytes(tensor: torch.Tensor) -> bytearray:
    # Convert a torch tensor to bytes
    np_arr = tensor.numpy()

    return npArray2bytes(np_arr)


def postprocess_tts_wave_int16(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform with numpy.int16 to bytes
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk * (2**15)
    chunk = chunk.astype(np.int16)
    return chunk.tobytes()

def postprocess_tts_wave(chunk: torch.Tensor | list) -> bytes:
    r"""
    Post process the output waveform with numpy.float32 to bytes
    """
    if isinstance(chunk, list):
        chunk = torch.cat(chunk, dim=0)
    chunk = chunk.clone().detach().cpu().numpy()
    chunk = chunk[None, : int(chunk.shape[0])]
    chunk = np.clip(chunk, -1, 1)
    chunk = chunk.astype(np.float32)
    return chunk.tobytes()


def convertSampleRateTo16khz(audio_data: bytes | bytearray, original_sample_rate):
    if original_sample_rate == 16000:
        return audio_data

    pcm_data = np.frombuffer(audio_data, dtype=np.int16)
    pcm_data_16K = resample_audio(pcm_data, original_sample_rate, 16000)
    audio_data = pcm_data_16K.tobytes()

    return audio_data


def resample_audio(pcm_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    num_samples = int(len(pcm_data) * target_rate / original_rate)
    resampled_audio = signal.resample(pcm_data, num_samples)
    # resampled_audio = signal.resample_poly(pcm_data, target_rate, original_rate)
    return resampled_audio.astype(np.int16)


def convert_sampling_rate_to_16k(input_file, output_file):
    original_rate, data = read(input_file)
    if original_rate == 16000:
        return
    up = 16000
    down = original_rate
    resampled_data = signal.resample_poly(data, up, down)
    write(output_file, 16000, resampled_data.astype(np.int16))
    

# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()
