import os
import wave
import aiofiles
import numpy as np
import asyncio

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
    audio_data, file_name, audio_dir="audio_files", audio_format="wav"
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
            wave_file.setnchannels(1)  # mono audio
            wave_file.setsampwidth(2)
            wave_file.setframerate(16000)
            wave_file.writeframes(audio_data)
    
    return file_path
