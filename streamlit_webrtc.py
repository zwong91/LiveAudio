import os
os.environ["no_proxy"] = "localhost,172.16.87.75,127.0.0.1"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import cv2
import asyncio
import aiohttp
import threading
import queue
from aiortc.contrib.media import MediaPlayer

# RTC Configuration with a public STUN server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define the WebRTC client settings
WEBRTC_CLIENT_SETTINGS = {
    "mode": WebRtcMode.SENDRECV,
    "rtc_configuration": RTC_CONFIGURATION,
    "media_stream_constraints": {"video": True, "audio": True},
}

# Queue for inter-thread communication
audio_queue = queue.Queue()
video_queue = queue.Queue()

def main():
    st.title("Real-Time Video Transfer with WebRTC")

    # Option to select audio source
    audio_option = st.radio("Select Audio Source", ("Microphone", "Audio File"))

    if audio_option == "Microphone":
        st.write("Using microphone for audio input.")

        # Start the WebRTC streamer for audio and video
        webrtc_ctx = webrtc_streamer(
            key="audio-video",
            **WEBRTC_CLIENT_SETTINGS,
            audio_receiver_size=1024,
            video_receiver_size=640,
            video_frame_callback=video_frame_callback,
            audio_frame_callback=audio_frame_callback,
            async_processing=True,
        )

    else:
        st.write("Upload an audio file to generate video.")
        audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "aac"])
        if audio_file is not None:
            st.success("Audio file uploaded successfully.")
            # Play the uploaded audio file
            st.audio(audio_file)

            # Process the audio file and start streaming video
            # Create a separate thread to handle the audio processing and streaming
            threading.Thread(target=process_audio_file, args=(audio_file,)).start()

def video_frame_callback(frame):
    # Receive video frames from the server and display them
    img = frame.to_ndarray(format="bgr24")
    # For demonstration, we'll draw a rectangle on the frame
    cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def audio_frame_callback(frame):
    # Send audio data to the server for processing
    audio_data = frame.to_ndarray()
    # Place the audio data in the queue to be sent to the server
    audio_queue.put(audio_data)
    # Return the frame unmodified
    return frame

def process_audio_file(audio_file):
    # Read the audio file and send it to the server
    audio_bytes = audio_file.read()
    # Save the audio file locally
    with open("temp_audio_file", "wb") as f:
        f.write(audio_bytes)
    # Use MediaPlayer to read the audio file
    player = MediaPlayer("temp_audio_file", loop=True)
    audio_stream = player.audio

    # Set up an asyncio event loop for asynchronous operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_audio_data(audio_stream))
    loop.close()


async def send_audio_data(audio_stream):
    # Connect to the server's WebSocket endpoint
    ws_url = "ws://localhost:8010/ws"  # Adjust the URL to match your server
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(ws_url) as websocket:
            try:
                while True:
                    # Read audio frame from the media player
                    frame = await audio_stream.recv()
                    if frame is None:
                        break  # End of stream
                    # Get audio data as bytes
                    audio_data = bytes(frame.planes[0])
                    # Encode audio data to base64 string to ensure safe transmission
                    await websocket.send_json({
                        'type': 'audio',
                        'data': audio_data.decode('latin-1')  # Convert bytes to string
                    })
                    # Sleep to match the audio frame duration
                    await asyncio.sleep(frame.time_base * frame.samples)
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
