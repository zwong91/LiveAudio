# #!/usr/bin/env python
import asyncio
import json
import aiohttp
import pyaudio
import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import websockets
import os

pc = None

async def send_offer(session, url):
    global pc

    pc = RTCPeerConnection()

    # Add transceivers for video (recvonly) and audio (sendonly)
    pc.addTransceiver('video', direction='recvonly')
    pc.addTransceiver('audio', direction='sendonly')

    # Create an offer and set the local description
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Wait for ICE gathering to complete
    await ice_gathering_complete(pc)

    # Send the offer to the /offer endpoint on the server
    async with session.post(f'{url}/offer', json={
        'sdp': pc.localDescription.sdp,
        'type': pc.localDescription.type
    }) as response:
        answer = await response.json()

    # Set the remote description
    sessionid = answer.get('sessionid')
    print(f"Session ID: {sessionid}")
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'], type=answer['type']))

    return sessionid, pc

async def ice_gathering_complete(pc):
    if pc.iceGatheringState == 'complete':
        return

    ice_complete = asyncio.Future()

    @pc.on('icegatheringstatechange')
    def check_ice():
        if pc.iceGatheringState == 'complete':
            ice_complete.set_result(True)

    await ice_complete

async def send_audio_data(websocket, sessionid, audio_data):
    # Encode audio data to base64 string to ensure safe transmission
    await websocket.send(json.dumps({
        'type': 'audio',
        'sessionid': sessionid,
        'data': audio_data.decode('latin-1')  # Convert bytes to string
    }))

async def receive_video(track):
    # Optionally, process the received video frames
    while True:
        frame = await track.recv()
        # Process the video frame (e.g., display, save, etc.)
        # For this example, we will just print a message
        print("Received video frame")

async def main(audio_file_path="data/audio/elon.wav"):
    url = "http://localhost:8010"  # Replace with your server's address
    ws_url = "ws://localhost:8010/ws"  # WebSocket URL

    async with aiohttp.ClientSession() as session:
        # Establish WebRTC connection
        sessionid, pc = await send_offer(session, url)

        # Set up WebSocket connection
        async with websockets.connect(ws_url) as websocket:
            # Send session ID to associate with server-side processing
            await websocket.send(json.dumps({'type': 'session', 'sessionid': sessionid}))

            # Set up audio source (microphone or pre-recorded file)
            audio_source = None
            # p = pyaudio.PyAudio()
            # try:
            #     # Try to open the microphone for audio input
            #     stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            #     audio_source = 'microphone'
            #     print("Using microphone for audio input")
            # except OSError as e:
            #     print(f"Error opening microphone: {e}")
            #     # No microphone available, use a pre-recorded audio file
            #     audio_source = 'file'
            #     print("Using pre-recorded audio file for audio input")
            #     if not os.path.exists(audio_file_path):
            #         print(f"Audio file not found at {audio_file_path}")
            #         return
            #     audio_player = MediaPlayer(audio_file_path)
            #     audio_stream = audio_player.audio
            
            audio_source = 'file'
            print("Using pre-recorded audio file for audio input")
            if not os.path.exists(audio_file_path):
                print(f"Audio file not found at {audio_file_path}")
                return
            audio_player = MediaPlayer(audio_file_path, loop=True)
            audio_stream = audio_player.audio

            # Start receiving video
            @pc.on("track")
            async def on_track(track):
                if track.kind == "video":
                    print("Received video track")
                    await receive_video(track)

            try:
                if audio_source == 'microphone':
                    while True:
                        # Read audio data from microphone
                        audio_data = stream.read(1024, exception_on_overflow=False)
                        # Send audio data through WebSocket
                        await send_audio_data(websocket, sessionid, audio_data)
                elif audio_source == 'file':
                    while True:
                        # Read audio frame from the media player
                        frame = await audio_stream.recv()
                        # Get audio data as bytes
                        audio_data = bytes(frame.planes[0])
                        # Send audio data through WebSocket
                        await send_audio_data(websocket, sessionid, audio_data)
                        # Sleep to match the audio frame duration
                        await asyncio.sleep(frame.time_base * frame.samples)
            except KeyboardInterrupt:
                print("Stopping...")
            finally:
                # # Clean up
                # if audio_source == 'microphone':
                #     stream.stop_stream()
                #     stream.close()
                #     p.terminate()
                # elif audio_source == 'file':
                #     audio_player.close()
                await pc.close()
                await websocket.close()

def run_async_main():
    asyncio.run(main())

if __name__ == "__main__":
    run_async_main()

