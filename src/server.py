import asyncio
import json
import logging
import ssl
import uuid
import base64
import uvicorn
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torchaudio


import ormsgpack

from typing import List
import shutil


from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import MediaStreamTrack, VideoStreamTrack
import av
from av.frame import Frame
import aiohttp

from src.client import Client

relay = MediaRelay()
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
        print(frame)
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
        print("Let's Speech to Speech!")
        self.client.append_audio_data(torchTensor2bytes(frame_array), "default")
        try:
            self.client.process_audio(
                self.datachannel, self.vad_pipeline, self.asr_pipeline, self.llm_pipeline, self.tts_pipeline
            )
        except RuntimeError as e:
            logging.error(f"Processing error for {client.client_id}: {e}")
        return frame


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

class TTSRequest(BaseModel):
    tts_text: str
    vc_uid: str

class TTSManager:
    def __init__(self, tts_pipeline):
        self.task_queue = asyncio.Queue()  # 用于存储任务
        self.processing_tasks = {}  # 用于跟踪任务状态
        self.tts_pipeline = tts_pipeline
        self.lock = asyncio.Lock()  # 用于保护并发

    async def _process_task(self, task_id, text, vc_uid):
        """
        处理队列中的每个 TTS 任务。
        """
        try:
            audio_path = await self.tts_pipeline.text_to_speech(text, vc_uid)
            # 将生成的文件返回给调用者
            self.processing_tasks[task_id] = {'status': 'completed', 'file_path': audio_path, 'media_type': 'audio/wav'}
        except Exception as e:
            # 任务失败时记录
            self.processing_tasks[task_id] = {'status': 'failed', 'error': str(e)}

    async def gen_tts(self, text: str, vc_uid: str):
        """
        启动一个新的任务，返回任务 ID
        """
        task_id = uuid.uuid4().hex[:8]  # 生成任务 ID
        await self.task_queue.put((task_id, text, vc_uid))  # 将任务放入队列
        return task_id

    async def start_processing(self):
        """
        启动一个异步任务处理队列
        """
        while True:
            task_id, text, vc_uid = await self.task_queue.get()  # 从队列获取任务
            await self._process_task(task_id, text, vc_uid)  # 处理任务
            self.task_queue.task_done()  # 标记任务已完成

    async def get_task_result(self, task_id: str):
        """
        获取任务的处理结果
        """
        # 如果任务未处理完成，返回正在处理中
        if task_id not in self.processing_tasks:
            return JSONResponse(content={"status": "pending", "message": "Task is being processed."}, status_code=202)

        task = self.processing_tasks[task_id]

        # Ensure task is a dictionary before accessing
        if isinstance(task, dict):
            # 如果任务已完成，返回文件路径和媒体类型
            if task.get('status') == 'completed':
                return JSONResponse(content={
                    "status": "completed",
                    "file_path": task['file_path'],
                    "media_type": task['media_type'],
                    "message": "Task completed successfully."
                }, status_code=200)
            
            # 如果任务失败，返回错误信息
            elif task.get('status') == 'failed':
                return JSONResponse(content={
                    "status": "failed",
                    "error": task.get('error'),
                    "message": "Task failed during processing."
                }, status_code=500)

        # 如果任务状态不明，返回未知状态
        return JSONResponse(content={
            "status": "unknown",
            "message": "Task status is unknown."
        }, status_code=400)



class Server:
    """
    WebSocket server for real-time audio transcription with VAD and ASR pipelines.
    """

    def __init__(
        self,
        vad_pipeline,
        asr_pipeline,
        llm_pipeline,
        tts_pipeline,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
    ):
        self.vad_pipeline = vad_pipeline
        self.asr_pipeline = asr_pipeline
        self.llm_pipeline = llm_pipeline
        self.tts_pipeline = tts_pipeline
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.connected_clients = {}
        
        self.pcs = set()
        self.app = FastAPI(
            title="Audio AI Server",
            description='',
            version='0.0.1',
            contact={
                "url": ''
            },
            license_info={
                "name": "",
                "url": ''
            }
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 初始化 TTSManager
        self.tts_manager = TTSManager(tts_pipeline)
        self.templates = Jinja2Templates(directory="templates")

        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)

        self.app.get("/asset/{filename}")(self.get_asset_file)
        self.app.post("/generate_accent/{vc_name}")(self.upload_audio_files)
        self.app.post("/generate_tts")(self.generate_tts)
        self.app.get("/get_task_result/{task_id}")(self.get_task_result)
        self.app.get("/health")(self.health)

        self.app.websocket("/stream")(self.websocket_endpoint)
        self.app.websocket("/stream-vc")(self.websocket_endpoint)
        
        self.app.post("/offer")(self.offer_endpoint)

    async def startup(self):
        """Called on startup to set up additional services."""
        logging.info(f"Starting server at {self.host}:{self.port}")
        # 启动任务处理的后台任务
        asyncio.create_task(self.tts_manager.start_processing())

    async def shutdown(self):
        logging.info(f"shutdown server ...")
        # Shutdown tasks: Close WebRTC connections
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()

    async def offer_endpoint(self, request: Request):
        
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # Generate a unique sessionid
        sessionid = str(uuid.uuid4())
        client = Client(sessionid, self.sampling_rate, self.samples_width)
        # Create a new RTCPeerConnection
        pc = RTCPeerConnection()
        s2s_response = pc.createDataChannel(
            label="response",
            ordered=True,
        )
        self.pcs.add(pc)

        # signaling = create_signaling()
        # recorder = MediaBlackhole()

        @pc.on("datachannel")
        def on_datachannel(channel):
            logging.info(f"on_datachannel {channel}")

            @channel.on("message")
            def on_message(message):
                if isinstance(message, str):
                    channel.send("pong" + message)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.info(f"Connection state is {pc.connectionState}")
            if pc.connectionState in ["failed", "closed"]:
                await pc.close()
                self.pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logging.info(f"Track {track.kind} received")
            if track.kind == "audio":
                audio_track = ClientStreamTrack(
                    relay.subscribe(
                        track=track,
                    ),
                    "audio",
                    client,
                    self.vad_pipeline,
                    self.asr_pipeline,
                    self.llm_pipeline,
                    self.tts_pipeline,
                    peer_connection=pc,
                    datachannel=s2s_response, 
                )
                pc.addTrack(audio_track)
                # Add tracks
                #pc.addTrack(stream_track.video)
                #pc.addTrack(stream_track.audio)

            @track.on("ended")
            async def on_ended():
                logging.info(f"Track {track.kind} ended")
                #await recorder.stop()
                    
        # Add transceivers
        # pc.addTransceiver('video', direction='sendonly')
        pc.addTransceiver('audio', direction='sendrecv')

        # Set codec preferences for video
        for transceiver in pc.getTransceivers():
            if transceiver.kind == 'video':
                capabilities = RTCRtpSender.getCapabilities('video')
                preferences = [codec for codec in capabilities.codecs if codec.name in ('H264', 'VP8')]
                transceiver.setCodecPreferences(preferences)
                transceiver.direction = 'sendonly'
            elif transceiver.kind == 'audio':
                transceiver.direction = 'sendrecv'

        await pc.setRemoteDescription(offer)
        #await recorder.start()

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})

    async def push(self, push_url):
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        sessionid = str(uuid.uuid4())
        client = Client(sessionid, self.sampling_rate, self.samples_width)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logging.info(f"Track {track.kind} received")
            if track.kind == "audio":
                stream_track = ClientStreamTrack(
                    relay.subscribe(
                        track=track,
                        ),
                    "audio",
                    client,
                    self.vad_pipeline,
                    self.asr_pipeline,
                    self.llm_pipeline,
                    self.tts_pipeline,
                    peer_connection=pc,
                    datachannel=s2s_response,
                )
                pc.addTrack(stream_track.audio)
                #pc.addTrack(stream_track.video)

            @track.on("ended")
            async def on_ended():
                logging.info(f"Track {track.kind} ended")

        await pc.setLocalDescription(await pc.createOffer())
        answer = await post(push_url, {"sdp": pc.localDescription.sdp})
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()

        logging.info(f"accept client: {websocket.client}")
        client_id = str(uuid.uuid4())
        client = Client(client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client
        logging.info(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        finally:
            del self.connected_clients[client_id]
            logging.info(f"Client {client_id} disconnected")

    async def handle_audio(self, client, websocket):
        sessionid = None  # To store the sessionid
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                #message = await websocket.receive_bytes()
                # Decode the MessagePack data
                #data = ormsgpack.unpackb(message)
                msg_type = data.get('event')
                if msg_type == 'session':
                    sessionid = data.get("sessionid")
                    if sessionid is not None:
                        # Optionally, send confirmation back to the client
                        await websocket.send_json({"type": "session_ack", "sessionid": sessionid})
                    else:
                        await websocket.send_json({"type": "error", "message": "No rtc sessionid provided."})              

                elif msg_type == 'start':
                    request_data = data.get('request', {})
                    chunk = request_data.get('audio')
                    audio_data = base64.b64decode(chunk)
                    latency = request_data.get('latency')
                    format = request_data.get('format')
                    prosody = request_data.get('prosody', {})
                    vc_uid = request_data.get('vc_uid')

                    # Print or process the extracted data
                    logging.debug(f"Audio Data: {audio_data}, Latency: {latency}, Format: {format}")
                    logging.debug(f"Prosody: {prosody}, VC UID: {vc_uid}")

                    #TODO: Pass the message to your processing function
                    client.append_audio_data(audio_data, vc_uid)
                    # 异步task处理音频
                    self._process_audio(client, websocket)

                elif msg_type == 'stop':
                    if sessionid is not None:
                        logging.info(f"Session {sessionid} ended.")
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

            except WebSocketDisconnect as e:
                logging.error(f"Connection with {client.client_id} closed: {e}")
                break
            except Exception as e:
                logging.error(f"Error handling audio for {client.client_id}: {e}")
                break

    def _process_audio(self, client, websocket):
        try:
            client.process_audio(
                websocket, self.vad_pipeline, self.asr_pipeline, self.llm_pipeline, self.tts_pipeline
            )
        except RuntimeError as e:
            logging.error(f"Processing error for {client.client_id}: {e}")

    async def handle_text_message(self, client, message):
        """Handles incoming JSON text messages for config updates."""
        try:
            config = json.loads(message)
            if config.get("type") == "config":
                client.update_config(config["data"])
                logging.debug(f"Updated config: {client.config}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode config message: {e}")

    async def get_asset_file(self, filename: str):
        file_path = os.path.join('/asset', filename)
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        mime_types = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.webm': 'audio/webm'
        }
        ext = os.path.splitext(filename)[1].lower()
        media_type = mime_types.get(ext, 'application/octet-stream')

        return FileResponse(
            path=file_path,
            media_type=media_type,
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Disposition': 'inline'
            }
        )

    async def post(self, url, data):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    return await response.text()
        except aiohttp.ClientError as e:
            print(f'Error: {e}')


    async def upload_audio_files(self, vc_name: str, files: List[UploadFile] = File(...)):
        file_paths = []
        file_uuid = uuid.uuid4().hex[:8]
        
        # 检查文件是否为空以及大小是否超过20MB
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="File is empty")
            
            # 检查文件大小是否超过20MB
            file_size = await self.get_file_size(file)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File is too large. Max size is 20MB")
            
            filename = file.filename
            file_name_without_ext, file_extension = os.path.splitext(filename)
            
            # 为每个文件生成一个独特的文件路径
            file_location = os.path.join("vc", f"{file_uuid}_{vc_name}{file_extension}")
            file_paths.append(file_location)
            
            # 保存文件到磁盘
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # 处理音频文件进行声音清理
        cleaned_file_paths = []
        for file_path in file_paths:
            cleaned_file_path = await self.clean_audio(file_path, False)
            cleaned_file_paths.append(cleaned_file_path)

        # 返回清理后的文件路径和 vc_uid
        return {"vc_uid": file_uuid, "file_paths": cleaned_file_paths}

    async def get_file_size(self, file: UploadFile) -> int:
        """ 获取上传文件的大小 """
        # 将文件指针移动到文件的开始位置
        file.file.seek(0, os.SEEK_END)
        size = file.file.tell()  # 获取文件大小
        file.file.seek(0)  # 恢复文件指针位置
        return size

    async def clean_audio(self, speaker_wav: str, voice_cleanup: bool) -> str:
        """
        使用ffmpeg进行音频清理, 包括低通、高通滤波、去除静音等
        针对麦克风输入进行过滤，因为麦克风通常会有背景噪音，可能会在开始和结束时有静音。快速过滤，效果一般
        """
        lowpassfilter = True
        trim = True

        # Apply all on demand
        if lowpassfilter:
            lowpass_highpass = "lowpass=8000,highpass=75,"
        else:
            lowpass_highpass = ""

        if trim:
            # better to remove silence in beginning and end for microphone
            trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
        else:
            trim_silence = ""

        if voice_cleanup:
            try:
                # Generate a unique filename for the cleaned audio
                out_filename = f"{speaker_wav}_{str(uuid.uuid4())}.cleaned.wav"

                # ffmpeg command for filtering the audio
                shell_command = f"ffmpeg -y -i {speaker_wav} -af {lowpass_highpass}{trim_silence} {out_filename}".split(" ")

                # Run the ffmpeg command
                command_result = subprocess.run(
                    [item for item in shell_command],
                    capture_output=False,
                    text=True,
                    check=True,
                )
                print(f"Filtered audio saved to: {out_filename}")
                return out_filename
            except subprocess.CalledProcessError:
                # There was an error in the ffmpeg command
                print("Error: failed to filter audio, returning original file")
                return speaker_wav
        else:
            # If no cleanup is requested, return the original file
            return speaker_wav

    async def generate_tts(self, request: TTSRequest):
        task_id = await self.tts_manager.gen_tts(request.tts_text, request.vc_uid)
        return {"task_id": task_id}

    async def get_task_result(self, task_id: str):
        result = await self.tts_manager.get_task_result(task_id)
        return result

    async def health(self):
        return {"status": "ojbk"}

    def create_uvicorn_server(self):
        """Creates and returns a Uvicorn server instance."""
        uvicorn_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            ssl_certfile=self.certfile,
            ssl_keyfile=self.keyfile,
            loop="uvloop",
            log_level="info",
            workers=os.cpu_count(),
            limit_concurrency=1000,
            limit_max_requests=10000,
            backlog=2048
        )
        server = uvicorn.Server(uvicorn_config)
        return server

    def start(self):
        """Start the WebSocket server."""
        if self.certfile and self.keyfile:
            logging.info(f"Starting secure WebSocket server on {self.host}:{self.port}")
        else:
            logging.info(f"Starting WebSocket server on {self.host}:{self.port}")

        server = self.create_uvicorn_server()
        server.run()
