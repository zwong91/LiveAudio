import asyncio
import json
import logging
import ssl
import subprocess
import uuid
import base64
import uvicorn
import signal
import os

from typing import Optional, List
import shutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import MediaStreamTrack, VideoStreamTrack


from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
import ngrok

from .utils.audio_utils import read_audio_file, ulaw_to_pcm16k

import aiohttp
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(override=True)

USERNAME = os.getenv('USERNAME')
CREDENTIAL = os.getenv('CREDENTIAL')

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_API_SECRET = os.getenv('TWILIO_API_SECRET')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

ngrok.set_auth_token(os.getenv("NGROK_AUTHTOKEN"))
NGROK_URL = os.getenv("NGROK_URL", "https://ngrok.io")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
    raise ValueError('Missing Twilio configuration. Please set it in the .env file.')

from src.client import Client
from src.stream_track import ClientStreamTrack

class TTSRequest(BaseModel):
    tts_text: str
    vc_uid: str

class TTSRequestV1(BaseModel):
    tts_text: str
    vc_uid: str
    speed:  Optional[float]

class TTSManager:
    def __init__(self, tts):
        self.task_queue = asyncio.Queue()  # 用于存储任务
        self.processing_tasks = {}  # 用于跟踪任务状态
        self.tts = tts
        self.lock = asyncio.Lock()  # 用于保护并发

    async def _process_task(self, task_id, text, vc_uid, speed):
        """
        处理队列中的每个 TTS 任务。
        """
        try:
            audio_path = await self.tts.text_to_speech(text, vc_uid, speed)
            # 将生成的文件返回给调用者
            self.processing_tasks[task_id] = {'status': 'completed', 'file_path': audio_path, 'media_type': 'audio/wav'}
        except Exception as e:
            # 任务失败时记录
            self.processing_tasks[task_id] = {'status': 'failed', 'error': str(e)}

    async def gen_tts(self, text: str, vc_uid: str, speed: Optional[float] = 1.0):
        """
        启动一个新的任务，返回任务 ID
        """
        task_id = uuid.uuid4().hex[:8]  # 生成任务 ID
        await self.task_queue.put((task_id, text, vc_uid, speed))  # 将任务放入队列
        return task_id

    async def start_processing(self):
        """
        启动一个异步任务处理队列
        """
        while True:
            task_id, text, vc_uid, speed = await self.task_queue.get()  # 从队列获取任务
            await self._process_task(task_id, text, vc_uid, speed)  # 处理任务
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
        filter,
        asr,
        vad,
        eou,
        llm,
        tts,
        host="localhost",
        port=8765,
        sampling_rate=16000,
        samples_width=2,
        certfile=None,
        keyfile=None,
        whip_url=None,
    ):
        self.filter = filter
        self.asr = asr
        self.vad = vad
        self.eou = eou
        self.llm = llm
        self.tts = tts
        self.host = host
        self.port = port
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.certfile = certfile
        self.keyfile = keyfile
        self.whip_url = whip_url
        self.connected_clients = {}

        self.relay = MediaRelay()
        self.pcs = set()
        self.app = FastAPI(
            title="Voice Agent",
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

        self.tts_manager = TTSManager(tts)
        self.templates = Jinja2Templates(directory="templates")

        self.app.add_event_handler("startup", self.startup)
        #self.app.add_event_handler("shutdown", self.shutdown)

        self.app.get("/v1/asset/{filename}")(self.get_asset_file)
        self.app.post("/v1/generate_accent/{vc_name}")(self.upload_audio_files)
        self.app.post("/v1/generate_tts")(self.generate_tts)
        self.app.get("/v1/get_task_result/{task_id}")(self.get_task_result)
        self.app.get("/v1/health")(self.health)

        self.app.post("/offer")(self.offer_endpoint)
        self.app.post("/live")(self.live)

        self.app.websocket("/media-stream")(self.websocket_endpoint)

        self.app.post("/twilio/inbound_call")(self.handle_incoming_call)

        self.app.add_api_route(
            "/twilio/outbound_call",
            self.handle_outgoing_call,
            methods=["GET", "POST"]
        )

        self.app.post("/make_call")(self.make_call)


    async def startup(self):
        """Called on startup to set up additional services."""
        logging.debug(f"Starting server at {self.host}:{self.port}")
        # 启动任务处理的后台任务
        asyncio.create_task(self.tts_manager.start_processing())

    async def offer_endpoint(self, request: Request):

        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # Generate a unique sessionid
        sessionid = str(uuid.uuid4())
        use_webrtc = True
        client = Client(use_webrtc, sessionid, self.sampling_rate, self.samples_width)
        # STUN 和 TURN 服务器配置
        #result = await self.turn({'ttl': 86400})
        ice_servers = [
            RTCIceServer(
                urls=["stun:gtp.aleopool.cc:3478"]
            ),
            RTCIceServer(
                urls=["turn:gtp.aleopool.cc:3478"],
                username=USERNAME,
                credential=CREDENTIAL,
            ),
            # RTCIceServer(
            #     urls=["stun:stun.cloudflare.com:3478",
            #         "turn:turn.cloudflare.com:3478?transport=udp",
            #         "turn:turn.cloudflare.com:3478?transport=tcp",
            #         "turns:turn.cloudflare.com:5349?transport=tcp"],
            #     username=result['username'],
            #     credential=result["credential"],
            # ),
        ]

        # 使用 RTCConfiguration 配置 ICE 服务器
        config = RTCConfiguration(iceServers=ice_servers)

        # 创建一个新的 RTCPeerConnection 并传递 RTCConfiguration
        pc = RTCPeerConnection(configuration=config)
        # Create a new DataChannel after the peer connection is created
        s2s_response = pc.createDataChannel(
            label="s-events",
            ordered=True,
        )
        self.pcs.add(pc)
        logging.debug(f"Peer Connection Created for: {request.client.host}")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            print(f"ICE connection state is {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            print(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
            if pc.connectionState == "closed":
                self.pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            logging.debug(f"Track {track.kind} received")
            if track.kind == "audio":
                audio_track = ClientStreamTrack(
                    self.relay.subscribe(
                        track=track,
                    ),
                    "audio",
                    client,
                    self.vad,
                    self.asr,
                    self.llm,
                    self.tts,
                    pc,
                    s2s_response,
                )
                pc.addTrack(audio_track)

            @track.on("ended")
            async def on_ended():
                logging.debug(f"Track {track.kind} ended")
                track.stop()
                #await recorder.stop()

        @pc.on("datachannel")
        def on_datachannel(channel):
            logging.debug(f"DataChannel created: {channel.label}")
            @channel.on("open")
            async def on_open():
                logging.debug("DataChannel opened")
                channel.send(json.dumps({'type': 'pong'}))
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str):
                    try:
                        # 尝试解析 JSON 格式的字符串消息
                        json_msg = json.loads(message)
                        message_type = json_msg.get("type")
                        if message_type == "config":
                            is_simultaneous = json_msg["data"].get("is_simultaneous")
                            target_lang = json_msg["data"].get("target_lang")
                            print(f"Configuration received - Simultaneous: {is_simultaneous}, Target: {target_lang}")
                            client.update_config(json_msg["data"])
                            logging.debug(f"Updated config: {client.config}")
                        elif message_type == "ping":
                            logging.debug("Ping received. Sending pong...")
                            channel.send(json.dumps({'type': 'pong'}))
                        else:
                            logging.warning(f"Unknown message type: {message_type}")
                    except json.JSONDecodeError:
                        logging.error("Failed to decode JSON from string message")
                else:
                    logging.warning("Received an unsupported message type")

        @s2s_response.on("open")
        async def on_open():
            print(f"DataChannel {s2s_response.label} opened")

        @s2s_response.on("message")
        def on_message(message):
            print(f"Received message on channel: {s2s_response.label}")
            # 检查消息类型
            if isinstance(message, str):
                try:
                    # 尝试解析 JSON 格式的字符串消息
                    json_msg = json.loads(message)
                    message_type = json_msg.get("type")
                    if message_type == "config":
                        is_simultaneous = json_msg["data"].get("is_simultaneous")
                        target_lang = json_msg["data"].get("target_lang")
                        print(f"Configuration received - Simultaneous: {is_simultaneous}, Target: {target_lang}")
                        client.update_config(json_msg["data"])
                        logging.debug(f"Updated config: {client.config}")
                    elif message_type == "ping":
                        logging.debug("Ping received. Sending pong...")
                    else:
                        logging.warning(f"Unknown message type: {message_type}")
                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON from string message")
            else:
                logging.warning("Received an unsupported message type")

        # signaling = create_signaling()
        # recorder = MediaBlackhole()

        #audio_sender = pc.addTrack(MediaPlayer("assets/silence.wav", format="wav", loop=True).audio)
        #video_sender = pc.addTrack(MediaPlayer("assets/silence.mp4", format="wav", loop=True).video)

        # Set codec preferences for video
        # capabilities = RTCRtpSender.getCapabilities("video")
        # preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
        # preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
        # preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
        # transceiver = pc.getTransceivers()[1]
        # transceiver.setCodecPreferences(preferences)

        await pc.setRemoteDescription(offer)
        #await recorder.start()

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # push to cloudflare calls
        # await self.push('https://live.xyz666.org/publish/my-live')

        return JSONResponse(content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid})

    async def turn(self, data):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': 'Bearer 92d1cf73915fe293f3402775db92d40b552dd6ea84babd32d17869733cb34e2b',
                    'Content-Type': 'application/json',
                }
                turn_url = f'https://rtc.live.cloudflare.com/v1/turn/keys/{CF_TURN_KEY}/credentials/generate'
                async with session.post(turn_url, json=data, headers=headers) as response:
                    # 检查响应状态码
                    if response.status == 201:
                        # 获取响应体（SDP 数据）
                        ice_servers = await response.json()
                        # iceServers 中提取 username 和 credential
                        username = ice_servers.get('username')
                        credential = ice_servers.get('credential')
                        logging.debug(f"Username: {username}")
                        logging.debug(f"Credential: {credential}")
                        # 返回相关信息，可以根据需要自定义返回内容
                        return {
                            'username': username,
                            'credential': credential
                        }
                    else:
                        logging.debug(f"Request failed with status code {response.status}")
                        return None
        except aiohttp.ClientError as e:
            logging.debug(f'Error: {e}')
            return None

    async def post(self, url, data):
        try:
            async with aiohttp.ClientSession() as session:
                ## test url
                url = "https://live.xyz666.org/publish/my-live"
                async with session.post(url, json=data) as response:
                    # 检查响应状态码
                    if response.status == 201:
                        sdp_data = await response.text()
                        protocol_version = response.headers.get('protocol-version')
                        etag = response.headers.get('etag')
                        location = response.headers.get('location')
                        logging.debug(f"SDP Data: {sdp_data}")
                        logging.debug(f"Protocol Version: {protocol_version}")
                        logging.debug(f"ETag: {etag}")
                        logging.debug(f"Location: {location}")
                        return {
                            'sdp_data': sdp_data,
                            'protocol_version': protocol_version,
                            'etag': etag,
                            'location': location
                        }
                    else:
                        logging.debug(f"Request failed with status code {response.status}")
                        return None
        except aiohttp.ClientError as e:
            logging.debug(f'Error: {e}')
            return None

    async def live(self, whip_url, session_id):
        #create a new RTCPeerConnection, live to cloudflare webrtc calls livestream
        pc = RTCPeerConnection()
        self.pcs.add(pc)

        use_webrtc = True
        sessionid = str(uuid.uuid4())
        client = Client(use_webrtc, sessionid, self.sampling_rate, self.samples_width)

        # Create a new DataChannel after the peer connection is created
        s2s_response = pc.createDataChannel(
            label="s-events",
            ordered=True,
        )
        @s2s_response.on("open")
        async def on_open():
            logging.debug("DataChannel s2s_response opened")

        @pc.on("datachannel")
        def on_datachannel(channel):
            logging.debug(f"DataChannel created: {channel.label}")
            @channel.on("open")
            async def on_open():
                logging.debug("DataChannel opened")
                channel.send('ping')
            @channel.on("message")
            def on_message(message):
                logging.debug("Received message on channel: %s", message)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logging.debug(f"Connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
            if pc.connectionState == "closed":
                self.pcs.discard(pc)

        pc.addTrack(MediaPlayer("assets/liuyifei.wav", format="wav", loop=True).audio)
        await pc.setLocalDescription(await pc.createOffer())
        # live-whep protocol to cloudflare calls 201
        result = await self.post(whip_url, {"sdp": pc.localDescription.sdp})
        await pc.setRemoteDescription(RTCSessionDescription(sdp=result["sdp_data"], type='answer'))


    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        print("Handshake complete with Twilio 🎉")
        logging.debug(f"Client {websocket.client} accepted, waiting for messages.")
        client_id = str(uuid.uuid4())
        use_webrtc = False
        client = Client(use_webrtc, client_id, self.sampling_rate, self.samples_width)
        self.connected_clients[client_id] = client
        logging.debug(f"Client {client_id} connected")

        try:
            await self.handle_audio(client, websocket)
        finally:
            del self.connected_clients[client_id]
            logging.debug(f"Client {client_id} disconnected")
            #await websocket.close()

    """
    Twilio Voice API endpoint to handle incoming calls.
    """
    async def handle_incoming_call(self, request: Request):
        """Handle incoming call and return TwiML response to connect to Media Stream."""
        response = VoiceResponse()
        # <Say> punctuation to improve text-to-speech flow
        response = VoiceResponse()
        #response.say("Ahoy,稍等一下哦，正在为你接通你的AI girlfriends", voice='Google.cmn-TW-Wavenet-A', language='cmn-TW')
        connect = Connect()
        connect.stream(url=f'wss://{request.url.hostname}/media-stream')
        response.append(connect)
        return HTMLResponse(content=str(response), media_type="application/xml")

    async def make_call(self, request: Request):
        """Make an outgoing call to the specified phone number."""
        data = await request.json()
        to_phone_number = data.get("to")
        if not to_phone_number:
            return {"error": "Phone number is required"}
        call = twilio_client.calls.create(
            url=f"{NGROK_URL}/outgoing-call",
            to=to_phone_number,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"Call started with SID: {call.sid}")
        return {"call_sid": call.sid}

    async def handle_outgoing_call(self, request: Request):
        """Handle outgoing call and return TwiML response to connect to Media Stream."""
        response = VoiceResponse()
        response.say("稍等一下哦，正在召唤全宇宙最聪明的AI语音助理……")
        response.pause(length=1)
        response.say("好了，它上线啦！想说啥尽管说吧~")

        connect = Connect()
        connect.stream(url=f'wss://{request.url.hostname}/media-stream')
        response.append(connect)
        return HTMLResponse(content=str(response), media_type="application/xml")

    async def handle_audio(self, client, websocket):
        sessionid = None
        latest_media_timestamp = 0
        mark_queue = []
        while True:
            try:
                text = await websocket.receive_text()
                data = json.loads(text)
                if data['event'] == 'media':
                    latest_media_timestamp = int(data['media']['timestamp'])
                    chunk = data['media']['payload']
                    #TODO: g711_ulaw format
                    pcm_chunk = ulaw_to_pcm16k(base64.b64decode(chunk))
                    filtered_chunk = await self.filter.filter(pcm_chunk)
                    client.append_audio_data(filtered_chunk, 0)
                    # 异步task处理音频
                    await client.process_audio(
                        websocket, self.asr, self.vad, self.eou, self.llm, self.tts
                    )

                elif data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    print(f"Incoming stream has started {stream_sid}")
                    client.set_stream_sid(stream_sid)

                    first_messgae = "您好！請問您最近还好吗？你想要老婆不要?"
                    await client.send_initial_conversation(
                        websocket, first_messgae, self.llm, self.tts
                    )
                    latest_media_timestamp = 0
                elif data['event'] == 'mark':
                    client.pop_mark_queue()
                elif data['event'] == 'stop':
                    print(f"Call ended, stream {stream_sid} stopped")
                    client.set_stream_sid(None)
                    client.clear_buffer()
                    await websocket.close()
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown message type: {data['event']}"})

            except WebSocketDisconnect as e:
                logging.error(f"Connection with {client.client_id} closed: {e}")
                break
            except Exception as e:
                logging.error(f"Error handling audio for {client.client_id}: {e}")
                break

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

    async def upload_audio_files(self, vc_name: str, files: List[UploadFile] = File(...)):
        file_paths = []
        file_uuid = uuid.uuid4().hex[:8]
        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
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
            file_location = os.path.join("assets", f"{file_uuid}_{vc_name}{file_extension}")
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
                logging.debug(f"Filtered audio saved to: {out_filename}")
                return out_filename
            except subprocess.CalledProcessError:
                # There was an error in the ffmpeg command
                logging.debug("Error: failed to filter audio, returning original file")
                return speaker_wav
        else:
            # If no cleanup is requested, return the original file
            return speaker_wav

    async def generate_tts(self, request: TTSRequestV1):
        task_id = await self.tts_manager.gen_tts(request.tts_text, request.vc_uid, request.speed)
        return {"task_id": task_id}

    async def get_task_result(self, task_id: str):
        result = await self.tts_manager.get_task_result(task_id)
        return result

    async def health(self):
        return {"status": "ojbk"}

    async def proxy(self):
        # Open Ngrok tunnel
        listener = await ngrok.forward(f"http://localhost:{self.port}")
        print(f"Ngrok tunnel opened at {listener.url()} for port {self.port}")
        NGROK_URL = listener.url()
        INCOMING_CALL_ROUTE = "/twilio/inbound_call"
        # Set ngrok URL to ne the webhook for the appropriate Twilio number
        twilio_numbers = twilio_client.incoming_phone_numbers.list()
        twilio_number_sid = [num.sid for num in twilio_numbers if num.phone_number == TWILIO_PHONE_NUMBER][0]
        twilio_client.incoming_phone_numbers(twilio_number_sid).update(TWILIO_ACCOUNT_SID, voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")

        # 获取所有的 SIP 域名
        sip_domains = twilio_client.sip.domains.list()

        # 遍历每个 SIP 域名并更新其 voice_url
        for domain in sip_domains:
            updated_domain = twilio_client.sip.domains(domain.sid).update(voice_url=f"{NGROK_URL}{INCOMING_CALL_ROUTE}")
            print(f"SIP 域名 '{updated_domain.domain_name}' 的 voice_url 已更新为: {updated_domain.voice_url}")


    async def start_server(self):
        """Start the Uvicorn server as a coroutine."""
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

        # 捕获外部中断信号（SIGINT、SIGTERM）进行优雅关闭
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(self.shutdown(server)))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(self.shutdown(server)))

        await server.serve()

    async def shutdown(self, server):
        """Gracefully shutdown the server."""
        print("Shutting down server gracefully...")
        await server.shutdown()

        # Shutdown rtc tasks: Close WebRTC connections
        if self.pcs:
            coros = [pc.close() for pc in self.pcs]
            await asyncio.gather(*coros)
            self.pcs.clear()

        print("WebRTC connections closed.")

    async def run_tasks(self):
        """Run additional asynchronous tasks."""
        max_sessions = 1
        whip_url = self.whip_url
        tasks = []
        for k in range(max_sessions):
            url = whip_url if k == 0 else f"{whip_url}{k}"
            tasks.append(self.live(url, k))

        await asyncio.gather(*tasks)

    async def start(self):
        """Start both the server and tasks concurrently."""
        await asyncio.gather(
            self.start_server(),  # Run Uvicorn server
            self.proxy(),         # Run Ngrok proxy
            #self.run_tasks()      # Run additional logic
        )
