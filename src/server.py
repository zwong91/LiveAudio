import asyncio
import json
import logging
import ssl
import uuid
import base64
import uvicorn
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware


import torch
import torchaudio

from src.client import Client

from pydantic import BaseModel
from typing import List
import shutil

path = '/asset'

# 检查路径是否存在
if not os.path.exists(path):
    os.makedirs(path)
    print(f"目录 {path} 已创建")

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
            _, audio_path = await self.tts_pipeline.text_to_speech(text, vc_uid, True)
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
        static_dir="assets",  # 静态文件目录
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
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 初始化 TTSManager
        self.tts_manager = TTSManager(tts_pipeline)
        self.templates = Jinja2Templates(directory="templates")

        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)
        #self.app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

        self.app.get("/asset/{filename}")(self.get_asset_file)
        self.app.post("/generate_accent/{vc_name}")(self.upload_mp3_files)
        self.app.post("/generate_tts")(self.generate_tts)
        self.app.get("/get_task_result/{task_id}")(self.get_task_result)
        self.app.get("/health")(self.health)

        self.app.websocket("/stream")(self.websocket_endpoint)
        self.app.websocket("/stream-vc")(self.websocket_endpoint)

    async def startup(self):
        """Called on startup to set up additional services."""
        logging.info(f"Starting server at {self.host}:{self.port}")
        # 启动任务处理的后台任务
        asyncio.create_task(self.tts_manager.start_processing())

    async def shutdown():
        logging.info(f"shutdown server ...")

    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
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
        while True:
            try:
                payload = await websocket.receive_text()
                message = json.loads(payload)
                bytes = base64.b64decode(message[2])
                client.append_audio_data(bytes)
                # 异步处理音频
                self._process_audio(client, websocket)

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

    async def upload_mp3_files(self, vc_name: str, files: List[UploadFile] = File(...)):
        file_paths = []
        # 为每个文件生成一个8位 UUID 前缀
        file_uuid = uuid.uuid4().hex[:8]
        for file in files:
            # Save each file to disk
            file_location = os.path.join("vc", f"{file_uuid}_{vc_name}_{file.filename}")
            file_paths.append(file_location)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # After saving, you can process the files, send them to TTS, etc.
        return {"vc_uid": file_uuid, "file_paths": file_paths}

    async def generate_tts(self, request: TTSRequest):
        task_id = await self.tts_manager.gen_tts(request.tts_text, request.vc_uid)
        return {"task_id": task_id}

    async def get_task_result(self, task_id: str):
        result = await self.tts_manager.get_task_result(task_id)
        return result

    async def health(self):
        return {"status": "ojbk"}

    def create_uvicorn_server(self, ssl_context=None):
        """Creates and returns a Uvicorn server instance."""
        uvicorn_config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            ssl_certfile=self.certfile,
            ssl_keyfile=self.keyfile,
            loop="uvloop",
            log_level="debug",
            workers=os.cpu_count(),
            limit_concurrency=1000,
            limit_max_requests=10000,
            backlog=2048
        )
        server = uvicorn.Server(uvicorn_config)
        return server

    def start(self):
        """Start the WebSocket server."""
        ssl_context = None
        if self.certfile and self.keyfile:
            logging.info(f"Starting secure WebSocket server on {self.host}:{self.port}")
        else:
            logging.info(f"Starting WebSocket server on {self.host}:{self.port}")

        server = self.create_uvicorn_server(ssl_context)
        server.run()
