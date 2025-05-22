import logging
import json
import argparse

from src.filter.filter_factory import FilterFactory
from src.asr.asr_factory import ASRFactory
from src.vad.vad_factory import VADFactory
from src.turn.turn_factory import TurnFactory
from src.llm.llm_factory import LLMFactory
from src.tts.tts_factory import TTSFactory

from src.server import Server

import asyncio

def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio AI Server: Real-time audio conversation "
                    "using self-hosted STT,LLM,TTS pipeline and WebSocket/WebRTC."
    )
    parser.add_argument("--vad-type", type=str, default="silero", help="VAD pipeline type")
    parser.add_argument("--vad-args", type=str, default='{"auth_token": "huggingface_token"}', help="VAD args (JSON string)")
    parser.add_argument("--turn-type", type=str, default="livekit", help="turn taking type")
    parser.add_argument("--filter-type", type=str, default='noisereduce', help="Filter noise type")
    parser.add_argument("--asr-type", type=str, default="whisper", help="ASR pipeline type")
    parser.add_argument("--asr-args", type=str, default='{"model_size": "large-v3-turbo"}', help="ASR args (JSON string)")
    parser.add_argument("--llm-type", type=str, default="ollama", help="OPENAI pipeline type")
    parser.add_argument("--tts-type", type=str, default="elevenlabs", help="TTS pipeline type")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the WebSocket server")
    parser.add_argument("--port", type=int, default=8765, help="Port for the WebSocket server")
    parser.add_argument("--certfile", type=str, default=None, help="Path to SSL certificate file")
    parser.add_argument("--keyfile", type=str, default=None, help="Path to SSL key file")
    parser.add_argument('--transport', type=str, default='webrtc')
    parser.add_argument('--whip_url', type=str, default='http://108.137.9.108:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error"], help="Logging level")
    return parser.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(level=args.log_level.upper())
    logging.debug(f"Arguments: {args}")

    try:
        vad_args = json.loads(args.vad_args)
        asr_args = json.loads(args.asr_args)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON arguments: {e}")
        return

    # Create VAD and ASR and LLM and TTS pipelines
    filter = FilterFactory.create_filter_pipeline(args.filter_type)
    if filter:
        logging.info(f"Using filter pipeline: {args.filter_type}")
    else:
        logging.warning(f"No filter pipeline found for type: {args.filter_type}")
    asr = ASRFactory.create_asr_pipeline(args.asr_type, **asr_args)
    vad = VADFactory.create_vad_pipeline(args.vad_type, **vad_args)
    eou = TurnFactory.create_turn_pipeline(args.turn_type)
    llm = LLMFactory.create_llm_pipeline(args.llm_type)
    tts = TTSFactory.create_tts_pipeline(args.tts_type)

    # Create and start server
    server = Server(filter, asr, vad, eou, llm, tts, host=args.host, port=args.port, certfile=args.certfile, keyfile=args.keyfile)
    asyncio.run(server.start())

if __name__ == "__main__":
    main()
