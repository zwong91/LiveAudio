from dotenv import load_dotenv

from livekit import agents
from livekit.agents import stt, AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=openai.STT(detect_language=True, model="deepdml/faster-whisper-large-v3-turbo-ct2", base_url="http://localhost:8000/v1"),
        llm=openai.LLM.with_ollama(
            model="gemma3:12b",
            base_url="http://localhost:11434/v1",
        ),
        tts=openai.TTS(model="kokoro", voice="af_alloy", base_url="http://localhost:8880/v1"),
        # stt=openai.STT(),
        # llm=openai.LLM(),
        # tts=openai.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="请问您现在有空吗？要老婆不要哦."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="my-telephony-agent"
    ))
