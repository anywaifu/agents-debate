"""Utilities for Text-to-Speech (TTS) functionality."""

import asyncio
import os
import subprocess
import tempfile
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.workflow import Context
from openai import AsyncOpenAI

load_dotenv()

tts_client: Optional[AsyncOpenAI] = None
_openai_api_key = os.getenv("OPENAI_API_KEY")

if _openai_api_key:
    tts_client = AsyncOpenAI()
else:
    print("Info: OPENAI_API_KEY not set. TTS functionality will be disabled.")

async def get_tts_params_from_state(ctx: Context, agent_name: str) -> tuple[str, str]:
    """Retrieves TTS model and agent-specific voice from workflow state."""
    current_workflow_state = await ctx.get("state") # type: ignore
    tts_config = current_workflow_state.get("tts_config", {})
    tts_model = tts_config.get("model", "gpt-4o-mini-tts")  # Fallback model
    agent_voices = tts_config.get("voices", {})
    tts_voice = agent_voices.get(agent_name, "alloy")  # Fallback voice
    return tts_model, tts_voice

async def speak_text(
    text_to_speak: str,
    model: str,
    voice: str,
    response_format: str = "mp3"
):
    """Helper function to speak text using OpenAI TTS."""
    if not tts_client or not text_to_speak.strip():
        return

    temp_file_path_obj: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as tmp_file:
            temp_file_path_obj = Path(tmp_file.name)

        tts_params = {
            "model": model,
            "voice": voice,
            "input": text_to_speak,
            "response_format": response_format,
            "speed": "1.2"
        }

        async with tts_client.audio.speech.with_streaming_response.create(**tts_params) as response: # type: ignore
            await response.stream_to_file(temp_file_path_obj) # type: ignore

        if temp_file_path_obj and temp_file_path_obj.exists() and temp_file_path_obj.stat().st_size > 0:
            playback_command = ["ffplay", "-autoexit", "-nodisp", str(temp_file_path_obj)]
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    playback_command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                ) # noqa
            except FileNotFoundError:
                print("Error: ffplay command not found. Ensure ffplay is installed and in PATH for audio playback.")
            except subprocess.CalledProcessError as e_play:
                print(f"Error during ffplay playback for {temp_file_path_obj}: {e_play}")
                if e_play.stderr:
                    print(f"ffplay stderr: {e_play.stderr.decode(errors='ignore')}")
        elif temp_file_path_obj and temp_file_path_obj.exists():
            print(f"Skipping playback of empty/invalid TTS file: {temp_file_path_obj}")

    except Exception as e:
        print(f"Error during TTS processing or file operations: {e}")
    finally:
        if temp_file_path_obj and temp_file_path_obj.exists():
            try:
                os.remove(temp_file_path_obj)
            except OSError as e_os:
                print(f"Error deleting temporary file {temp_file_path_obj}: {e_os}")