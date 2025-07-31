#!/usr/bin/env python3

# Vox: Voice Transcription Tool using Deepgram API
# Usage: ./vox.py [debug]
# Dependencies: pip install sounddevice numpy requests pyperclip scipy
# Setup: Set DEEPGRAM_API_KEY environment variable and run 'chmod +x vox.py'

import json
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyperclip
import requests
import scipy.io.wavfile as wavfile
import sounddevice as sd

# Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
MAX_DURATION = int(os.getenv("VOX_MAX_DURATION", 120))  # Default 120 seconds
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1  # Mono
AUDIO_FORMAT = "wav"
DEEPGRAM_MODEL = os.getenv("VOX_DEEPGRAM_MODEL", "nova-3")  # Default nova-3
DEBUG = False
PID_FILE = Path.home() / ".recordpid"
AUDIO_FILE_TRACKER = Path.home() / ".recordaudio"
BASE_FILE = Path.home() / ".vox" / "recording"

# Global state
recording_process = None
audio_data = []
stop_event = threading.Event()


def check_environment():
    """Check environment variables."""
    if not DEEPGRAM_API_KEY:
        print(
            "Error: DEEPGRAM_API_KEY environment variable is not set.", file=sys.stderr
        )
        print(
            "Please set it with: export DEEPGRAM_API_KEY='your-api-key'",
            file=sys.stderr,
        )
        print("Get your API key from https://console.deepgram.com", file=sys.stderr)
        sys.exit(1)


def callback(indata, frames, time, status):
    """Audio stream callback."""
    if status:
        print(f"Error: {status}", file=sys.stderr)
    if not stop_event.is_set():
        audio_data.append(indata.copy())


def cleanup():
    """Clean up resources."""
    global recording_process
    if recording_process:
        recording_process.stop()
        recording_process.close()
        recording_process = None
    PID_FILE.unlink(missing_ok=True)
    AUDIO_FILE_TRACKER.unlink(missing_ok=True)


def signal_handler(sig, frame):
    """Handle signals."""
    print("\n> Ctrl+C detected, stopping recording...")
    stop_event.set()
    cleanup()
    sys.exit(0)


def display_counter():
    """Display recording counter."""
    start_time = time.time()
    while not stop_event.is_set():
        elapsed = int(time.time() - start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        sys.stdout.write(f"\r> Recording time: {minutes:02d}:{seconds:02d}")
        sys.stdout.flush()
        time.sleep(0.1)  # Update every 0.1 seconds
    sys.stdout.write("\r" + " " * 30 + "\r")  # Clear the line
    sys.stdout.flush()


def start_recording():
    """Start audio recording."""
    global recording_process
    cleanup()  # Ensure clean state before starting
    BASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    audio_file = f"{BASE_FILE}_{time.strftime('%Y%m%d_%H%M%S')}.{AUDIO_FORMAT}"
    with open(AUDIO_FILE_TRACKER, "w") as f:
        f.write(audio_file)
    print(f"> Recording started to {audio_file}...")
    print(f"Press Enter to stop early (max {MAX_DURATION} seconds).")
    audio_data.clear()
    stop_event.clear()
    recording_process = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback
    )
    recording_process.start()
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    # Start threads for counter and input monitoring
    counter_thread = threading.Thread(target=display_counter, daemon=True)
    counter_thread.start()

    def monitor_input():
        rlist, _, _ = select.select([sys.stdin], [], [], MAX_DURATION)
        if rlist:  # Enter key pressed
            stop_event.set()

    threading.Thread(target=monitor_input, daemon=True).start()
    stop_event.wait(timeout=MAX_DURATION)
    if not stop_event.is_set():
        stop_event.set()  # Timeout reached
    return stop_recording()


def stop_recording():
    """Stop audio recording."""
    global recording_process
    print("> Stopping recording...")
    if PID_FILE.exists() and recording_process:
        stop_event.set()
        audio_file = AUDIO_FILE_TRACKER.read_text().strip()
        if audio_data:  # Only save if there's data
            wavfile.write(audio_file, SAMPLE_RATE, np.concatenate(audio_data))
            cleanup()
            print("Recording stopped.")
            return audio_file
        else:
            print("No audio data recorded.")
            cleanup()
            sys.exit(1)
    else:
        print("No recording to stop.")
        cleanup()
        sys.exit(1)


def transcribe(audio_file):
    """Transcribe audio file using Deepgram API."""
    if not audio_file or not Path(audio_file).exists():
        print("Error: Audio file missing or not set", file=sys.stderr)
        sys.exit(1)
    print(f"> Transcribing with Deepgram API (model: {DEEPGRAM_MODEL})...")
    error_file = f"{audio_file}.error"
    json_file = f"{audio_file}.json"
    txt_file = f"{audio_file}.txt"
    try:
        with open(audio_file, "rb") as f:
            response = requests.post(
                f"https://api.deepgram.com/v1/listen?model={DEEPGRAM_MODEL}&smart_format=true",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": f"audio/{AUDIO_FORMAT}",
                },
                data=f,
            )
        response.raise_for_status()
        with open(json_file, "w") as f:
            json.dump(response.json(), f)
        if DEBUG:
            print(
                f"DEBUG: Full Deepgram API response saved in {json_file}:",
                file=sys.stderr,
            )
            print(json.dumps(response.json(), indent=2), file=sys.stderr)
        transcript = response.json()["results"]["channels"][0]["alternatives"][0][
            "transcript"
        ]
        with open(txt_file, "w") as f:
            f.write(transcript.rstrip())
        pyperclip.copy(transcript)
        print(f"> Transcript copied to clipboard from {txt_file}")
        Path(error_file).unlink(missing_ok=True)
    except requests.RequestException as e:
        with open(error_file, "w") as f:
            f.write(str(e))
        print(
            f"Error: Transcription failed. Deepgram response saved in {error_file}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (KeyError, IndexError):
        print(
            f"Error: Failed to extract transcript from JSON, check {json_file}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    check_environment()  # Check environment variables at startup
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            globals()["DEBUG"] = True
    audio_file = start_recording()
    if audio_file:  # Only transcribe if recording was successful
        transcribe(audio_file)


if __name__ == "__main__":
    main()
