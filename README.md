# vox

Record and transcribe audio using Deepgram API.

## Features

- Record audio (mono, 16kHz WAV)
- Transcribe recordings with Deepgram API
- Copy transcripts to clipboard

## Requirements

- Python 3.13+
- Deepgram API Key (get one from [Deepgram Console](https://deepgram.com/)).

## Installation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Terminal version

Export your `DEEPGRAM_API_KEY` as env variable:

```
export DEEPGRAM_API_KEY=xxx
```

Run the script:
```
uv run vox.py
```

### GUI version

Launch the GUI:

```
uv run vox_gui.py
