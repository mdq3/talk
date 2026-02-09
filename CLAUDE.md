# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Voice chat app running on a Raspberry Pi 5 with Hailo AI HAT+ (Hailo 10H). Records speech from a USB microphone, transcribes via Whisper on the Hailo NPU, sends transcriptions to an LLM (also on the Hailo NPU), and speaks responses via Piper TTS. Both Whisper and the LLM share a single Hailo VDevice for efficient multi-model inference.

## Running

```bash
source venv/bin/activate
python talk.py                    # voice chat: STT → LLM → TTS
python talk.py --no-tts           # text-only (no voice output)
python talk.py --variant tiny     # faster, less accurate
python talk.py --variant tiny.en  # english-only, hailo10h only
python talk.py --duration 20      # record up to 20 seconds
python talk.py --boost "Hailo:2.0" --boost "Raspberry:1.5"  # boost specific words
python talk.py --boost-file custom.json                      # load boost words from file
python talk.py --llm-model qwen2                             # different LLM model
python talk.py --tts-voice en_GB-alba-medium                 # different TTS voice
python talk.py --system-prompt "You are a pirate."           # custom system prompt
```

## Setup

Run `./setup.sh [variant] [tts-voice]` to create the venv, install deps, and download models (including Piper TTS voice). The HailoRT Python wheel must be installed from `/usr/local/hailo/resources/packages/hailort-*-linux_aarch64.whl` — it is not on PyPI.

The LLM runs directly on the Hailo NPU sharing the same VDevice as Whisper (no separate server needed). LLM models are managed via `hailo-ollama pull <model>` and stored in `/usr/share/hailo-ollama/models/`.

For development tools (ruff):

```bash
pip install -r requirements-dev.txt
```

## Architecture

The app follows a record → preprocess → encode → decode → clean → LLM → TTS pipeline:

1. **`talk.py`** — Thin CLI wrapper. Parses arguments and delegates to `lib/app.py`. Heavy imports are deferred so `--help` responds instantly.

2. **`lib/app.py`** — Main run loop. Creates a shared Hailo VDevice, loads the LLM and Whisper pipeline on it, then loops: prompt user → record → preprocess → transcribe → send to LLM (streaming) → speak response via TTS. Supports 'r' to replay the last response and 'q' to quit.

3. **`lib/record_utils.py`** — Records from the default input device at its native sample rate (44100 Hz for the USB PnP Sound Device), then anti-alias filters and resamples to 16 kHz using `resample_poly`. Saves a WAV to `/tmp/talk_recording.wav`. Supports early stop via Enter key using `select`.

4. **`lib/preprocessing.py`** — Audio cleanup pipeline applied before inference:
   - Butterworth bandpass filter (80–7500 Hz) to isolate voice
   - Spectral gating noise reduction (`noisereduce` library)
   - RMS normalization to -20 dBFS
   - Energy-based voice activity detection to find speech onset
   - `preprocess()` slices audio into fixed-length chunks and converts to log-mel spectrograms in NHWC layout for Hailo

5. **`lib/audio_utils.py`** — Whisper-compatible mel spectrogram computation using PyTorch. Loads mel filter bank from `models/mel_filters.npz`. Also provides `load_audio()` which shells out to `ffmpeg` for format conversion.

6. **`lib/pipeline.py`** — `HailoWhisperPipeline` class that manages Hailo device inference:
   - Runs encoder + decoder on a background thread via `VDevice` with round-robin scheduling
   - Accepts an optional external `vdevice` for sharing with the LLM; `create_shared_vdevice()` creates one with `group_id="SHARED"`
   - Encoder produces features from mel spectrograms; decoder autoregressively generates tokens
   - Uses HuggingFace `AutoTokenizer` (loaded offline via `local_files_only=True`) for token decoding
   - `HEF_REGISTRY` dict maps (variant, hw_arch) → HEF filenames
   - Accepts optional `boost_words` dict to bias decoder logits toward target vocabulary during token selection
   - Communicates via `Queue`: `send_data()` to submit, `get_transcription()` to receive

7. **`lib/postprocessing.py`** — Repetition penalty during decoding (penalizes recently-generated tokens in logits), word boost logit biasing (`apply_word_boost()`), and `clean_transcription()` to deduplicate repeated sentences in output.

8. **`lib/boost_words.py`** — `load_boost_words()` merges boost config from a JSON file and CLI `--boost` args (CLI overrides file entries).

9. **`lib/spinner.py`** — Braille-dot loading spinner. `spinner(message)` returns `(done, thread)`; set `done` to stop.

10. **`lib/llm.py`** — `HailoLLM` class wrapping `hailo_platform.genai.LLM`. Runs the LLM on the Hailo NPU sharing the same VDevice as Whisper. Resolves model HEF paths from the hailo-ollama model store. `stream_to_terminal()` prints tokens as they arrive.

11. **`lib/tts.py`** — Piper TTS wrapper. `PiperTTS` loads an ONNX voice model, synthesizes text to audio, and plays via a callback-based `sounddevice.OutputStream` that continuously feeds silence to keep the HDMI sink awake (prevents audio cutoff at start of playback). Resamples from Piper's native rate to the output device rate. Includes `clean_text_for_tts()` to strip markdown and noisy characters before synthesis.

12. **`boost_words.json`** — Default word boost config loaded automatically. Maps words to boost factors (e.g. `{"Hailo": 2.0}`). Empty by default.

## Testing

Tests live next to their source files with a `_test` suffix (e.g. `lib/preprocessing_test.py`). All new code should have unit tests for any testable logic (pure functions, parsers, data transforms). Hardware-dependent code (Hailo, audio devices) is excluded.

```bash
pytest              # run all tests
pytest lib/preprocessing_test.py  # run one file
pytest -k "test_bandpass"         # run by name
```

## Documentation

Keep `ARCHITECTURE.md` up to date when changing the application structure, data flow, hardware requirements, or multi-model inference setup.

## Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, configured in `pyproject.toml` (Python 3.13, 100-char line length, `E`/`F`/`W`/`I` rules).

```bash
ruff check .       # lint
ruff format .      # format
```

Run both before committing. All code must pass `ruff check` and `ruff format --check`.

## Hardware Constraints

- USB mic only supports 44100 Hz native — never try to open it at 16 kHz directly (`sounddevice` will raise `paInvalidSampleRate`)
- The `hailo_platform` module comes from the system-installed HailoRT wheel, not PyPI
- Model HEF files live in `models/` (gitignored) and vary by variant and hardware arch
- `tiny.en` variant is only available for hailo10h
