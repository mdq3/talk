# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Speech-to-text app running OpenAI Whisper on a Raspberry Pi 5 with Hailo AI HAT+ (Hailo 10H) hardware acceleration. Records from a USB microphone, preprocesses the audio, runs inference on the Hailo NPU, and prints transcriptions.

## Running

```bash
source venv/bin/activate
python talk.py                    # default: base model on hailo10h
python talk.py --variant tiny     # faster, less accurate
python talk.py --variant tiny.en  # english-only, hailo10h only
python talk.py --duration 20      # record up to 20 seconds
python talk.py --boost "Hailo:2.0" --boost "Raspberry:1.5"  # boost specific words
python talk.py --boost-file custom.json                      # load boost words from file
```

## Setup

Run `./setup.sh [variant]` to create the venv, install deps, and download models. The HailoRT Python wheel must be installed from `/usr/local/hailo/resources/packages/hailort-*-linux_aarch64.whl` — it is not on PyPI.

For development tools (ruff):

```bash
pip install -r requirements-dev.txt
```

## Architecture

The app follows a record → preprocess → encode → decode → clean pipeline:

1. **`talk.py`** — Thin CLI wrapper. Parses arguments and delegates to `lib/app.py`. Heavy imports are deferred so `--help` responds instantly.

2. **`lib/app.py`** — Main run loop. Prompts user to record, orchestrates the record → preprocess → encode → decode → clean pipeline, prints transcriptions.

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
   - Encoder produces features from mel spectrograms; decoder autoregressively generates tokens
   - Uses HuggingFace `AutoTokenizer` (loaded offline via `local_files_only=True`) for token decoding
   - `HEF_REGISTRY` dict maps (variant, hw_arch) → HEF filenames
   - Accepts optional `boost_words` dict to bias decoder logits toward target vocabulary during token selection
   - Communicates via `Queue`: `send_data()` to submit, `get_transcription()` to receive

7. **`lib/postprocessing.py`** — Repetition penalty during decoding (penalizes recently-generated tokens in logits), word boost logit biasing (`apply_word_boost()`), and `clean_transcription()` to deduplicate repeated sentences in output.

8. **`lib/boost_words.py`** — `load_boost_words()` merges boost config from a JSON file and CLI `--boost` args (CLI overrides file entries).

9. **`lib/spinner.py`** — Braille-dot loading spinner. `spinner(message)` returns `(done, thread)`; set `done` to stop.

10. **`boost_words.json`** — Default word boost config loaded automatically. Maps words to boost factors (e.g. `{"Hailo": 2.0}`). Empty by default.

## Testing

Tests live next to their source files with a `_test` suffix (e.g. `lib/preprocessing_test.py`).

```bash
pytest              # run all tests
pytest lib/preprocessing_test.py  # run one file
pytest -k "test_bandpass"         # run by name
```

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
