# Talk - Voice Chat on Hailo AI HAT+

![Tests](https://github.com/mdq3/talk/actions/workflows/tests.yml/badge.svg)
![Coverage](https://raw.githubusercontent.com/mdq3/talk/main/coverage.svg)

Voice chat application running on a Raspberry Pi 5 with Hailo AI HAT+. Speak into a USB microphone, get transcriptions via Whisper, responses from an LLM, and hear them spoken back via Piper TTS — all accelerated on the Hailo NPU.

## Prerequisites

- Raspberry Pi 5 with AI HAT+ (Hailo 8, 8L, or 10H)
- USB microphone
- Audio output (HDMI or other)
- HailoRT 5.1.1 installed (the `hailort` system package)
- `hailo-ollama` installed (`sudo apt install hailo-ollama`) — for LLM model management
- Python 3.13+
- `ffmpeg` installed (`sudo apt install ffmpeg`)
- `libportaudio2` installed (`sudo apt install libportaudio2`)

Verify your Hailo device is detected:

```bash
hailortcli fw-control identify
```

## Quick Setup

The setup script handles everything — creates the virtual environment, installs dependencies, and downloads model files:

```bash
./setup.sh          # defaults to whisper base model + en_US-amy-medium TTS voice
./setup.sh tiny     # use tiny model instead
./setup.sh tiny.en  # english-only tiny model
./setup.sh base en_GB-alba-medium  # different TTS voice
```

You also need an LLM model downloaded for chat. Pull one via `hailo-ollama`:

```bash
hailo-ollama pull qwen2
```

## Manual Setup

If you prefer to set things up step by step:

### 1. Create virtual environment

```bash
cd ~/Projects/talk
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

### 2. Install the HailoRT Python bindings

The wheel is included with the `hailort` system package:

```bash
pip install /usr/local/hailo/resources/packages/hailort-*-linux_aarch64.whl
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional: ruff linter/formatter
```

### 4. Download Whisper model files

Download the HEF model files and decoder assets for your hardware. Replace `VARIANT` with the model variant (`base`, `tiny`, or `tiny.en`):

```bash
VARIANT=base
BASE_HEF=https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/whisper
BASE_ASSETS="https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/npy%20files/whisper/decoder_assets"
```

Download encoder and decoder HEFs:

```bash
mkdir -p models/hefs/h10h/$VARIANT
wget -P models/hefs/h10h/$VARIANT/ "$BASE_HEF/h10h/${VARIANT}-whisper-encoder-10s.hef"
wget -P models/hefs/h10h/$VARIANT/ "$BASE_HEF/h10h/${VARIANT}-whisper-decoder-10s-out-seq-64.hef"
```

> Note: The exact HEF filenames vary by variant. Check `lib/pipeline.py` HEF_REGISTRY for the correct filenames.

Download decoder tokenization assets:

```bash
mkdir -p models/decoder_assets/$VARIANT/decoder_tokenization
wget -P models/decoder_assets/$VARIANT/decoder_tokenization/ \
  "$BASE_ASSETS/$VARIANT/decoder_tokenization/token_embedding_weight_$VARIANT.npy"
wget -P models/decoder_assets/$VARIANT/decoder_tokenization/ \
  "$BASE_ASSETS/$VARIANT/decoder_tokenization/onnx_add_input_$VARIANT.npy"
```

Download mel filter bank:

```bash
mkdir -p models
wget -O models/mel_filters.npz \
  "https://github.com/openai/whisper/raw/main/whisper/assets/mel_filters.npz"
```

### Verify setup

Your `models/` directory should look like this (example for base):

```
models/
  mel_filters.npz
  hefs/h10h/base/
    base-whisper-encoder-10s.hef
    base-whisper-decoder-10s-out-seq-64.hef
  decoder_assets/base/decoder_tokenization/
    token_embedding_weight_base.npy
    onnx_add_input_base.npy
  piper/
    en_US-amy-medium.onnx
    en_US-amy-medium.onnx.json
```

## Usage

```bash
source venv/bin/activate
python talk.py
```

1. Press **Enter** to start recording (up to 10 seconds by default)
2. Speak into the microphone
3. Press **Enter** to stop recording early, or wait for the timeout
4. The transcription appears as `>>> your text here`
5. The LLM response streams to the terminal, then is spoken via TTS
6. Type **w** then Enter to type a message instead of speaking
7. Press **r** then Enter to replay the last spoken response
8. Type **q** then Enter to quit

### Options

```
--variant {base,tiny,tiny.en}          Whisper model variant (default: base)
--hw-arch {hailo8,hailo8l,hailo10h}    Hailo architecture (default: hailo10h)
--duration SECONDS                     Max recording duration (default: 10)
--llm-model MODEL                      LLM model name (default: qwen2)
--tts-voice VOICE                      Piper TTS voice (default: en_US-amy-medium)
--no-tts                               Disable TTS voice output (text-only)
--system-prompt PROMPT                 Override the default LLM system prompt
--boost "word:factor"                  Boost a word during decoding (repeatable)
--boost-file PATH                      JSON file with word boost factors
```

Examples:

```bash
python talk.py --variant tiny          # Faster, less accurate
python talk.py --variant tiny.en       # English-only (hailo10h only)
python talk.py --duration 20           # Record up to 20 seconds
python talk.py --no-tts                # Text-only (no voice output)
python talk.py --llm-model qwen2       # Use a different LLM model
python talk.py --system-prompt "You are a pirate."  # Custom personality
python talk.py --boost "Hailo:2.0" --boost "Raspberry:1.5"  # Boost specific words
```

## Model Variants

| Variant  | Parameters | Notes                        |
|----------|-----------|-------------------------------|
| base     | 74M       | Best accuracy, default        |
| tiny     | 39M       | Faster inference              |
| tiny.en  | 39M       | English-only, hailo10h only   |

## Troubleshooting

**`paInvalidSampleRate` error**: Your USB microphone may not support 16kHz directly. The app records at the mic's native sample rate (typically 44100Hz) and resamples automatically. If you still see this error, check `python3 -c "import sounddevice; print(sounddevice.query_devices())"` to verify your mic is detected.

**`No module named 'hailo_platform'`**: Make sure you installed the HailoRT wheel into the venv (step 2 above).

**`HEF file not found`**: Make sure you downloaded the model files for the correct architecture and variant (step 4 above). Check that the filenames match what's listed in `lib/pipeline.py` HEF_REGISTRY.

**No speech detected**: Speak louder or closer to the microphone. The app applies voice activity detection and will skip recordings with no audible speech.

**`LLM model not found`**: Make sure you've pulled the model via `hailo-ollama pull qwen2` (or whichever model you're using with `--llm-model`).

**TTS audio cuts off at start**: The app uses a persistent audio stream to keep the HDMI sink awake. If you still experience cutoff, check your audio output device with `python3 -c "import sounddevice; print(sounddevice.query_devices())"`.

**`HAILO_OUT_OF_PHYSICAL_DEVICES`**: Another process is using the Hailo device. Stop any other Hailo applications (including `hailo-ollama serve`) before running talk.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for details on multi-model NPU inference, the audio pipeline, and hardware requirements.
