# Architecture

## Overview

Talk is a voice chat application that runs entirely on a Raspberry Pi 5 with a Hailo AI HAT+. It chains three AI models together — speech-to-text (Whisper), a large language model, and text-to-speech (Piper) — to create a conversational voice assistant with no cloud dependency.

```
Microphone → Record → Preprocess → Whisper STT → LLM → Piper TTS → Speaker
                                        ↑              ↑
                                        └──── Hailo NPU ────┘
```

## Hardware

### Required

- **Raspberry Pi 5** — host CPU, runs audio I/O and orchestration
- **Hailo AI HAT+** with **Hailo 10H** (or Hailo 8/8L) — NPU for Whisper and LLM inference
- **USB microphone** — audio input (the USB PnP Sound Device is the tested device; only supports 44100 Hz native sample rate)
- **Audio output** — HDMI or other output device for TTS playback

### Software Dependencies

- **HailoRT 5.1.1** — system package providing `hailo_platform` Python bindings (installed from a local `.whl`, not PyPI)
- **hailo-ollama** — manages LLM model downloads and stores HEF blobs at `/usr/share/hailo-ollama/models/`
- **ffmpeg** — audio format conversion
- **libportaudio2** — audio I/O via `sounddevice`

## Multi-Model NPU Inference

The core architectural challenge is running two distinct neural networks (Whisper and an LLM) on a single Hailo NPU. This is solved with a **shared VDevice** and **round-robin scheduling**.

### Shared VDevice

A Hailo `VDevice` is a virtual abstraction over the physical NPU. By creating one with `group_id="SHARED"` and `ROUND_ROBIN` scheduling, multiple models can time-share the hardware:

```python
params = VDevice.create_params()
params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
params.group_id = "SHARED"
vdevice = VDevice(params)
```

This single VDevice is created once at startup and passed to both the Whisper pipeline and the LLM. The round-robin scheduler interleaves inference requests from each model, so they can coexist without conflicts.

### Whisper (Speech-to-Text)

Whisper runs as two separate HEF models on the NPU:

1. **Encoder** — takes a log-mel spectrogram and produces encoded audio features. Runs once per utterance.
2. **Decoder** — autoregressively generates text tokens from the encoded features. Runs in a loop (one NPU call per token) until an end-of-sequence token or the sequence length limit.

Both are loaded onto the shared VDevice via `vdevice.create_infer_model()`. The encoder and decoder run on a background thread, communicating with the main thread via queues.

### LLM (Language Model)

The LLM uses `hailo_platform.genai.LLM`, a higher-level API that handles tokenization and autoregressive decoding internally. It is initialized with the same shared VDevice:

```python
from hailo_platform.genai import LLM
llm = LLM(vdevice, hef_path)
```

LLM models are stored as HEF blobs managed by `hailo-ollama`. The `resolve_hef_path()` function reads the hailo-ollama manifest to locate the correct blob on disk.

### Inference Flow

Whisper and the LLM never run simultaneously — they take turns on the NPU:

1. User speaks → audio recorded and preprocessed
2. **Whisper encoder** runs on NPU → produces features
3. **Whisper decoder** runs on NPU (iteratively) → produces text tokens
4. **LLM** runs on NPU (iteratively) → produces response tokens, streamed to terminal
5. Piper TTS synthesizes response on CPU → plays through speaker
6. Loop back to step 1

The round-robin scheduler ensures clean handoff between models. There is no model loading/unloading between turns — both remain loaded on the VDevice throughout the session.

## Audio Pipeline

### Recording

The USB microphone only supports 44100 Hz. Audio is recorded at the native rate, then anti-alias filtered (8th-order Butterworth at ~7600 Hz) and resampled to 16 kHz using `resample_poly` for Whisper.

### Preprocessing

Before Whisper inference, audio goes through:

1. **Bandpass filter** (80–7500 Hz) — isolates human voice frequencies
2. **Spectral gating** — suppresses steady-state background noise
3. **RMS normalization** — brings speech to a consistent level (-20 dBFS)
4. **Voice activity detection** — finds where speech begins, skips leading silence
5. **Mel spectrogram conversion** — fixed-length chunks in NHWC layout for Hailo

### TTS Playback

Piper TTS runs on the CPU using an ONNX voice model. A persistent `sounddevice.OutputStream` feeds silence continuously to keep the HDMI audio sink awake, preventing the first-playback cutoff common with HDMI audio. When speech is ready, it's queued into the stream with a 300ms silence pad to absorb any remaining pipeline latency.

## Startup Sequence

Heavy loading stages each get an animated spinner (run in a forked child process so the animation stays smooth even when C extensions hold the GIL):

1. **TTS voice** — load Piper ONNX model, start audio stream
2. **Dependencies** — import torch, transformers, hailo_platform, etc.
3. **Hailo device** — create the shared VDevice
4. **LLM** — resolve HEF path and load model onto VDevice
5. **Whisper** — load encoder + decoder HEFs, tokenizer, and start inference thread

## Model Variants

Whisper models are available in multiple sizes. HEF files are architecture-specific:

| Variant  | Hailo 10H | Hailo 8 | Hailo 8L |
|----------|-----------|---------|----------|
| base     | yes       | yes     | yes      |
| tiny     | yes       | yes     | yes      |
| tiny.en  | yes       | no      | no       |

LLM models are pulled via `hailo-ollama pull <model>` and stored in `/usr/share/hailo-ollama/models/`. The default is `qwen2`.
