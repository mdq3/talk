"""Piper TTS wrapper — synthesize speech and play via sounddevice."""

import io
import os
import re
import wave
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly


def _suppress_native_stderr(func):
    """Call func() with native stderr (fd 2) redirected to /dev/null.

    onnxruntime writes GPU discovery warnings at the C level during import,
    which Python's redirect_stderr cannot catch. This redirects the actual
    file descriptor.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    try:
        os.dup2(devnull, 2)
        return func()
    finally:
        os.dup2(old_fd, 2)
        os.close(devnull)
        os.close(old_fd)


DEFAULT_VOICE = "en_US-amy-medium"


def clean_text_for_tts(text):
    """Clean text for TTS synthesis.

    Strips markdown formatting and noisy characters that cause artifacts.
    Adapted from hailo-apps voice assistant.
    """
    # Remove markdown bold/italic
    text = re.sub(r"[*_]{1,3}", "", text)
    # Remove code backticks
    text = re.sub(r"`+", "", text)
    # Remove headers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Convert links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove noisy symbols
    text = re.sub(r"[~@^|\\<>{}\[\]#]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PiperTTS:
    """Text-to-speech using Piper ONNX models."""

    def __init__(self, model_dir, voice_name=DEFAULT_VOICE):
        onnx_path = f"{model_dir}/{voice_name}.onnx"

        def _load():
            from piper import PiperVoice

            return PiperVoice.load(onnx_path)

        self.voice = _suppress_native_stderr(_load)
        self.native_rate = self.voice.config.sample_rate

    def synthesize(self, text):
        """Synthesize text to a float32 numpy array at the model's native sample rate."""
        text = clean_text_for_tts(text)
        if not text.strip():
            return np.array([], dtype=np.float32)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            with redirect_stderr(StringIO()):
                self.voice.synthesize_wav(text, wf)

        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wf:
            data = wf.readframes(wf.getnframes())
            width = wf.getsampwidth()

        if width == 2:
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(data, dtype=np.float32)

        return audio

    def speak(self, text):
        """Synthesize text and play it through the default output device (blocking)."""
        audio = self.synthesize(text)
        if len(audio) == 0:
            return

        # Resample to the default output device's sample rate if needed
        dev = sd.query_devices(kind="output")
        playback_rate = int(dev["default_samplerate"])

        if playback_rate != self.native_rate:
            from math import gcd

            divisor = gcd(self.native_rate, playback_rate)
            up = playback_rate // divisor
            down = self.native_rate // divisor
            audio = resample_poly(audio, up, down).astype(np.float32)

        sd.play(audio, samplerate=playback_rate)
        sd.wait()
