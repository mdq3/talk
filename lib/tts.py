"""Piper TTS wrapper — synthesize speech and play via sounddevice."""

import io
import os
import queue
import re
import threading
import wave
from contextlib import redirect_stderr
from io import StringIO
from math import gcd

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
    """Text-to-speech using Piper ONNX models.

    Keeps a persistent audio stream that continuously feeds silence to the
    output device, preventing HDMI sink sleep. When speak() is called, the
    audio data is queued and played without any wake-up delay.
    """

    def __init__(self, model_dir, voice_name=DEFAULT_VOICE):
        onnx_path = f"{model_dir}/{voice_name}.onnx"

        def _load():
            from piper import PiperVoice

            return PiperVoice.load(onnx_path)

        self.voice = _suppress_native_stderr(_load)
        self.native_rate = self.voice.config.sample_rate

        dev = sd.query_devices(kind="output")
        self.playback_rate = int(dev["default_samplerate"])

        # Compute resampling factors once
        if self.playback_rate != self.native_rate:
            divisor = gcd(self.native_rate, self.playback_rate)
            self._resample_up = self.playback_rate // divisor
            self._resample_down = self.native_rate // divisor
        else:
            self._resample_up = None
            self._resample_down = None

        # Audio playback via a callback stream that continuously runs.
        # Feeds silence when idle so the HDMI sink never sleeps.
        self._audio_queue = queue.Queue()
        self._audio_buf = np.empty(0, dtype=np.float32)
        self._done_event = threading.Event()
        self._stream = sd.OutputStream(
            samplerate=self.playback_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, outdata, frames, time_info, status):
        """Fill the output buffer from queued audio, or silence if idle."""
        needed = frames
        written = 0

        while written < needed:
            # Drain current buffer
            if len(self._audio_buf) > 0:
                chunk = min(len(self._audio_buf), needed - written)
                outdata[written : written + chunk, 0] = self._audio_buf[:chunk]
                self._audio_buf = self._audio_buf[chunk:]
                written += chunk
            else:
                # Try to get more audio from the queue
                try:
                    self._audio_buf = self._audio_queue.get_nowait()
                except queue.Empty:
                    # No audio left — fill remainder with silence
                    outdata[written:, 0] = 0.0
                    self._done_event.set()
                    return

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
        """Synthesize text and play via the persistent stream (blocking)."""
        audio = self.synthesize(text)
        if len(audio) == 0:
            return

        if self._resample_up is not None:
            audio = resample_poly(audio, self._resample_up, self._resample_down).astype(np.float32)

        self._done_event.clear()
        self._audio_queue.put(audio)
        self._done_event.wait()

    def close(self):
        """Stop and close the audio stream."""
        self._stream.stop()
        self._stream.close()
