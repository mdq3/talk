# Adapted from hailo-apps speech recognition
#
# Records audio from the system's default input device (USB mic).
#
# The USB PnP Sound Device only supports 44100 Hz natively, but Whisper
# expects 16000 Hz. We record at the device's native rate, apply a
# low-pass anti-aliasing filter, then resample down to 16 kHz using a
# polyphase filter (resample_poly), which is cleaner than the naive
# scipy.signal.resample for non-integer rate ratios.

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly, butter, sosfilt
from math import gcd
import select
import sys
import queue
import time

TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1


def _get_device_sample_rate():
    """Get the native sample rate of the default input device."""
    dev = sd.query_devices(kind='input')
    return int(dev['default_samplerate'])


def _anti_alias_and_resample(audio, device_rate, target_rate):
    """Downsample audio with proper anti-aliasing.

    Before reducing the sample rate, we must remove frequencies above
    the new Nyquist frequency (target_rate / 2) to prevent aliasing
    artifacts. Then we use a polyphase resampling filter which handles
    the non-integer ratio (44100 -> 16000) much better than plain
    scipy.signal.resample.
    """
    # Low-pass at slightly below the new Nyquist to give the filter room
    nyquist = target_rate / 2.0
    cutoff = nyquist * 0.95  # 7600 Hz — just under the 8 kHz Nyquist
    sos = butter(8, cutoff, btype='low', fs=device_rate, output='sos')
    audio = sosfilt(sos, audio).astype(np.float32)

    # resample_poly needs integer up/down factors
    # For 44100 -> 16000: gcd(44100, 16000) = 100, so up=160, down=441
    divisor = gcd(device_rate, target_rate)
    up = target_rate // divisor
    down = device_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32)


def enter_pressed():
    return select.select([sys.stdin], [], [], 0.0)[0]


def record_audio(duration, audio_path):
    """Record audio from the default mic and save as 16 kHz WAV.

    Records at the device's native sample rate (typically 44100 Hz for
    the USB PnP Sound Device), then resamples to 16 kHz with proper
    anti-aliasing for Whisper input.
    """
    q = queue.Queue()
    recorded_frames = []
    device_rate = _get_device_sample_rate()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    print(f"Recording for up to {duration} seconds. Press Enter to stop early...")

    start_time = time.time()
    with sd.InputStream(samplerate=device_rate,
                        channels=CHANNELS,
                        dtype="float32",
                        callback=audio_callback):
        sys.stdin = open('/dev/stdin')
        while True:
            if time.time() - start_time >= duration:
                print("Max duration reached.")
                break
            if enter_pressed():
                sys.stdin.read(1)
                print("Early stop requested.")
                break
            try:
                frame = q.get(timeout=0.1)
                recorded_frames.append(frame)
            except queue.Empty:
                continue

    print("Recording finished. Processing...")

    audio_data = np.concatenate(recorded_frames, axis=0)
    # Collapse to mono if multi-channel
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16 kHz with anti-aliasing if needed
    if device_rate != TARGET_SAMPLE_RATE:
        audio_data = _anti_alias_and_resample(audio_data, device_rate, TARGET_SAMPLE_RATE)

    wav.write(audio_path, TARGET_SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    return audio_data
