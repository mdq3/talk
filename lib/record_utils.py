# Adapted from hailo-apps speech recognition
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
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


def enter_pressed():
    return select.select([sys.stdin], [], [], 0.0)[0]


def record_audio(duration, audio_path):
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
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if the device uses a different rate
    if device_rate != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio_data) * TARGET_SAMPLE_RATE / device_rate)
        audio_data = scipy.signal.resample(audio_data, num_samples)

    wav.write(audio_path, TARGET_SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    return audio_data
