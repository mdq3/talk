# Adapted from hailo-apps speech recognition
from . import audio_utils
import numpy as np


def preprocess(audio, is_nhwc=False, chunk_length=10, chunk_offset=0, max_duration=60, overlap=0.0):
    sample_rate = audio_utils.SAMPLE_RATE
    max_samples = max_duration * sample_rate
    offset = int(chunk_offset * sample_rate)

    segment_duration = chunk_length
    segment_samples = segment_duration * sample_rate
    step = int(segment_samples * (1 - overlap))

    audio = audio[offset:max_samples]
    mel_spectrograms = []

    for start in range(0, len(audio), step):
        if start >= len(audio):
            break
        end = int(start + segment_samples)
        chunk = audio[start:end]
        chunk = audio_utils.pad_or_trim(chunk, int(segment_duration * sample_rate))
        mel = audio_utils.log_mel_spectrogram(chunk).to("cpu")
        mel = np.expand_dims(mel, axis=0)
        mel = np.expand_dims(mel, axis=2)
        if is_nhwc:
            mel = np.transpose(mel, [0, 2, 3, 1])
        mel_spectrograms.append(mel)

    return mel_spectrograms


def apply_gain(audio, gain_db):
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear


def improve_input_audio(audio, vad=True, low_audio_gain=True):
    if low_audio_gain and np.max(audio) < 0.1:
        if np.max(audio) < 0.1:
            audio = apply_gain(audio, gain_db=20)
        elif np.max(audio) < 0.2:
            audio = apply_gain(audio, gain_db=10)

    start_time = 0
    if vad:
        start_time = detect_first_speech(audio, audio_utils.SAMPLE_RATE, threshold=0.2, frame_duration=0.2)
        if start_time is None:
            pass  # No speech detected
    return audio, start_time


def detect_first_speech(audio_data, sample_rate, threshold=0.2, frame_duration=0.02):
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    frame_size = int(frame_duration * sample_rate)
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]
    energy = [np.sum(np.abs(frame)**2) / len(frame) for frame in frames]

    max_energy = max(energy)
    if max_energy > 0:
        energy = [e / max_energy for e in energy]

    for i, e in enumerate(energy):
        if e > threshold:
            start_time = i * frame_duration
            return round(start_time, 1)

    return None
