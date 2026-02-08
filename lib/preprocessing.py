# Adapted from hailo-apps speech recognition
#
# Audio preprocessing pipeline for Whisper on Hailo.
# Cleans up raw microphone input to improve transcription accuracy:
#   1. Bandpass filter  — isolates human voice frequencies, cuts mic hiss and hum
#   2. Noise reduction  — spectral gating to suppress steady-state background noise
#   3. Normalization    — RMS-based gain to bring speech to a consistent level
#   4. Voice activity   — energy-based detection of where speech starts

import noisereduce as nr
import numpy as np
from scipy.signal import butter, sosfilt

from . import audio_utils

# --- Frequency filter ---

# Human speech fundamentals sit around 80-300 Hz, but harmonics and consonants
# (sibilants like "s", "f", "th") extend up to ~7.5 kHz. Filtering to this
# range removes low-frequency hum (mains, vibration) and high-frequency hiss
# (cheap USB mic self-noise) that would otherwise confuse the model.
VOICE_LOW_HZ = 80
VOICE_HIGH_HZ = 7500


def bandpass_filter(audio, sample_rate, low_hz=VOICE_LOW_HZ, high_hz=VOICE_HIGH_HZ, order=5):
    """Apply a Butterworth bandpass filter to isolate human voice frequencies.

    Uses a second-order sections (sos) representation for numerical stability,
    which matters on the 32-bit float audio we're working with.
    """
    nyquist = sample_rate / 2.0
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)  # clamp below Nyquist to avoid filter instability
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# --- Noise reduction ---


def reduce_noise(audio, sample_rate):
    """Remove steady-state background noise using spectral gating.

    noisereduce estimates the noise profile from the quieter parts of the
    signal, then subtracts it in the frequency domain. This is especially
    effective against the constant hiss/hum of cheap USB microphones.

    stationary=True assumes the noise floor doesn't change over time,
    which is a good assumption for mic self-noise and room tone.
    """
    return nr.reduce_noise(
        y=audio,
        sr=sample_rate,
        stationary=True,
        prop_decrease=0.75,  # how aggressively to suppress noise (0=none, 1=full)
    ).astype(np.float32)


# --- Normalization ---

# Target RMS level for the audio signal. -20 dBFS is a conservative level
# that leaves headroom and works well with Whisper's expected input range.
TARGET_RMS_DBFS = -20


def normalize_rms(audio, target_db=TARGET_RMS_DBFS):
    """Normalize audio to a target RMS level in dBFS.

    Unlike peak normalization, RMS-based normalization matches perceived
    loudness, so whispered and shouted speech both reach Whisper at a
    similar energy level. This is much more robust than the previous
    peak-amplitude gain boost.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        # Silence — nothing to normalize
        return audio
    target_rms = 10 ** (target_db / 20.0)
    gain = target_rms / rms
    audio = audio * gain
    # Clip to [-1, 1] to prevent overflow when converting to int16 later
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# --- Voice activity detection ---


def detect_first_speech(audio_data, sample_rate, threshold=0.2, frame_duration=0.02):
    """Find the timestamp (in seconds) where speech first appears.

    Splits the audio into short frames, computes the energy of each,
    and returns the time of the first frame whose normalized energy
    exceeds the threshold. Returns None if no speech is found.
    """
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    frame_size = int(frame_duration * sample_rate)
    frames = [audio_data[i : i + frame_size] for i in range(0, len(audio_data), frame_size)]
    energy = [np.sum(np.abs(frame) ** 2) / len(frame) for frame in frames]

    max_energy = max(energy)
    if max_energy > 0:
        energy = [e / max_energy for e in energy]

    for i, e in enumerate(energy):
        if e > threshold:
            return round(i * frame_duration, 1)

    return None


# --- Main audio improvement pipeline ---


def improve_input_audio(audio, sample_rate=audio_utils.SAMPLE_RATE, vad=True):
    """Clean up raw microphone audio for better Whisper transcription.

    Pipeline order matters:
      1. Bandpass first  — removes out-of-band noise before it affects later steps
      2. Noise reduction — works on the filtered signal for cleaner estimation
      3. Normalize       — bring speech to consistent level after filtering
      4. VAD last        — detect speech onset in the cleaned signal
    """
    audio = bandpass_filter(audio, sample_rate)
    audio = reduce_noise(audio, sample_rate)
    audio = normalize_rms(audio)

    start_time = 0
    if vad:
        start_time = detect_first_speech(audio, sample_rate, threshold=0.2, frame_duration=0.02)
        if start_time is None:
            pass  # No speech detected — caller handles this

    return audio, start_time


# --- Mel spectrogram chunking for Hailo ---


def preprocess(audio, is_nhwc=False, chunk_length=10, chunk_offset=0, max_duration=60, overlap=0.0):
    """Split audio into fixed-length chunks and convert to mel spectrograms.

    Whisper's encoder expects mel spectrograms of a fixed duration (the
    chunk_length, typically 10s or 30s depending on the HEF model).
    This function slices the audio into those chunks, pads the last one
    if needed, and converts each to a log-mel spectrogram in the tensor
    layout the Hailo model expects.

    Args:
        is_nhwc: If True, output tensors in NHWC layout (Hailo convention).
                 If False, use NCHW (PyTorch convention).
        chunk_offset: Skip this many seconds from the start (e.g., to skip
                      silence detected by VAD).
        overlap: Fraction of overlap between consecutive chunks (0.0 = none).
    """
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
