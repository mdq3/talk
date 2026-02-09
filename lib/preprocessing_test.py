import numpy as np

from lib.preprocessing import (
    bandpass_filter,
    detect_first_speech,
    improve_input_audio,
    normalize_rms,
    preprocess,
    reduce_noise,
)

SAMPLE_RATE = 16000


class TestBandpassFilter:
    def test_passes_voice_frequencies(self):
        t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
        tone_300hz = np.sin(2 * np.pi * 300 * t).astype(np.float32)
        filtered = bandpass_filter(tone_300hz, SAMPLE_RATE)
        # Voice-range tone should retain most energy
        assert np.sqrt(np.mean(filtered**2)) > 0.3

    def test_attenuates_low_frequency(self):
        t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
        tone_30hz = np.sin(2 * np.pi * 30 * t).astype(np.float32)
        filtered = bandpass_filter(tone_30hz, SAMPLE_RATE)
        # Below voice range — should be heavily attenuated
        assert np.sqrt(np.mean(filtered**2)) < 0.1

    def test_returns_float32_same_length(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)
        filtered = bandpass_filter(audio, SAMPLE_RATE)
        assert filtered.dtype == np.float32
        assert len(filtered) == len(audio)


class TestNormalizeRms:
    def test_normalizes_to_target(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.01
        result = normalize_rms(audio, target_db=-20)
        rms = np.sqrt(np.mean(result**2))
        target_rms = 10 ** (-20 / 20.0)
        assert abs(rms - target_rms) < 0.01

    def test_silence_unchanged(self):
        audio = np.zeros(1000, dtype=np.float32)
        result = normalize_rms(audio)
        np.testing.assert_array_equal(result, audio)

    def test_output_clipped(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = normalize_rms(audio, target_db=0)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_returns_float32(self):
        audio = np.random.randn(1000).astype(np.float32) * 0.1
        result = normalize_rms(audio)
        assert result.dtype == np.float32


class TestDetectFirstSpeech:
    def test_finds_onset_after_silence(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second silence
        speech = np.random.randn(SAMPLE_RATE).astype(np.float32)  # 1 second "speech"
        audio = np.concatenate([silence, speech])
        onset = detect_first_speech(audio, SAMPLE_RATE, threshold=0.2)
        assert onset is not None
        # Should detect speech around 1.0 second
        assert 0.8 <= onset <= 1.2

    def test_returns_none_for_silence(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = detect_first_speech(audio, SAMPLE_RATE)
        assert result is None

    def test_handles_stereo_input(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        speech = np.random.randn(SAMPLE_RATE).astype(np.float32)
        mono = np.concatenate([silence, speech])
        stereo = np.column_stack([mono, mono])  # 2D array, shape (N, 2)
        onset = detect_first_speech(stereo, SAMPLE_RATE, threshold=0.2)
        assert onset is not None


class TestReduceNoise:
    def test_returns_float32_same_length(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        result = reduce_noise(audio, SAMPLE_RATE)
        assert result.dtype == np.float32
        assert len(result) == len(audio)

    def test_reduces_constant_noise(self):
        # White noise should be suppressed
        noise = np.random.randn(SAMPLE_RATE * 2).astype(np.float32) * 0.05
        result = reduce_noise(noise, SAMPLE_RATE)
        assert np.sqrt(np.mean(result**2)) < np.sqrt(np.mean(noise**2))


class TestImproveInputAudio:
    def test_returns_audio_and_start_time(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        speech = np.random.randn(SAMPLE_RATE).astype(np.float32)
        audio = np.concatenate([silence, speech])
        result_audio, start_time = improve_input_audio(audio, SAMPLE_RATE)
        assert isinstance(result_audio, np.ndarray)
        assert result_audio.dtype == np.float32
        assert start_time is not None or start_time == 0

    def test_no_vad(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        result_audio, start_time = improve_input_audio(audio, SAMPLE_RATE, vad=False)
        assert start_time == 0

    def test_silence_returns_none_start(self):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        _, start_time = improve_input_audio(audio, SAMPLE_RATE, vad=True)
        assert start_time is None


class TestPreprocess:
    def test_returns_list_of_spectrograms(self):
        # 10 seconds of audio at 16kHz
        audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
        result = preprocess(audio, chunk_length=10)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_spectrogram_shape_nchw(self):
        audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
        result = preprocess(audio, is_nhwc=False, chunk_length=10)
        mel = result[0]
        # NCHW: (1, 80, 1, time_frames)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        assert mel.shape[2] == 1

    def test_spectrogram_shape_nhwc(self):
        audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
        result = preprocess(audio, is_nhwc=True, chunk_length=10)
        mel = result[0]
        # NHWC: (1, 1, time_frames, 80)
        assert mel.shape[0] == 1
        assert mel.shape[1] == 1
        assert mel.shape[3] == 80

    def test_chunk_offset_skips_audio(self):
        # 20 seconds of audio, skip first 10
        audio = np.random.randn(SAMPLE_RATE * 20).astype(np.float32) * 0.1
        full = preprocess(audio, chunk_length=10)
        skipped = preprocess(audio, chunk_length=10, chunk_offset=10)
        assert len(skipped) < len(full)

    def test_multiple_chunks(self):
        # 25 seconds of audio with 10s chunks — should produce multiple chunks
        audio = np.random.randn(SAMPLE_RATE * 25).astype(np.float32) * 0.1
        result = preprocess(audio, chunk_length=10)
        assert len(result) >= 2
