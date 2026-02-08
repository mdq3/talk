import numpy as np

from lib.preprocessing import bandpass_filter, detect_first_speech, normalize_rms

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
