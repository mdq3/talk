import numpy as np

from lib.record_utils import _anti_alias_and_resample


class TestAntiAliasAndResample:
    def test_downsamples_to_target_length(self):
        # 1 second at 44100 Hz -> 1 second at 16000 Hz
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        result = _anti_alias_and_resample(audio, 44100, 16000)
        assert len(result) == 16000

    def test_returns_float32(self):
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        result = _anti_alias_and_resample(audio, 44100, 16000)
        assert result.dtype == np.float32

    def test_preserves_low_frequency_content(self):
        # A 300 Hz tone should survive downsampling to 16 kHz (Nyquist = 8 kHz)
        t = np.linspace(0, 1.0, 44100, dtype=np.float32)
        tone = np.sin(2 * np.pi * 300 * t)
        result = _anti_alias_and_resample(tone, 44100, 16000)
        rms = np.sqrt(np.mean(result**2))
        assert rms > 0.2

    def test_removes_above_nyquist(self):
        # A 10 kHz tone is above 16 kHz Nyquist/2 = 8 kHz, should be suppressed
        t = np.linspace(0, 1.0, 44100, dtype=np.float32)
        tone = np.sin(2 * np.pi * 10000 * t)
        result = _anti_alias_and_resample(tone, 44100, 16000)
        rms = np.sqrt(np.mean(result**2))
        assert rms < 0.1

    def test_other_rate_ratios(self):
        # 48000 -> 16000 (3:1 ratio)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = _anti_alias_and_resample(audio, 48000, 16000)
        assert len(result) == 16000
