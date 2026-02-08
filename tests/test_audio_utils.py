import numpy as np
import torch

from lib.audio_utils import (
    N_FFT,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    mel_filters,
    pad_or_trim,
)


class TestPadOrTrim:
    def test_pads_short_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = pad_or_trim(arr, 5)
        assert len(result) == 5
        np.testing.assert_array_equal(result[:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[3:], [0.0, 0.0])

    def test_trims_long_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_or_trim(arr, 3)
        assert len(result) == 3
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_exact_length_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = pad_or_trim(arr, 3)
        np.testing.assert_array_equal(result, arr)

    def test_works_with_torch_tensor_pad(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = pad_or_trim(t, 5)
        assert result.shape[0] == 5
        assert torch.equal(result[:3], t)
        assert torch.equal(result[3:], torch.zeros(2))

    def test_works_with_torch_tensor_trim(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pad_or_trim(t, 3)
        assert result.shape[0] == 3
        assert torch.equal(result, t[:3])


class TestMelFilters:
    def test_returns_correct_shape_80(self):
        filters = mel_filters("cpu", 80)
        assert filters.shape == (80, N_FFT // 2 + 1)

    def test_returns_torch_tensor(self):
        filters = mel_filters("cpu", 80)
        assert isinstance(filters, torch.Tensor)


class TestLogMelSpectrogram:
    def test_returns_expected_shape(self):
        audio = np.zeros(N_SAMPLES, dtype=np.float32)
        result = log_mel_spectrogram(audio, n_mels=80)
        assert result.shape[0] == 80  # n_mels
        assert result.shape[1] == N_SAMPLES // 160  # n_frames

    def test_accepts_numpy_input(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32)  # 1 second
        result = log_mel_spectrogram(audio)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 80

    def test_output_values_in_range(self):
        audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        result = log_mel_spectrogram(audio)
        # After normalization: (log10 + 4) / 4, clamped to max - 8
        # Values should be roughly in [-2, 2] range
        assert result.min() >= -3.0
        assert result.max() <= 3.0
