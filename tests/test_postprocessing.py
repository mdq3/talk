import numpy as np

from lib.postprocessing import apply_repetition_penalty, apply_word_boost, clean_transcription


class TestApplyRepetitionPenalty:
    def test_divides_recent_token_logits(self):
        logits = np.array([[10.0, 20.0, 30.0, 40.0]])
        result = apply_repetition_penalty(logits, [1, 2], penalty=2.0)
        assert result[1] == 10.0
        assert result[2] == 15.0

    def test_respects_last_window(self):
        logits = np.array([[10.0, 20.0, 30.0, 40.0]])
        result = apply_repetition_penalty(logits, [1, 2, 3], penalty=2.0, last_window=2)
        assert result[1] == 20.0  # outside window, unchanged
        assert result[2] == 15.0  # in window
        assert result[3] == 20.0  # in window

    def test_skips_excluded_tokens(self):
        logits = np.array([[0.0] * 14])
        logits[0, 11] = 10.0
        logits[0, 13] = 20.0
        result = apply_repetition_penalty(logits, [11, 13], penalty=2.0)
        assert result[11] == 10.0  # excluded, unchanged
        assert result[13] == 20.0  # excluded, unchanged

    def test_empty_generated_tokens(self):
        logits = np.array([[10.0, 20.0, 30.0]])
        result = apply_repetition_penalty(logits, [], penalty=2.0)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])


class TestApplyWordBoost:
    def test_multiplies_logits_for_boosted_tokens(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        boost_map = {1: 2.0, 3: 3.0}
        result = apply_word_boost(logits, boost_map)
        assert result[1] == 4.0
        assert result[3] == 12.0

    def test_skips_special_tokens(self):
        logits = np.zeros(50260)
        logits[50258] = 5.0
        boost_map = {50258: 2.0}
        result = apply_word_boost(logits, boost_map)
        assert result[50258] == 5.0  # unchanged

    def test_no_op_with_empty_map(self):
        logits = np.array([1.0, 2.0, 3.0])
        original = logits.copy()
        result = apply_word_boost(logits, {})
        np.testing.assert_array_equal(result, original)

    def test_multiple_tokens_different_factors(self):
        logits = np.array([10.0, 10.0, 10.0, 10.0])
        boost_map = {0: 1.5, 2: 2.0, 3: 0.5}
        result = apply_word_boost(logits, boost_map)
        assert result[0] == 15.0
        assert result[1] == 10.0  # not boosted
        assert result[2] == 20.0
        assert result[3] == 5.0


class TestCleanTranscription:
    def test_deduplicates_repeated_sentences(self):
        result = clean_transcription("Hello world. Hello world. Goodbye.")
        assert result.count("Hello world") == 1

    def test_appends_period_if_missing(self):
        result = clean_transcription("Hello world")
        assert result.endswith(".")

    def test_preserves_question_mark(self):
        result = clean_transcription("Is this working?")
        assert result == "Is this working?"

    def test_preserves_existing_period(self):
        result = clean_transcription("Hello world.")
        assert result == "Hello world."

    def test_single_sentence(self):
        result = clean_transcription("Just one sentence")
        assert result == "Just one sentence."

    def test_substring_containment(self):
        result = clean_transcription("Hello. Hello world.")
        # "Hello" is contained in "Hello world", so dedup triggers
        assert "Hello" in result
