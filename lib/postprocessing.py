# Adapted from hailo-apps speech recognition
import numpy as np
import re

excluded_tokens = [11, 13]  # Punctuation tokens to exclude from repetition penalty


def apply_repetition_penalty(logits, generated_tokens, penalty=1.5, last_window=8):
    logits = np.squeeze(logits, axis=0)
    recent_tokens = generated_tokens[-last_window:] if len(generated_tokens) >= last_window else generated_tokens
    recent_tokens = set(recent_tokens)
    for token in recent_tokens:
        if token not in excluded_tokens:
            logits[token] /= penalty
    return logits


def clean_transcription(transcription):
    sentences = re.split(r'(?<=[.?])\s+', transcription)
    unique_sentences = []

    for sentence in sentences:
        for unique_sentence in unique_sentences:
            normalized_current = sentence.lower().strip()
            normalized_unique = unique_sentence.lower().strip()
            if normalized_current in normalized_unique or normalized_unique in normalized_current:
                cleaned_transcription = ' '.join(unique_sentences)
                if not cleaned_transcription.endswith(('.', '?')):
                    cleaned_transcription += '.'
                return cleaned_transcription
        unique_sentences.append(sentence.strip())

    cleaned_transcription = ' '.join(unique_sentences)
    if not cleaned_transcription.endswith(('.', '?')):
        cleaned_transcription += '.'
    return cleaned_transcription
