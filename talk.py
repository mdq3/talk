#!/usr/bin/env python3
"""Voice-to-text using Whisper on Hailo AI HAT+."""

import argparse
import os

from lib.spinner import loading


def parse_args():
    parser = argparse.ArgumentParser(description="Speech-to-text using Whisper on Hailo AI HAT+")
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "tiny", "tiny.en"],
        help="Whisper model variant (default: base)",
    )
    parser.add_argument(
        "--hw-arch",
        type=str,
        default="hailo10h",
        choices=["hailo8", "hailo8l", "hailo10h"],
        help="Hailo hardware architecture (default: hailo10h)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Max recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--boost",
        action="append",
        default=[],
        help="Boost a word during decoding. Format: word:factor (default factor 1.5). Repeatable.",
    )
    default_boost_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "boost_words.json"
    )
    parser.add_argument(
        "--boost-file",
        type=str,
        default=default_boost_file,
        help="Path to JSON file with word boost factors (default: boost_words.json)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen2",
        help="LLM model name from hailo-ollama models (default: qwen2)",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="en_US-amy-medium",
        help="Piper TTS voice name (default: en_US-amy-medium)",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS voice output (text-only)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the default LLM system prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create TTS first so the audio stream warms up during heavy Hailo imports
    tts = None
    if not args.no_tts:

        def _load_tts():
            from lib.tts import PiperTTS

            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "piper")
            return PiperTTS(models_dir, args.tts_voice)

        tts = loading("TTS voice", _load_tts)

    def _import():
        from lib.app import run
        from lib.boost_words import load_boost_words
        return run, load_boost_words

    run, load_boost_words = loading("tools", _import)

    boost_words = load_boost_words(args.boost_file, args.boost)

    chat_opts = {
        "llm_model": args.llm_model,
        "system_prompt": args.system_prompt,
        "tts": tts,
    }

    run(args.variant, args.hw_arch, args.duration, boost_words, chat_opts=chat_opts)


if __name__ == "__main__":
    main()
