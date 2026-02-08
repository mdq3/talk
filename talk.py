#!/usr/bin/env python3
"""Voice-to-text using Whisper on Hailo AI HAT+."""

import argparse
import os


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
    default_boost_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "boost_words.json")
    parser.add_argument(
        "--boost-file",
        type=str,
        default=default_boost_file,
        help="Path to JSON file with word boost factors (default: boost_words.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from lib.boost_words import load_boost_words
    from lib.app import run

    boost_words = load_boost_words(args.boost_file, args.boost)
    run(args.variant, args.hw_arch, args.duration, boost_words)


if __name__ == "__main__":
    main()
