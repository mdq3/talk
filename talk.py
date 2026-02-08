#!/usr/bin/env python3
"""Voice-to-text using Whisper on Hailo AI HAT+."""

import argparse
import json
import os
import sys
import time



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


def load_boost_words(args):
    boost_words = {}
    if args.boost_file and os.path.exists(args.boost_file):
        with open(args.boost_file) as f:
            boost_words = json.load(f)
    for entry in args.boost:
        if ":" in entry:
            word, factor = entry.rsplit(":", 1)
            boost_words[word] = float(factor)
        else:
            boost_words[entry] = 1.5
    return boost_words


def main():
    args = parse_args()

    from lib.pipeline import HailoWhisperPipeline, get_hef_paths
    from lib.audio_utils import load_audio
    from lib.preprocessing import preprocess, improve_input_audio
    from lib.postprocessing import clean_transcription
    from lib.record_utils import record_audio

    print(f"Variant: whisper-{args.variant}")
    print(f"Hardware: {args.hw_arch}")

    encoder_path, decoder_path = get_hef_paths(args.variant, args.hw_arch)

    boost_words = load_boost_words(args)
    if boost_words:
        print(f"Word boost: {boost_words}")

    print("Loading Hailo Whisper pipeline...")
    pipeline = HailoWhisperPipeline(encoder_path, decoder_path, args.variant, boost_words=boost_words)
    print("Pipeline ready.")

    chunk_length = pipeline.get_model_input_audio_length()
    audio_path = "/tmp/talk_recording.wav"

    try:
        while True:
            user_input = input("\nPress Enter to record, or 'q' to quit: ")
            if user_input.strip().lower() == "q":
                break

            record_audio(args.duration, audio_path)

            audio = load_audio(audio_path)
            audio, start_time = improve_input_audio(audio, vad=True)

            if start_time is None:
                print("No speech detected. Try again.")
                continue

            chunk_offset = max(0, start_time - 0.2)
            mel_spectrograms = preprocess(
                audio,
                is_nhwc=True,
                chunk_length=chunk_length,
                chunk_offset=chunk_offset,
            )

            for mel in mel_spectrograms:
                pipeline.send_data(mel)
                time.sleep(0.1)
                transcription = clean_transcription(pipeline.get_transcription())
                print(f"\n>>> {transcription}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down pipeline...")
        pipeline.stop()
        print("Done.")


if __name__ == "__main__":
    main()
