#!/usr/bin/env python3
"""Voice-to-text using Whisper on Hailo AI HAT+."""

import argparse
import os
import sys
import time

from lib.pipeline import HailoWhisperPipeline, get_hef_paths
from lib.audio_utils import load_audio
from lib.preprocessing import preprocess, improve_input_audio
from lib.postprocessing import clean_transcription
from lib.record_utils import record_audio


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
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Variant: whisper-{args.variant}")
    print(f"Hardware: {args.hw_arch}")

    encoder_path, decoder_path = get_hef_paths(args.variant, args.hw_arch)

    print("Loading Hailo Whisper pipeline...")
    pipeline = HailoWhisperPipeline(encoder_path, decoder_path, args.variant)
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
        pipeline.stop()


if __name__ == "__main__":
    main()
