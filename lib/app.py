import time

from .audio_utils import load_audio
from .pipeline import HailoWhisperPipeline, get_hef_paths
from .postprocessing import clean_transcription
from .preprocessing import improve_input_audio, preprocess
from .record_utils import record_audio


def run(variant, hw_arch, duration, boost_words):
    print(f"Variant: whisper-{variant}")
    print(f"Hardware: {hw_arch}")

    encoder_path, decoder_path = get_hef_paths(variant, hw_arch)

    if boost_words:
        print(f"Word boost: {boost_words}")

    print("Loading Hailo Whisper pipeline...")
    pipeline = HailoWhisperPipeline(encoder_path, decoder_path, variant, boost_words=boost_words)
    print("Pipeline ready.")

    chunk_length = pipeline.get_model_input_audio_length()
    audio_path = "/tmp/talk_recording.wav"

    try:
        while True:
            user_input = input("\nPress Enter to record, or 'q' to quit: ")
            if user_input.strip().lower() == "q":
                break

            record_audio(duration, audio_path)

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
