import sys
import time

from .audio_utils import load_audio
from .pipeline import HailoWhisperPipeline, create_shared_vdevice, get_hef_paths
from .postprocessing import clean_transcription
from .preprocessing import improve_input_audio, preprocess
from .record_utils import record_audio


def run(variant, hw_arch, duration, boost_words, chat_opts=None):
    print(f"Variant: whisper-{variant}")
    print(f"Hardware: {hw_arch}")

    encoder_path, decoder_path = get_hef_paths(variant, hw_arch)

    if boost_words:
        print(f"Word boost: {boost_words}")

    llm = None
    vdevice = None

    if chat_opts:
        print(f"Chat mode: {chat_opts['llm_model']}")
        if chat_opts["tts"]:
            print("TTS: enabled")
        else:
            print("TTS: disabled")

        # Create shared VDevice for both Whisper and LLM
        from .llm import HailoLLM

        print("Creating shared Hailo device...")
        vdevice = create_shared_vdevice()

        print(f"Loading LLM ({chat_opts['llm_model']})...")
        llm = HailoLLM(vdevice, chat_opts["llm_model"])

    print("Loading Hailo Whisper pipeline...")
    pipeline = HailoWhisperPipeline(
        encoder_path, decoder_path, variant, boost_words=boost_words, vdevice=vdevice
    )
    print("Pipeline ready.")

    chunk_length = pipeline.get_model_input_audio_length()
    audio_path = "/tmp/talk_recording.wav"
    chat_history = []

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

            if llm and transcription:
                _chat_respond(transcription, llm, chat_opts, chat_history)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down...")
        pipeline.stop()
        if llm:
            llm.release()
        if vdevice:
            vdevice.release()
        print("Done.")


def _chat_respond(transcription, llm, chat_opts, history):
    """Send transcription to LLM, stream response, and optionally speak it."""
    from .llm import DEFAULT_SYSTEM_PROMPT, stream_to_terminal
    from .spinner import spinner

    history.append({"role": "user", "content": transcription})
    system_prompt = chat_opts["system_prompt"] or DEFAULT_SYSTEM_PROMPT

    done, thread = spinner("Thinking...")
    first_token = True

    def token_stream():
        nonlocal first_token
        for token in llm.chat(history, system_prompt):
            if first_token:
                done.set()
                thread.join()
                sys.stdout.write("\n")
                first_token = False
            yield token

    try:
        response = stream_to_terminal(token_stream())
    except Exception as e:
        done.set()
        thread.join()
        print(f"\nLLM error: {e}")
        history.pop()
        return

    history.append({"role": "assistant", "content": response})

    tts = chat_opts["tts"]
    if tts and response.strip():
        tts.speak(response)
