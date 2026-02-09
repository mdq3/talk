import sys
import time

GREEN = "\033[32m"
RESET = "\033[0m"

from .audio_utils import load_audio
from .pipeline import HailoWhisperPipeline, create_shared_vdevice, get_hef_paths
from .postprocessing import clean_transcription
from .preprocessing import improve_input_audio, preprocess
from .record_utils import record_audio


def run(variant, hw_arch, duration, boost_words, chat_opts=None):
    from .spinner import loading

    encoder_path, decoder_path = get_hef_paths(variant, hw_arch)

    llm = None
    vdevice = None

    if chat_opts:
        from .llm import HailoLLM

        vdevice = loading("Hailo device", create_shared_vdevice)
        llm = loading(
            f"LLM ({chat_opts['llm_model']})",
            lambda: HailoLLM(vdevice, chat_opts["llm_model"]),
        )

    pipeline = loading(
        f"Whisper ({variant})",
        lambda: HailoWhisperPipeline(
            encoder_path, decoder_path, variant, boost_words=boost_words, vdevice=vdevice
        ),
    )

    chunk_length = pipeline.get_model_input_audio_length()
    audio_path = "/tmp/talk_recording.wav"
    chat_history = []
    last_response = None

    try:
        while True:
            tts = chat_opts["tts"] if chat_opts else None
            opts = ["'w' to type"]
            if tts and last_response:
                opts.append("'r' to replay")
            opts.append("'q' to quit")
            prompt = f"\nPress Enter to record, {', '.join(opts)}: "

            user_input = input(prompt).strip().lower()
            if user_input == "q":
                break
            if user_input == "r" and tts and last_response:
                tts.speak(last_response)
                continue

            if user_input == "w":
                transcription = input(f"{GREEN}>>> ").strip()
                sys.stdout.write(RESET)
                if not transcription:
                    continue
            else:
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
                    print(f"\n{GREEN}>>> {transcription}{RESET}")

            if llm and transcription:
                last_response = _chat_respond(transcription, llm, chat_opts, chat_history)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down...")
        pipeline.stop()
        if llm:
            llm.release()
        if vdevice:
            vdevice.release()
        tts = chat_opts["tts"] if chat_opts else None
        if tts:
            tts.close()
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
        return None

    history.append({"role": "assistant", "content": response})

    tts = chat_opts["tts"]
    if tts and response.strip():
        tts.speak(response)

    return response
