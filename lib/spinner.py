import os
import signal
import sys
import time
from itertools import cycle
from threading import Event, Thread

SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# 256-color gradient for text highlight: dim gray → bright white
_TEXT_COLORS = [240, 244, 248, 252, 255]


def _build_text_frames(text):
    """Precompute frames with a bright highlight bouncing across the text."""
    n = len(text)
    nc = len(_TEXT_COLORS)

    if n <= 1:
        return [text]

    positions = list(range(n)) + list(range(n - 2, 0, -1))
    frames = []

    for center in positions:
        parts = []
        for i, ch in enumerate(text):
            ci = max(0, nc - 1 - abs(i - center))
            parts.append(f"\033[38;5;{_TEXT_COLORS[ci]}m{ch}")
        parts.append("\033[0m")
        frames.append("".join(parts))

    return frames


def spinner(message):
    """Start a spinner on a background thread. Returns (done_event, thread)."""
    done = Event()
    thread = Thread(target=_spin, args=(message, done), daemon=True)
    thread.start()
    return done, thread


def loading(message, func, done_message=None, spin_message=None):
    """Run func() while showing an animated spinner in a forked child process.

    A bright highlight sweeps back and forth across the message text, with a
    braille spinner at the front. The child process has its own GIL, so the
    animation stays smooth even when func() runs GIL-holding C extension code.
    """
    if done_message is None:
        done_message = f"Loaded {message}."
    spin_text = spin_message or f"Loading {message}..."

    frames = _build_text_frames(spin_text)
    n_frames = len(frames)
    n_spin = len(SPINNER_CHARS)

    pid = os.fork()
    if pid == 0:
        i = 0
        try:
            while True:
                spin = SPINNER_CHARS[i // 2 % n_spin]
                text = frames[i % n_frames]
                os.write(1, f"\r{spin} {text}".encode())
                time.sleep(0.04)
                i += 1
        except BaseException:
            pass
        os._exit(0)

    try:
        result = func()
    finally:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    print(done_message)
    return result


def _spin(message, done):
    for char in cycle(SPINNER_CHARS):
        if done.is_set():
            break
        sys.stdout.write(f"\r{char} {message}")
        sys.stdout.flush()
        done.wait(0.08)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
