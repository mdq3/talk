import os
import signal
import sys
import time
from itertools import cycle
from threading import Event, Thread

SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def spinner(message):
    """Start a spinner on a background thread. Returns (done_event, thread)."""
    done = Event()
    thread = Thread(target=_spin, args=(message, done), daemon=True)
    thread.start()
    return done, thread


def loading(message, func):
    """Run func() while showing a spinner in a forked child process.

    The child process has its own GIL, so the spinner stays animated even
    when func() runs GIL-holding C extension code.
    """
    pid = os.fork()
    if pid == 0:
        i = 0
        try:
            while True:
                os.write(1, f"\r{SPINNER_CHARS[i % 10]} Loading {message}...".encode())
                time.sleep(0.08)
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

    print(f"Loaded {message}.")
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
