import sys
from itertools import cycle
from threading import Event, Thread

SPINNER_CHARS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

def _spin(message, done):
    for char in cycle(SPINNER_CHARS):
        if done.is_set():
            break
        sys.stdout.write(f"\r{char} {message}")
        sys.stdout.flush()
        done.wait(0.08)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def spinner(message):
    done = Event()
    thread = Thread(target=_spin, args=(message, done), daemon=True)
    thread.start()
    return done, thread
