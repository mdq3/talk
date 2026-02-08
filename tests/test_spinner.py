from threading import Event, Thread

from lib.spinner import spinner


class TestSpinner:
    def test_returns_event_and_thread(self):
        done, thread = spinner("test")
        assert isinstance(done, Event)
        assert isinstance(thread, Thread)
        done.set()
        thread.join(timeout=1)

    def test_thread_is_alive_after_start(self):
        done, thread = spinner("test")
        assert thread.is_alive()
        done.set()
        thread.join(timeout=1)

    def test_thread_stops_after_done(self):
        done, thread = spinner("test")
        done.set()
        thread.join(timeout=1)
        assert not thread.is_alive()

    def test_thread_is_daemon(self):
        done, thread = spinner("test")
        assert thread.daemon
        done.set()
        thread.join(timeout=1)
