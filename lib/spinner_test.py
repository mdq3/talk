from threading import Event, Thread

from lib.spinner import _build_text_frames, loading, spinner


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


class TestBuildTextFrames:
    def test_frame_count_matches_ping_pong(self):
        frames = _build_text_frames("hello")
        # ping-pong: 0,1,2,3,4,3,2,1 = 5 + 3 = 8
        assert len(frames) == 2 * len("hello") - 2

    def test_single_char_returns_plain(self):
        frames = _build_text_frames("x")
        assert frames == ["x"]

    def test_empty_string_returns_plain(self):
        frames = _build_text_frames("")
        assert frames == [""]

    def test_frames_contain_all_original_chars(self):
        text = "abc"
        frames = _build_text_frames(text)
        for frame in frames:
            # Strip ANSI codes to get raw text
            raw = ""
            i = 0
            while i < len(frame):
                if frame[i] == "\033":
                    while i < len(frame) and frame[i] != "m":
                        i += 1
                    i += 1  # skip the 'm'
                else:
                    raw += frame[i]
                    i += 1
            assert raw == text

    def test_frames_contain_ansi_codes(self):
        frames = _build_text_frames("hi")
        for frame in frames:
            assert "\033[" in frame

    def test_frames_end_with_reset(self):
        frames = _build_text_frames("hi")
        for frame in frames:
            assert frame.endswith("\033[0m")


class TestLoading:
    def test_returns_func_result(self):
        result = loading("test", lambda: 42)
        assert result == 42

    def test_passes_through_return_value(self):
        result = loading("test", lambda: {"key": "value"})
        assert result == {"key": "value"}

    def test_propagates_exception(self):
        def fail():
            raise ValueError("boom")

        try:
            loading("test", fail)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert str(e) == "boom"

    def test_custom_messages(self, capsys):
        loading("x", lambda: None, spin_message="Working...", done_message="Finished!")
        captured = capsys.readouterr()
        assert "Finished!" in captured.out

    def test_default_done_message(self, capsys):
        loading("widgets", lambda: None)
        captured = capsys.readouterr()
        assert "Loaded widgets." in captured.out
