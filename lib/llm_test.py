from lib.llm import stream_to_terminal


class TestStreamToTerminal:
    def test_returns_full_response(self):
        tokens = ["Hello", " ", "world"]
        result = stream_to_terminal(iter(tokens))
        assert result == "Hello world"

    def test_empty_generator(self):
        result = stream_to_terminal(iter([]))
        assert result == ""

    def test_single_token(self):
        result = stream_to_terminal(iter(["Hi"]))
        assert result == "Hi"
