from lib.tts import clean_text_for_tts


class TestCleanTextForTts:
    def test_strips_bold(self):
        assert clean_text_for_tts("**bold**") == "bold"

    def test_strips_italic(self):
        assert clean_text_for_tts("*italic*") == "italic"

    def test_strips_bold_italic(self):
        assert clean_text_for_tts("***both***") == "both"

    def test_strips_underscores(self):
        assert clean_text_for_tts("__bold__ and _italic_") == "bold and italic"

    def test_strips_backticks(self):
        assert clean_text_for_tts("use `code` here") == "use code here"

    def test_strips_code_block(self):
        assert clean_text_for_tts("```python\nprint()```") == "python print()"

    def test_strips_headers(self):
        assert clean_text_for_tts("## Heading\nBody") == "Heading Body"

    def test_converts_links(self):
        assert clean_text_for_tts("[click here](https://example.com)") == "click here"

    def test_strips_noisy_symbols(self):
        assert clean_text_for_tts("a~b@c^d|e") == "a b c d e"

    def test_normalizes_whitespace(self):
        assert clean_text_for_tts("too   many    spaces") == "too many spaces"

    def test_plain_text_unchanged(self):
        assert clean_text_for_tts("Hello world.") == "Hello world."

    def test_empty_string(self):
        assert clean_text_for_tts("") == ""

    def test_combined_markdown(self):
        text = "## Title\n**Bold** and *italic* with [a link](url)"
        result = clean_text_for_tts(text)
        assert "**" not in result
        assert "*" not in result
        assert "##" not in result
        assert "[" not in result
        assert "Bold" in result
        assert "italic" in result
        assert "a link" in result
