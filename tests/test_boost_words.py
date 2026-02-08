import json

from lib.boost_words import load_boost_words


class TestLoadBoostWords:
    def test_loads_from_json_file(self, tmp_path):
        f = tmp_path / "boost.json"
        f.write_text(json.dumps({"hello": 2.0, "world": 1.5}))
        result = load_boost_words(str(f), [])
        assert result == {"hello": 2.0, "world": 1.5}

    def test_parses_cli_args_with_factor(self):
        result = load_boost_words(None, ["hello:2.0", "world:3.0"])
        assert result == {"hello": 2.0, "world": 3.0}

    def test_default_factor_when_no_colon(self):
        result = load_boost_words(None, ["hello"])
        assert result == {"hello": 1.5}

    def test_cli_overrides_file(self, tmp_path):
        f = tmp_path / "boost.json"
        f.write_text(json.dumps({"hello": 2.0, "other": 1.0}))
        result = load_boost_words(str(f), ["hello:5.0"])
        assert result["hello"] == 5.0
        assert result["other"] == 1.0

    def test_empty_when_no_file_no_args(self):
        result = load_boost_words(None, [])
        assert result == {}

    def test_missing_file_gracefully(self):
        result = load_boost_words("/nonexistent/path/boost.json", [])
        assert result == {}

    def test_colon_in_word(self):
        result = load_boost_words(None, ["key:word:2.0"])
        assert result == {"key:word": 2.0}
