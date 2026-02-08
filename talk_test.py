import sys
from unittest.mock import patch

from talk import parse_args


class TestParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["talk.py"]):
            args = parse_args()
        assert args.variant == "base"
        assert args.hw_arch == "hailo10h"
        assert args.duration == 10
        assert args.boost == []
        assert args.boost_file.endswith("boost_words.json")

    def test_variant_choices(self):
        for variant in ["base", "tiny", "tiny.en"]:
            with patch("sys.argv", ["talk.py", "--variant", variant]):
                args = parse_args()
            assert args.variant == variant

    def test_invalid_variant_exits(self):
        with patch("sys.argv", ["talk.py", "--variant", "large"]):
            try:
                parse_args()
                assert False, "Should have raised SystemExit"
            except SystemExit:
                pass

    def test_hw_arch_choices(self):
        for arch in ["hailo8", "hailo8l", "hailo10h"]:
            with patch("sys.argv", ["talk.py", "--hw-arch", arch]):
                args = parse_args()
            assert args.hw_arch == arch

    def test_duration(self):
        with patch("sys.argv", ["talk.py", "--duration", "20"]):
            args = parse_args()
        assert args.duration == 20

    def test_boost_single(self):
        with patch("sys.argv", ["talk.py", "--boost", "hello:2.0"]):
            args = parse_args()
        assert args.boost == ["hello:2.0"]

    def test_boost_multiple(self):
        with patch("sys.argv", ["talk.py", "--boost", "hello:2.0", "--boost", "world:1.5"]):
            args = parse_args()
        assert args.boost == ["hello:2.0", "world:1.5"]

    def test_boost_file(self):
        with patch("sys.argv", ["talk.py", "--boost-file", "/tmp/custom.json"]):
            args = parse_args()
        assert args.boost_file == "/tmp/custom.json"
