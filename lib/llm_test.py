import json

from lib.llm import resolve_hef_path, stream_to_terminal


class TestResolveHefPath:
    def test_resolves_valid_model(self, tmp_path, monkeypatch):
        # Build a fake model store
        manifest_dir = tmp_path / "manifests" / "qwen2" / "1B"
        manifest_dir.mkdir(parents=True)
        sha = "abc123"
        manifest = {"hef_h10h": sha, "generation_params": {"stop_tokens": ["<|end|>"]}}
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
        blob_dir = tmp_path / "blob"
        blob_dir.mkdir()
        (blob_dir / f"sha256_{sha}").write_text("fake hef")

        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        hef_path, stop_tokens = resolve_hef_path("qwen2")
        assert hef_path == str(blob_dir / f"sha256_{sha}")
        assert stop_tokens == ["<|end|>"]

    def test_missing_model_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        try:
            resolve_hef_path("nonexistent")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "nonexistent" in str(e)

    def test_no_manifest_file_raises(self, tmp_path, monkeypatch):
        # Dir exists but no manifest.json inside
        (tmp_path / "manifests" / "qwen2" / "1B").mkdir(parents=True)
        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        try:
            resolve_hef_path("qwen2")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "No manifest" in str(e)

    def test_missing_blob_raises(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifests" / "qwen2" / "1B"
        manifest_dir.mkdir(parents=True)
        manifest = {"hef_h10h": "missing_sha"}
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "blob").mkdir()
        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        try:
            resolve_hef_path("qwen2")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "blob" in str(e)

    def test_no_stop_tokens_defaults_to_empty(self, tmp_path, monkeypatch):
        manifest_dir = tmp_path / "manifests" / "model" / "v1"
        manifest_dir.mkdir(parents=True)
        manifest = {"hef_h10h": "sha1"}
        (manifest_dir / "manifest.json").write_text(json.dumps(manifest))
        blob_dir = tmp_path / "blob"
        blob_dir.mkdir()
        (blob_dir / "sha256_sha1").write_text("fake")
        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        _, stop_tokens = resolve_hef_path("model")
        assert stop_tokens == []

    def test_picks_first_sorted_variant(self, tmp_path, monkeypatch):
        # Two size variants — should pick alphabetically first
        for variant in ["7B", "1B"]:
            d = tmp_path / "manifests" / "m" / variant
            d.mkdir(parents=True)
            manifest = {"hef_h10h": f"sha_{variant}"}
            (d / "manifest.json").write_text(json.dumps(manifest))
        blob_dir = tmp_path / "blob"
        blob_dir.mkdir()
        (blob_dir / "sha256_sha_1B").write_text("fake")
        monkeypatch.setattr("lib.llm.HAILO_OLLAMA_MODELS", str(tmp_path))
        hef_path, _ = resolve_hef_path("m")
        assert "sha_1B" in hef_path


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
