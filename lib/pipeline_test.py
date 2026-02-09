from lib.pipeline import HEF_REGISTRY, get_hef_paths


class TestHefRegistry:
    def test_all_variants_have_encoder_and_decoder(self):
        for variant, archs in HEF_REGISTRY.items():
            for arch, paths in archs.items():
                assert "encoder" in paths, f"Missing encoder for {variant}/{arch}"
                assert "decoder" in paths, f"Missing decoder for {variant}/{arch}"

    def test_base_supports_all_architectures(self):
        assert "hailo10h" in HEF_REGISTRY["base"]
        assert "hailo8" in HEF_REGISTRY["base"]
        assert "hailo8l" in HEF_REGISTRY["base"]

    def test_tiny_en_only_hailo10h(self):
        assert "hailo10h" in HEF_REGISTRY["tiny.en"]
        assert len(HEF_REGISTRY["tiny.en"]) == 1


class TestGetHefPaths:
    def test_invalid_variant_raises(self):
        try:
            get_hef_paths("nonexistent", "hailo10h")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "nonexistent" in str(e)

    def test_invalid_arch_raises(self):
        try:
            get_hef_paths("base", "hailo_fake")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "hailo_fake" in str(e)

    def test_valid_combo_returns_paths(self, tmp_path, monkeypatch):
        # Create fake HEF files
        enc = tmp_path / "hefs" / "h10h" / "base" / "base-whisper-encoder-10s.hef"
        dec = tmp_path / "hefs" / "h10h" / "base" / "base-whisper-decoder-10s-out-seq-64.hef"
        enc.parent.mkdir(parents=True)
        enc.write_text("fake")
        dec.write_text("fake")

        monkeypatch.setattr("lib.pipeline.MODELS_DIR", str(tmp_path))
        encoder_path, decoder_path = get_hef_paths("base", "hailo10h")
        assert encoder_path.endswith(".hef")
        assert decoder_path.endswith(".hef")

    def test_missing_hef_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lib.pipeline.MODELS_DIR", str(tmp_path))
        try:
            get_hef_paths("base", "hailo10h")
            assert False, "Should have raised"
        except FileNotFoundError as e:
            assert "not found" in str(e)
