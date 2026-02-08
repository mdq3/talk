"""LLM inference on Hailo NPU via hailo_platform.genai.LLM.

Shares the same VDevice as the Whisper pipeline for zero-overhead model switching.
"""

import json
import os
import sys

DEFAULT_SYSTEM_PROMPT = "Respond in up to three sentences."
DEFAULT_LLM_MODEL = "qwen2"

HAILO_OLLAMA_MODELS = "/usr/share/hailo-ollama/models"


def resolve_hef_path(model_name):
    """Resolve an LLM model name to its HEF file path.

    Looks up the hailo-ollama model store for the manifest and blob.
    """
    # Find manifest: try model_name/*/manifest.json
    manifests_dir = os.path.join(HAILO_OLLAMA_MODELS, "manifests", model_name)
    if not os.path.isdir(manifests_dir):
        raise FileNotFoundError(f"LLM model '{model_name}' not found in {HAILO_OLLAMA_MODELS}")

    # Pick the first available size variant
    for variant in sorted(os.listdir(manifests_dir)):
        manifest_path = os.path.join(manifests_dir, variant, "manifest.json")
        if os.path.isfile(manifest_path):
            break
    else:
        raise FileNotFoundError(f"No manifest found for model '{model_name}'")

    with open(manifest_path) as f:
        manifest = json.load(f)

    sha = manifest["hef_h10h"]
    hef_path = os.path.join(HAILO_OLLAMA_MODELS, "blob", f"sha256_{sha}")
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(f"HEF blob not found: {hef_path}")

    stop_tokens = manifest.get("generation_params", {}).get("stop_tokens", [])
    return hef_path, stop_tokens


class HailoLLM:
    """LLM running on the Hailo NPU, sharing a VDevice with Whisper."""

    def __init__(self, vdevice, model_name=DEFAULT_LLM_MODEL):
        from hailo_platform.genai import LLM

        hef_path, self.stop_tokens = resolve_hef_path(model_name)
        self.llm = LLM(vdevice, hef_path)
        self.model_name = model_name

    def chat(self, messages, system_prompt=None, max_tokens=200):
        """Stream chat tokens. Yields cleaned token strings."""
        all_messages = list(messages)
        if system_prompt:
            all_messages.insert(0, {"role": "system", "content": system_prompt})

        self.llm.clear_context()
        with self.llm.generate(
            prompt=all_messages, temperature=0.7, max_generated_tokens=max_tokens
        ) as gen:
            for token in gen:
                # Filter stop tokens that leak through
                if token in self.stop_tokens:
                    break
                yield token

    def release(self):
        self.llm.release()


def stream_to_terminal(token_generator):
    """Print tokens as they arrive and return the full response text."""
    full = []
    for token in token_generator:
        sys.stdout.write(token)
        sys.stdout.flush()
        full.append(token)
    print()
    return "".join(full)
