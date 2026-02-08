# Adapted from hailo-apps speech recognition
import numpy as np
import os
from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm, FormatType
from transformers import AutoTokenizer
from queue import Queue, Empty
from threading import Thread

from .postprocessing import apply_repetition_penalty, apply_word_boost

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")

# Map variant + hw_arch to HEF filenames
HEF_REGISTRY = {
    "base": {
        "hailo10h": {
            "encoder": "hefs/h10h/base/base-whisper-encoder-10s.hef",
            "decoder": "hefs/h10h/base/base-whisper-decoder-10s-out-seq-64.hef",
        },
        "hailo8": {
            "encoder": "hefs/h8/base/base-whisper-encoder-5s.hef",
            "decoder": "hefs/h8/base/base-whisper-decoder-fixed-sequence-matmul-split.hef",
        },
        "hailo8l": {
            "encoder": "hefs/h8l/base/base-whisper-encoder-5s_h8l.hef",
            "decoder": "hefs/h8l/base/base-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
        },
    },
    "tiny": {
        "hailo10h": {
            "encoder": "hefs/h10h/tiny/tiny-whisper-encoder-10s.hef",
            "decoder": "hefs/h10h/tiny/tiny-whisper-decoder-fixed-sequence.hef",
        },
        "hailo8": {
            "encoder": "hefs/h8/tiny/tiny-whisper-encoder-10s_15dB.hef",
            "decoder": "hefs/h8/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split.hef",
        },
        "hailo8l": {
            "encoder": "hefs/h8l/tiny/tiny-whisper-encoder-10s_15dB_h8l.hef",
            "decoder": "hefs/h8l/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
        },
    },
    "tiny.en": {
        "hailo10h": {
            "encoder": "hefs/h10h/tiny.en/tiny_en-whisper-encoder-10s.hef",
            "decoder": "hefs/h10h/tiny.en/tiny_en-whisper-decoder-fixed-sequence.hef",
        },
    },
}


def get_hef_paths(variant: str, hw_arch: str) -> tuple:
    try:
        entry = HEF_REGISTRY[variant][hw_arch]
    except KeyError:
        raise FileNotFoundError(
            f"HEF not available for model '{variant}' on hardware '{hw_arch}'."
        )
    encoder_path = os.path.join(MODELS_DIR, entry["encoder"])
    decoder_path = os.path.join(MODELS_DIR, entry["decoder"])
    for path in (encoder_path, decoder_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    return encoder_path, decoder_path


class HailoWhisperPipeline:
    def __init__(self, encoder_model_path: str, decoder_model_path: str, variant="base", boost_words=None):
        self.encoder_model_path = encoder_model_path
        self.decoder_model_path = decoder_model_path
        self.timeout_ms = 100000000
        self.variant = variant
        self.decoding_sequence_length = None

        self.token_embedding_weight = self._load_token_embedding_weight()
        self.onnx_add_input = self._load_onnx_add_input()
        self.constant_output_0 = np.array([1])
        self._load_tokenizer()
        self.boost_token_map = self._build_boost_token_map(boost_words or {})

        encoder_hef = HEF(self.encoder_model_path)
        self.input_audio_length = int((encoder_hef.get_input_vstream_infos()[0].shape[1]) / 100)

        self.data_queue = Queue()
        self.results_queue = Queue()
        self.running = True
        self.thread = Thread(target=self._inference_loop)
        self.thread.start()

    def _load_token_embedding_weight(self):
        file_path = os.path.join(
            MODELS_DIR,
            f"decoder_assets/{self.variant}/decoder_tokenization/token_embedding_weight_{self.variant}.npy"
        )
        return np.load(file_path)

    def _load_onnx_add_input(self):
        file_path = os.path.join(
            MODELS_DIR,
            f"decoder_assets/{self.variant}/decoder_tokenization/onnx_add_input_{self.variant}.npy"
        )
        return np.load(file_path)

    def _load_tokenizer(self):
        # Load from HuggingFace cache without network access to avoid timeout
        # errors on flaky connections. The tokenizer must have been downloaded
        # once before (e.g. via: python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('openai/whisper-base')")
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"openai/whisper-{self.variant}", local_files_only=True
        )

    def _build_boost_token_map(self, boost_words):
        token_map = {}
        for word, factor in boost_words.items():
            token_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for tid in token_ids:
                token_map[tid] = factor
        return token_map

    def _tokenization(self, decoder_input_ids, add_embed=True):
        gather_output = self.token_embedding_weight[decoder_input_ids]
        if add_embed:
            add_output = gather_output + self.onnx_add_input
            unsqueeze_output = np.expand_dims(add_output, axis=int(self.constant_output_0[0]))
            transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))
            return transpose_output
        else:
            unsqueeze_output = np.expand_dims(gather_output, axis=0)
            return unsqueeze_output

    def _inference_loop(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"

        decoder_hef = HEF(self.decoder_model_path)
        sorted_output_names = decoder_hef.get_sorted_output_names()
        decoder_model_name = decoder_hef.get_network_group_names()[0]
        self.decoding_sequence_length = decoder_hef.get_output_vstream_infos()[0].shape[1]

        with VDevice(params) as vdevice:
            encoder_infer_model = vdevice.create_infer_model(self.encoder_model_path)
            decoder_infer_model = vdevice.create_infer_model(self.decoder_model_path)
            encoder_infer_model.input().set_format_type(FormatType.FLOAT32)
            encoder_infer_model.output().set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer1").set_format_type(FormatType.FLOAT32)
            decoder_infer_model.input(f"{decoder_model_name}/input_layer2").set_format_type(FormatType.FLOAT32)

            for output_name in sorted_output_names:
                decoder_infer_model.output(output_name).set_format_type(FormatType.FLOAT32)

            useful_outputs = [name for name in sorted_output_names if "conv" in name]

            with encoder_infer_model.configure() as encoder_configured_infer_model:
                with decoder_infer_model.configure() as decoder_configured_infer_model:
                    encoder_bindings = encoder_configured_infer_model.create_bindings()
                    decoder_bindings = decoder_configured_infer_model.create_bindings()

                    while self.running:
                        try:
                            input_mel = self.data_queue.get(timeout=1)

                            input_mel = np.ascontiguousarray(input_mel)
                            encoder_bindings.input().set_buffer(input_mel)
                            buffer = np.zeros(encoder_infer_model.output().shape).astype(np.float32)
                            encoder_bindings.output().set_buffer(buffer)

                            encoder_configured_infer_model.run([encoder_bindings], self.timeout_ms)
                            encoded_features = encoder_bindings.output().get_buffer()

                            start_token_id = [50258]
                            decoder_input_ids = np.array(
                                [[start_token_id[0]]], dtype=np.int64
                            )
                            decoder_input_ids = np.concatenate(
                                [decoder_input_ids, np.zeros((1, self.decoding_sequence_length - 1), dtype=np.int64)],
                                axis=1
                            )

                            generated_tokens = []

                            for i in range(self.decoding_sequence_length - 1):
                                tokenized_ids = self._tokenization(decoder_input_ids, add_embed=False)

                                decoder_bindings.input(f"{decoder_model_name}/input_layer1").set_buffer(encoded_features)
                                decoder_bindings.input(f"{decoder_model_name}/input_layer2").set_buffer(tokenized_ids)

                                buffers = [
                                    np.zeros(decoder_infer_model.output(name).shape).astype(np.float32)
                                    for name in sorted_output_names
                                ]
                                for name, buf in zip(sorted_output_names, buffers):
                                    decoder_bindings.output(name).set_buffer(buf)

                                decoder_configured_infer_model.run([decoder_bindings], self.timeout_ms)

                                decoder_outputs = np.concatenate(
                                    [decoder_bindings.output(name).get_buffer() for name in useful_outputs],
                                    axis=2
                                )

                                repetition_penalty = 1.5
                                logits = apply_repetition_penalty(
                                    decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty
                                )
                                if self.boost_token_map:
                                    logits = apply_word_boost(logits, self.boost_token_map)
                                next_token = np.argmax(logits)

                                generated_tokens.append(next_token)
                                decoder_input_ids[0][i + 1] = np.array([[next_token]], dtype=np.int64)

                                if next_token == self.tokenizer.eos_token_id:
                                    break

                            transcription = self.tokenizer.decode(
                                generated_tokens, skip_special_tokens=True
                            )
                            self.results_queue.put(transcription)
                        except Empty:
                            pass

    def get_model_input_audio_length(self):
        return self.input_audio_length

    def send_data(self, data):
        self.data_queue.put(data)

    def get_transcription(self):
        return self.results_queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
