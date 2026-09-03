import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from qwen3_tn.model_loading import (
    load_local_causal_lm,
    load_local_tokenizer,
    parse_torch_dtype,
)


class ModelLoadingTests(unittest.TestCase):
    def test_parse_torch_dtype_rejects_unknown_name(self):
        self.assertIs(parse_torch_dtype("bfloat16"), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            parse_torch_dtype("int8")

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_local_model_defaults_and_device_are_centralized(self, from_pretrained):
        model = Mock()
        moved = Mock()
        model.to.return_value = moved
        from_pretrained.return_value = model

        result = load_local_causal_lm(
            Path("model"), dtype=torch.float32, device="cpu"
        )

        self.assertIs(result, moved)
        from_pretrained.assert_called_once_with(
            Path("model"),
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        model.to.assert_called_once_with("cpu")

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_tokenizer_can_ensure_padding(self, from_pretrained):
        tokenizer = SimpleNamespace(
            pad_token_id=None, pad_token=None, eos_token="<eos>"
        )
        from_pretrained.return_value = tokenizer

        result = load_local_tokenizer(Path("model"), ensure_padding=True)

        self.assertIs(result, tokenizer)
        self.assertEqual(tokenizer.pad_token, "<eos>")
        from_pretrained.assert_called_once_with(
            Path("model"), local_files_only=True
        )


if __name__ == "__main__":
    unittest.main()
