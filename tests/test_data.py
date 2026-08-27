import json, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from qwen3_tn.data import JsonlDocumentSource, prepare_causal_lm_data
from helpers import TinyTokenizer


class DataTests(unittest.TestCase):
    def test_jsonl_eos_blocks_padding_and_fingerprint(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text(
                json.dumps({"text": "ab"}) + "\n" + json.dumps({"text": "c"}) + "\n"
            )
            source = JsonlDocumentSource(path)
            prepared = prepare_causal_lm_data(
                source, TinyTokenizer(), max_length=3, batch_size=2, seed=4
            )
            batch = next(iter(prepared.dataloader))
            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertTrue(
                (batch["labels"][batch["attention_mask"] == 0] == -100).all()
            )
            self.assertTrue(prepared.fingerprint)

    def test_bad_record(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text('{"wrong": 1}\n')
            with self.assertRaises(ValueError):
                list(JsonlDocumentSource(path).documents())
