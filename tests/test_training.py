import random, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from qwen3_tn.data import CausalLMCollator, TokenBlockDataset
from qwen3_tn.model import install_tt_modules
from qwen3_tn.training import (
    TTFineTuneConfig,
    TTTrainingState,
    finetune_causal_lm,
    freeze_except_tt,
    load_training_checkpoint,
    save_training_checkpoint,
)
from helpers import TinyCausalLM, TinyTokenizer, make_module_set


class TrainingTests(unittest.TestCase):
    def test_one_step_updates_only_tt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyCausalLM()
            index = make_module_set(root / "source", model)
            install_tt_modules(model, index, trainable=True, core_dtype=torch.float32)
            dense = {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if "cores" not in name
            }
            cores = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            dataset = TokenBlockDataset([[1, 2, 3], [4, 5]])
            loader = DataLoader(
                dataset, batch_size=1, collate_fn=CausalLMCollator(TinyTokenizer())
            )
            config = TTFineTuneConfig(
                max_steps=1,
                gradient_accumulation_steps=1,
                device="cpu",
                bf16_autocast=False,
                logging_steps=1,
                save_steps=0,
            )
            metrics = finetune_causal_lm(
                model, loader, config, root / "run", model_path="tiny"
            )
            self.assertEqual(metrics["global_step"], 1)
            self.assertTrue(
                any(
                    not torch.equal(old, new)
                    for old, new in zip(
                        cores,
                        [p for p in model.parameters() if p.requires_grad],
                        strict=True,
                    )
                )
            )
            self.assertTrue(
                all(
                    torch.equal(old, dict(model.named_parameters())[name])
                    for name, old in dense.items()
                )
            )

    def test_checkpoint_restores_rng_and_state(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = TinyCausalLM()
            index = make_module_set(root / "source", model)
            install_tt_modules(model, index, trainable=True, core_dtype=torch.float32)
            freeze_except_tt(model)
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad])
            scheduler = LambdaLR(optimizer, lambda _: 1.0)
            config = TTFineTuneConfig(device="cpu", bf16_autocast=False)
            random.seed(3)
            torch.manual_seed(3)
            save_training_checkpoint(
                root / "ckpt",
                model,
                optimizer,
                scheduler,
                TTTrainingState(2, 1, 4),
                model_path="tiny",
                config=config,
            )
            expected = (random.random(), torch.rand(1))
            random.seed(99)
            torch.manual_seed(99)
            state = load_training_checkpoint(
                root / "ckpt", model, optimizer, scheduler, expected_config=config
            )
            self.assertEqual(state.global_step, 2)
            self.assertEqual(random.random(), expected[0])
            torch.testing.assert_close(torch.rand(1), expected[1])
