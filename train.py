# Copyright 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Xiaomi-Lens-0 training entry point.

The maintained recipes are `scripts/train_stage1.sh` (mask decoder warmup) and
`scripts/train_stage2.sh` (full-parameter SFT). Direct usage:

    python -m torch.distributed.run --nproc_per_node=8 train.py \
        --deepspeed src/xiaomi_lens/train/configs/zero2.json \
        --model_name_or_path /path/to/Qwen3.5-4B \
        --data_config /path/to/data.yaml \
        --output_dir /path/to/output \
        --unfreeze_all_parameters true --bf16
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

# Make the src-layout package importable without `pip install -e .`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from accelerate.utils import set_seed  # noqa: E402
from transformers import HfArgumentParser  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from xiaomi_lens.train.config import (  # noqa: E402
    XiaomiLensModelConfig,
    XiaomiLensScriptArguments,
    XiaomiLensTrainingConfig,
)
from xiaomi_lens.train.data import load_dataset_config  # noqa: E402
from xiaomi_lens.train.preflight import build_preflight_report, is_eval_enabled  # noqa: E402
from xiaomi_lens.train.trainer import XiaomiLensTrainer  # noqa: E402


def _normalize_resume_from_checkpoint(value: Any) -> Any:
    if value in (None, False):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no"}:
            return None
        if normalized in {"1", "true", "yes"}:
            return True
    return value


def _cli_arg_passed(name: str) -> bool:
    """True if the flag ``name`` was given on the command line, any spelling.

    Used to let an explicit flag win over ``--auto_resume``'s inferred value
    without needing HfArgumentParser to report which fields were defaulted.
    HfArgumentParser accepts dash- and underscore-separated aliases
    interchangeably (``--resume-from-checkpoint`` == ``--resume_from_checkpoint``)
    and a trailing ``=value``, so match on the canonical (dash→underscore) token.
    """
    canonical = name.lstrip("-").replace("-", "_")
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            continue
        token = arg[2:].split("=", 1)[0].replace("-", "_")
        if token == canonical:
            return True
    return False


def _resolve_auto_resume(
    training_args: XiaomiLensTrainingConfig, resume_from_checkpoint: Any, log: bool = True
) -> Any:
    """Return the checkpoint to resume from when ``--auto_resume`` is set.

    Picks the highest-step ``checkpoint-*`` under ``output_dir`` (via HF's
    ``get_last_checkpoint``); returns ``None`` (start fresh) when none exists.
    A no-op when auto-resume is off or ``--resume_from_checkpoint`` was passed
    explicitly. Flips ``ignore_data_skip`` on for a real resume unless the user
    set it explicitly. Filesystem-only + deterministic, so it runs identically
    on every rank; ``log`` gates the message to rank 0.
    """
    if not training_args.auto_resume or _cli_arg_passed("--resume_from_checkpoint"):
        return resume_from_checkpoint

    output_dir = pathlib.Path(training_args.output_dir)
    last_checkpoint = get_last_checkpoint(output_dir) if output_dir.is_dir() else None
    if last_checkpoint is None:
        if log:
            print(f"[auto_resume] no checkpoint under {output_dir}; starting fresh.")
        return None

    if not _cli_arg_passed("--ignore_data_skip"):
        training_args.ignore_data_skip = True
    if log:
        print(
            f"[auto_resume] resuming from {last_checkpoint} "
            f"(ignore_data_skip={training_args.ignore_data_skip})."
        )
    return last_checkpoint


def _validate_resume_target(resume_from_checkpoint: Any, output_dir: str) -> None:
    if resume_from_checkpoint is True:
        ckpt_dir = pathlib.Path(output_dir)
        if not ckpt_dir.exists() or not any(ckpt_dir.glob("checkpoint-*")):
            raise ValueError(
                f"Requested automatic checkpoint resume, but no checkpoints were found under {ckpt_dir}."
            )
    elif isinstance(resume_from_checkpoint, str):
        checkpoint_path = pathlib.Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")


def main(
    script_args: XiaomiLensScriptArguments,
    training_args: XiaomiLensTrainingConfig,
    model_args: XiaomiLensModelConfig,
) -> None:
    set_seed(training_args.seed)

    # Dataset construction is owned by ``XiaomiLensTrainer.__init__``. It scans
    # lightweight JSONL offset indexes into memory and does preprocessing in
    # DataLoader workers. Training and evaluation use separate `train:` and
    # `eval:` sections in one manifest.
    if is_eval_enabled(training_args):
        if not script_args.data_config:
            raise ValueError(
                "--eval_strategy requires --data_config with explicit `train:` "
                "and `eval:` sections."
            )
        # Validate the split before loading the model so a train-only manifest
        # fails immediately rather than after a large load.
        load_dataset_config(script_args.data_config, split="eval")

    trainer = XiaomiLensTrainer(
        model=model_args.model_name_or_path,
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
    )

    if trainer.is_world_process_zero():
        # Preflight runs after trainer init: ``trainer.train_dataset`` is
        # the lazy JSONL-backed dataset built inside ``__init__``.
        splits = {"train": trainer.train_dataset}
        if trainer.eval_dataset is not None:
            splits["validation"] = trainer.eval_dataset
        trainer.accelerator.print(
            build_preflight_report(
                script_args,
                training_args,
                model_args,
                trainer,
                splits,
            )
        )

    resume_from_checkpoint = _normalize_resume_from_checkpoint(training_args.resume_from_checkpoint)
    resume_from_checkpoint = _resolve_auto_resume(
        training_args, resume_from_checkpoint, log=trainer.is_world_process_zero()
    )
    _validate_resume_target(resume_from_checkpoint, training_args.output_dir)

    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(
        (XiaomiLensScriptArguments, XiaomiLensTrainingConfig, XiaomiLensModelConfig)
    )
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    main(script_args, training_args, model_args)
