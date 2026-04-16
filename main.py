import argparse
import json
import os
import tempfile
from typing import Literal

from datasets import load_from_disk

from gemma_earth import create_trainer
from gemma_earth.config import Settings
from gemma_earth.dataset import EarthDialDataset


# Keep temporary files and Hugging Face cache on the mounted data disk.
# This avoids out-of-disk-space issues on the smaller boot volume and preserves
# cache across VM restarts.
TMP_DIR = "/mnt/disks/data/tmp"
HF_DATASETS_CACHE = "/mnt/disks/data/.cache/huggingface/datasets"


def configure_runtime() -> None:
    """Configure temp/cache paths before importing heavy training modules."""
    os.makedirs(TMP_DIR, exist_ok=True)
    os.environ["TMPDIR"] = TMP_DIR
    os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
    tempfile.tempdir = TMP_DIR


def train(
    restore_policy: Literal["strict", "permissive"] = "permissive",
    model_checkpoint_source: Literal["tunix", "huggingface"] | None = None,
) -> None:
    """Run model training."""
    configure_runtime()

    gemma_earth = create_trainer(
        restore_policy=restore_policy,
        model_source=model_checkpoint_source,
    )
    gemma_earth.train()


def eval(
    start_index: int = 200,
    num_examples: int = 30,
    restore_policy: Literal["strict", "permissive"] = "permissive",
    model_checkpoint_source: Literal["tunix", "huggingface"] | None = None,
) -> None:
    """Run model evaluation and write JSON results."""
    configure_runtime()

    gemma_earth = create_trainer(
        restore_policy=restore_policy,
        model_source=model_checkpoint_source,
    )
    results = gemma_earth.eval(start_index=start_index, num_examples=num_examples)
    with open("results.json", "w") as fp:
        json.dump(results, fp, indent=4)


def dataset_info() -> None:
    """Print EarthDial remote size and local number of entries."""
    configure_runtime()

    settings = Settings()
    dataset = EarthDialDataset(settings=settings)

    size_bytes = dataset.get_dataset_size_bytes()
    if size_bytes >= 0:
        size_gb = size_bytes / (1024 ** 3)
        print(f"Remote dataset size: {size_gb:.2f} GB ({size_bytes} bytes)")
    else:
        print("Remote dataset size: unavailable")

    try:
        ds = load_from_disk(settings.dataset_dir)
        print(f"Local dataset entries: {len(ds)}")
    except Exception as exc:
        print(f"Local dataset entries: unavailable ({exc})")


def main() -> None:
    """Run training or evaluation from a simple CLI."""
    parser = argparse.ArgumentParser(description="Gemma Earth runner")
    parser.add_argument(
        "command",
        choices=["train", "eval", "dataset-info"],
        help="Whether to run training, evaluation, or dataset info.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=200,
        help="Starting dataset index for evaluation.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=30,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--train-restore-policy",
        choices=["strict", "permissive"],
        default="permissive",
        help="Checkpoint restore policy for train command.",
    )
    parser.add_argument(
        "--eval-restore-policy",
        choices=["strict", "permissive"],
        default="permissive",
        help="Checkpoint restore policy for eval command.",
    )
    parser.add_argument(
        "--model-checkpoint-source",
        choices=["tunix", "huggingface"],
        default=None,
        help="Optional base model checkpoint source override for train/eval.",
    )
    args = parser.parse_args()
    selected_model_source = args.model_checkpoint_source

    if args.command == "train":
        train(
            restore_policy=args.train_restore_policy,
            model_checkpoint_source=selected_model_source,
        )
        return

    if args.command == "dataset-info":
        dataset_info()
        return

    eval(
        start_index=args.start_index,
        num_examples=args.num_examples,
        restore_policy=args.eval_restore_policy,
        model_checkpoint_source=selected_model_source,
    )



if __name__ == "__main__":
    main()
