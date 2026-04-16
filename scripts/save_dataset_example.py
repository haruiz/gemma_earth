#!/usr/bin/env python3
"""Create EarthDialDataset instance and save one sample image + prompt."""

from __future__ import annotations

import argparse
from pathlib import Path

from gemma_earth.dataset import EarthDialDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Save one dataset example image and prompt")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--output-dir", default=".", help="Directory to write files")
    parser.add_argument("--image-filename", default="sample_image.jpg", help="Output image filename")
    parser.add_argument("--prompt-filename", default="sample_prompt.txt", help="Output prompt filename")
    args = parser.parse_args()

    dataset = EarthDialDataset()
    image_path, prompt_path = dataset.save_example_image_and_prompt(
        index=args.index,
        output_dir=Path(args.output_dir),
        image_filename=args.image_filename,
        prompt_filename=args.prompt_filename,
    )

    print(f"Saved image: {image_path}")
    print(f"Saved prompt: {prompt_path}")


if __name__ == "__main__":
    main()
