#!/usr/bin/env python3
"""Load a HF safetensors Gemma3 model and run single-image sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import safetensors_loader as safetensors_loader_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as params_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.processors import image_processor as image_processor_lib

from gemma_earth.config import Settings

DEFAULT_CLASS_PROMPT = (
    "Classify the given image in one of the following classes. "
    "Classes: non-irrigated arable land, dump sites, peatbogs, pastures, coniferous forest, "
    "agro-forestry areas, broad-leaved forest, sparsely vegetated areas, industrial or "
    "commercial units, airports, bare rock, vineyards, water courses, rice fields, salt marshes, "
    "sport and leisure facilities, sea and ocean, water bodies, inland marshes, annual crops "
    "associated with permanent crops, mixed forest, beaches, dunes, sands, complex cultivation "
    "patterns, road and rail networks and associated land, land principally occupied by "
    "agriculture, with significant areas of natural vegetation, moors and heathland, "
    "discontinuous urban fabric, continuous urban fabric, olive groves, intertidal flats, burnt "
    "areas, mineral extraction sites, permanently irrigated land, estuaries, green urban areas, "
    "construction sites, sclerophyllous vegetation, fruit trees and berry plantations, coastal "
    "lagoons, natural grassland, port areas, salines, transitional woodland/shrub. "
    "Answer in one word or a short phrase."
)


def _detect_vocab_size(model_dir: Path) -> int | None:
    """Infer vocab size from the first safetensors file header.

    Args:
        model_dir: Directory containing HF `*.safetensors` checkpoint shards.

    Returns:
        The detected vocab size from embedding weights, or `None` if unavailable.
    """
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    with open(files[0], "rb") as fp:
        header_size = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(header_size).decode("utf-8"))
    for key in ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight"):
        if key in header:
            shape = header[key].get("shape", ())
            if len(shape) == 2:
                return int(shape[0])
    return None


def _patched_key_mapping(cfg: gemma3_model_lib.ModelConfig):
    """Patch safetensors key mapping for vision position embeddings.

    Args:
        cfg: Gemma3 model config used to derive image patch dimensions.

    Returns:
        A key mapping compatible with current vision image/patch settings.
    """
    mapping = params_safetensors_lib._get_key_and_transform_mapping(cfg)
    if cfg.vision_config is None:
        return mapping

    pos_embed_key = r"vision_tower\.vision_model\.embeddings\.position_embedding\.weight"
    if pos_embed_key in mapping:
        num_patches = (
            cfg.vision_config.image_height // cfg.vision_config.patch_size[0]
        ) * (cfg.vision_config.image_width // cfg.vision_config.patch_size[1])
        mapped_name, _ = mapping[pos_embed_key]
        mapping[pos_embed_key] = (
            mapped_name,
            (None, (1, num_patches, cfg.vision_config.width)),
        )
    return mapping


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one-example evaluation.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Checkpoint location. For huggingface source: local dir with *.safetensors. "
            "For tunix source: checkpoint path (e.g. gs://... or local path)."
        ),
    )
    parser.add_argument("--image-path", required=True, help="Path to a single test image")
    parser.add_argument(
        "--prompt",
        default=DEFAULT_CLASS_PROMPT,
        help="User prompt text",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path override (default: Settings.gemma_tokenizer_path)",
    )
    parser.add_argument("--max-generation-steps", type=int, default=96)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--model-checkpoint-source",
        choices=["tunix", "huggingface"],
        default="tunix",
        help="Checkpoint source selector (default: tunix).",
    )
    return parser


def _create_mesh() -> jax.sharding.Mesh:
    """Create the minimal fsdp/tp mesh used for single-example inference.

    Args:
        None.

    Returns:
        JAX mesh with fsdp/tp axes.
    """
    return jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def _build_prompt(user_prompt: str) -> str:
    """Build one Gemma multimodal prompt from raw user text.

    Args:
        user_prompt: Raw user instruction text.

    Returns:
        Gemma-formatted prompt ending in model turn prefix.
    """
    return (
        "<start_of_turn>user\n"
        + "<start_of_image>"
        + "<img>" * 256
        + "<end_of_image>\n\n"
        + user_prompt.strip()
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )


def _create_model(
    model_checkpoint_source: str,
    model_ref: str,
    mesh: jax.sharding.Mesh,
    model_config: gemma3_model_lib.ModelConfig,
) -> gemma3_model_lib.Gemma3:
    """Create a Gemma model from the selected checkpoint source.

    Args:
        model_checkpoint_source: Source selector (`tunix` or `huggingface`).
        model_ref: Model checkpoint path or directory.
        mesh: JAX mesh used for model creation.
        model_config: Target Gemma model config.

    Returns:
        Loaded Gemma model ready for sampling.
    """
    if model_checkpoint_source == "huggingface":
        model_dir = Path(model_ref).expanduser().resolve()
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        vocab = _detect_vocab_size(model_dir)
        if vocab is not None and vocab != model_config.num_embed:
            model_config.num_embed = vocab
        with mesh:
            return safetensors_loader_lib.load_and_create_model(
                file_dir=str(model_dir),
                model_class=gemma3_model_lib.Gemma3,
                config=model_config,
                key_mapping=_patched_key_mapping,
                mesh=mesh,
                preprocess_fn=params_safetensors_lib._make_preprocess_fn(model_config),
                mode="original",
            )

    with mesh:
        return params_lib.create_model_from_checkpoint(
            model_ref,
            model_config,
            mesh,
            dtype=jnp.bfloat16,
        )


def _create_sampler(
    model: gemma3_model_lib.Gemma3,
    model_config: gemma3_model_lib.ModelConfig,
    tokenizer_path: str,
    max_prompt_length: int,
) -> sampler_lib.Sampler:
    """Create a sampler configured for one-example generation.

    Args:
        model: Loaded Gemma model.
        model_config: Gemma model configuration.
        tokenizer_path: Path to tokenizer model file.
        max_prompt_length: Maximum prompt length for cache sizing.

    Returns:
        Configured sampler object.
    """
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
    image_processor = image_processor_lib.ImageProcessor(config=model_config.vision_config)
    cache_config = sampler_lib.CacheConfig(
        cache_size=max(max_prompt_length + 128, 1024),
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    )
    return sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
        image_processor=image_processor,
    )


def _run_generation(
    sampler: sampler_lib.Sampler,
    image_path: Path,
    prompt: str,
    max_generation_steps: int,
    max_prompt_length: int,
    temperature: float,
) -> str:
    """Run generation for one image and prompt.

    Args:
        sampler: Configured sampler.
        image_path: Input image path.
        prompt: Formatted prompt string.
        max_generation_steps: Maximum generation steps.
        max_prompt_length: Maximum prompt length.
        temperature: Sampling temperature.

    Returns:
        Cleaned generated text.
    """
    image = Image.open(image_path).convert("RGB")
    output = sampler(
        input_strings=[prompt],
        images=[np.asarray(image)],
        max_generation_steps=max_generation_steps,
        max_prompt_length=max_prompt_length,
        temperature=temperature,
    )
    return output.text[0].split("<end_of_turn>")[0].strip()


def main() -> None:
    """Run single-image inference against Gemma3 Tunix or HF checkpoints."""
    args = _build_parser().parse_args()

    image_path = Path(args.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    mesh = _create_mesh()

    model_config = gemma3_model_lib.ModelConfig.gemma3_4b_it(text_only=False)
    model = _create_model(
        model_checkpoint_source=args.model_checkpoint_source,
        model_ref=args.model_dir,
        mesh=mesh,
        model_config=model_config,
    )
    settings = Settings()
    sampler = _create_sampler(
        model=model,
        model_config=model_config,
        tokenizer_path=args.tokenizer_path or settings.gemma_tokenizer_path,
        max_prompt_length=args.max_prompt_length,
    )
    text = _run_generation(
        sampler=sampler,
        image_path=image_path,
        prompt=_build_prompt(args.prompt),
        max_generation_steps=args.max_generation_steps,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
    )
    print(text)


if __name__ == "__main__":
    main()
