#!/usr/bin/env python3
"""Serve GemmaEarth single-image inference over FastAPI.

The model is loaded once at startup and reused for all requests.
"""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
from pathlib import Path
import struct
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import jax
from PIL import Image
import jax.numpy as jnp
import numpy as np
import uvicorn
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import safetensors_loader as safetensors_loader_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as params_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.processors import image_processor as image_processor_lib

from gemma_earth.config import Settings


@dataclass
class ServerConfig:
    model_dir: str
    model_checkpoint_source: str
    tokenizer_path: str | None
    max_prompt_length: int
    host: str
    port: int


class InferenceState:
    """Holds loaded inference objects shared by all requests."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.mesh = None
        self.model_config = None
        self.sampler = None


def _detect_vocab_size(model_dir: Path) -> int | None:
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


def _create_mesh() -> jax.sharding.Mesh:
    return jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def _build_prompt(user_prompt: str, include_image: bool) -> str:
    if include_image:
        user_payload = (
            "<start_of_image>"
            + "<img>" * 256
            + "<end_of_image>\n\n"
            + user_prompt.strip()
        )
    else:
        user_payload = user_prompt.strip()

    return (
        "<start_of_turn>user\n"
        + user_payload
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )


def _create_model(
    model_checkpoint_source: str,
    model_ref: str,
    mesh: jax.sharding.Mesh,
    model_config: gemma3_model_lib.ModelConfig,
) -> gemma3_model_lib.Gemma3:
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve GemmaEarth inference with FastAPI")
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Checkpoint location. For huggingface source: local dir with *.safetensors. "
            "For tunix source: checkpoint path (e.g. gs://... or local path)."
        ),
    )
    parser.add_argument(
        "--model-checkpoint-source",
        choices=["tunix", "huggingface"],
        default="tunix",
        help="Checkpoint source selector (default: tunix).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path override (default: Settings.gemma_tokenizer_path)",
    )
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI app and initialize model at startup."""
    state = InferenceState()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.mesh = _create_mesh()
        state.model_config = gemma3_model_lib.ModelConfig.gemma3_4b_it(text_only=False)

        model = _create_model(
            model_checkpoint_source=config.model_checkpoint_source,
            model_ref=config.model_dir,
            mesh=state.mesh,
            model_config=state.model_config,
        )

        settings = Settings()
        state.sampler = _create_sampler(
            model=model,
            model_config=state.model_config,
            tokenizer_path=config.tokenizer_path or settings.gemma_tokenizer_path,
            max_prompt_length=config.max_prompt_length,
        )
        state.ready = True
        yield

    app = FastAPI(
        title="GemmaEarth Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok" if state.ready else "loading",
            "model_checkpoint_source": config.model_checkpoint_source,
            "model_dir": config.model_dir,
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile | None = File(default=None),
        message: str = Form(...),
        max_generation_steps: int = Form(96),
        max_prompt_length: int = Form(config.max_prompt_length),
        temperature: float = Form(0.0),
    ) -> dict[str, object]:
        if not state.ready or state.sampler is None:
            raise HTTPException(status_code=503, detail="Model is still loading")

        pil_image: Image.Image | None = None
        if image is not None:
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Uploaded file must be an image")

            raw = await image.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")

            try:
                pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        formatted_prompt = _build_prompt(message, include_image=pil_image is not None)

        with state.lock:
            sampler_kwargs = {
                "input_strings": [formatted_prompt],
                "max_generation_steps": max_generation_steps,
                "max_prompt_length": max_prompt_length,
                "temperature": temperature,
            }
            if pil_image is not None:
                sampler_kwargs["images"] = [np.asarray(pil_image)]

            output = state.sampler(**sampler_kwargs)

        prediction = output.text[0].split("<end_of_turn>")[0].strip()
        return {
            "prediction": prediction,
            "message": message,
            "has_image": pil_image is not None,
        }

    return app


def main() -> None:
    args = _build_parser().parse_args()
    config = ServerConfig(
        model_dir=args.model_dir,
        model_checkpoint_source=args.model_checkpoint_source,
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        host=args.host,
        port=args.port,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
