from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Literal, Optional

import jax
from flax import nnx
from huggingface_hub import snapshot_download
from tunix.models import safetensors_loader as safetensors_loader_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

from .. import logger
from ..config import Settings
from ..hf_export import export_gemma3_lora_merged_hf_safetensors
from .base import CheckpointModelSource, GemmaEarth


class HuggingFaceCheckpointTrainer(GemmaEarth):
    """Trainer that uses Hugging Face safetensors checkpoints."""

    CHECKPOINT_SOURCE: CheckpointModelSource = "huggingface"


    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_config: Optional[gemma3_model_lib.ModelConfig] = None,
        restore_policy: Literal["strict", "permissive"] = "permissive",
    ) -> None:
        """Initialize Hugging Face trainer state.

        Args:
            settings: Optional settings object; defaults to environment-loaded
                settings when omitted.
            model_config: Optional explicit Gemma model config override.
            restore_policy: Checkpoint restore behavior for eval.

        Returns:
            None.
        """
        super().__init__(
            settings=settings,
            model_config=model_config,
            restore_policy=restore_policy,
        )
        self.hf_checkpoint_local_path: str | None = None

    @staticmethod
    def _detect_vocab_size(model_dir: Path) -> int | None:
        """Infer vocab size from a local HF safetensors checkpoint directory.

        Args:
            model_dir: Local directory containing `*.safetensors` files.

        Returns:
            Detected embedding vocab size, or ``None`` when unavailable.
        """
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            return None
        with open(safetensor_files[0], "rb") as fp:
            header_size = struct.unpack("<Q", fp.read(8))[0]
            header = json.loads(fp.read(header_size).decode("utf-8"))
        for key in ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight"):
            if key in header:
                shape = header[key].get("shape", ())
                if len(shape) == 2:
                    return int(shape[0])
        return None

    @staticmethod
    def _patched_key_mapping(cfg: gemma3_model_lib.ModelConfig):
        """Patch mapping for HF vision position embedding tensor layout.

        Args:
            cfg: Gemma3 model config used to derive patch-grid shape.

        Returns:
            Adjusted safetensors key mapping function output.
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

    def _compat_spec(self) -> dict[str, Any]:
        """Extend base compatibility spec with HF-specific checkpoint fields.

        Args:
            None.

        Returns:
            Compatibility spec including HF model id and ignore patterns.
        """
        spec = super()._compat_spec()
        spec["hf_model_id"] = self.settings.hf_model_id
        spec["hf_ignore_patterns"] = self.settings.hf_ignore_patterns
        return spec

    def load_base_model(self, mesh: jax.sharding.Mesh) -> None:
        """Load model weights from HF safetensors.

        Args:
            mesh: JAX mesh used for model construction and sharding.

        Returns:
            None. Stores the loaded model in ``self.base_model``.
        """
        ignore_patterns = [
            p.strip() for p in self.settings.hf_ignore_patterns.split(",") if p.strip()
        ] or None
        self.hf_checkpoint_local_path = snapshot_download(
            repo_id=self.settings.hf_model_id,
            ignore_patterns=ignore_patterns,
        )
        logger.info(
            "Loading model config %s with HF checkpoint %s at %s...",
            self.model_config.__class__.__qualname__,
            self.settings.hf_model_id,
            self.hf_checkpoint_local_path,
        )

        ckpt_vocab_size = self._detect_vocab_size(Path(self.hf_checkpoint_local_path))
        if ckpt_vocab_size is not None and ckpt_vocab_size != self.model_config.num_embed:
            logger.warning(
                "Adjusting model_config.num_embed from %d to %d to match checkpoint.",
                self.model_config.num_embed,
                ckpt_vocab_size,
            )
            self.model_config.num_embed = ckpt_vocab_size

        with mesh:
            self.base_model = safetensors_loader_lib.load_and_create_model(
                file_dir=self.hf_checkpoint_local_path,
                model_class=gemma3_model_lib.Gemma3,
                config=self.model_config,
                key_mapping=self._patched_key_mapping,
                mesh=mesh,
                preprocess_fn=params_safetensors_lib._make_preprocess_fn(self.model_config),
                mode="original",
            )

    def _post_train(self, lora_model: nnx.Module) -> None:
        """Export merged HF model after LoRA training.

        Args:
            lora_model: Trained LoRA-augmented model to merge/export.

        Returns:
            None.

        Raises:
            RuntimeError: If no local HF checkpoint path is available.
        """
        if self.hf_checkpoint_local_path is None:
            raise RuntimeError("HF checkpoint path is not available for export.")
        logger.info("HF checkpoint path: %s", self.hf_checkpoint_local_path)
        export_dir = str(self._experiment_root() / "hf_safetensors")
        logger.info("HF merged safetensors output dir: %s", export_dir)
        export_gemma3_lora_merged_hf_safetensors(
            local_model_path=self.hf_checkpoint_local_path,
            output_dir=str(export_dir),
            lora_model=lora_model,
            rank=self.settings.lora_rank,
            alpha=self.settings.lora_alpha,
        )
