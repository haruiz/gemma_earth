from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as jnp
from tunix.models.gemma3 import params as params_lib
from tunix.models.gemma3 import model as gemma3_model_lib

from .. import logger
from ..config import Settings
from .base import CheckpointModelSource, GemmaEarth


class TunixCheckpointTrainer(GemmaEarth):
    """Trainer that uses Tunix checkpoints and supports permissive eval restore."""

    CHECKPOINT_SOURCE: CheckpointModelSource = "tunix"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_config: Optional[gemma3_model_lib.ModelConfig] = None,
        restore_policy: Literal["strict", "permissive"] = "permissive",
    ) -> None:
        """Initialize Tunix permissiver state.

        Args:
            settings: Optional settings object; defaults to environment-loaded
                settings when omitted.
            model_config: Optional explicit Gemma model config override.
            restore_policy: Checkpoint restore policy used for evaluation.

        Returns:
            None.
        """
        super().__init__(
            settings=settings,
            model_config=model_config,
            restore_policy=restore_policy,
        )

    def load_base_model(self, mesh: jax.sharding.Mesh) -> None:
        """Load the Tunix checkpoint into a sharded Gemma model.

        Args:
            mesh: JAX mesh used for checkpoint restore/model sharding.

        Returns:
            None. Stores the loaded model in ``self.base_model``.
        """
        logger.info(
            "Loading model config %s with Tunix checkpoint from %s...",
            self.model_config.__class__.__qualname__,
            self.settings.model_ckpt_path,
        )
        with mesh:
            self.base_model = params_lib.create_model_from_checkpoint(
                self.settings.model_ckpt_path,
                self.model_config,
                mesh,
                dtype=jnp.bfloat16,
            )

