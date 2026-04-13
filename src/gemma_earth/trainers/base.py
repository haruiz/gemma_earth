from collections.abc import Callable
from abc import ABC, abstractmethod
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Optional
import json
import os
import shutil

from .. import logger
from ..config import Settings
from ..dataset import EarthDialDataset

from datasets import load_from_disk
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
import qwix
from flax import nnx
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.processors import image_processor as image_processor_lib
from tunix.sft import checkpoint_manager as sft_checkpoint_manager
from tunix.sft import metrics_logger, peft_trainer, utils

CheckpointModelSource = Literal["tunix", "huggingface"]


class GemmaEarth(ABC):
    """Encapsulates Gemma model setup, LoRA wrapping, and training execution."""
    CHECKPOINT_SOURCE: CheckpointModelSource = "tunix"

    LORA_MODULE_PATH = (
        r".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|"
        r".*up_proj|.*query_proj|.*key_proj|.*value_proj|.*out_proj|.*fc1|.*fc2"
    )

    def __init__(
        self,
        settings: Settings | None = None,
        model_config: Optional[gemma3_model_lib.ModelConfig] = None,
        restore_policy: Literal["strict", "permissive"] = "permissive",
    ) -> None:
        """Create trainer state with settings, dataset, and selected model config.

        Args:
            settings: Optional settings object. Uses environment-loaded defaults
                when omitted.
            model_config: Optional explicit ModelConfig override.
            restore_policy: Checkpoint restore behavior for eval.

        Returns:
            None.
        """
        self.settings = settings or Settings()
        self.dataset = EarthDialDataset(settings=self.settings)
        self.model_config = (
            gemma3_model_lib.ModelConfig.gemma3_4b_it(text_only=False)
            if model_config is None
            else model_config
        )
        self.base_model: nnx.Module | None = None
        self.restore_policy = restore_policy

    def _pkg_version(self, package_name: str) -> str:
        """Return the installed package version or unknown when unavailable.

        Args:
            package_name: Distribution name to look up via importlib metadata.

        Returns:
            Installed package version string, or ``"unknown"`` if not found.
        """
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "unknown"

    def _parameter_spec(self) -> dict[str, Any]:
        """Return training/model parameters used for experiment identity.

        Returns:
            Parameter-only compatibility fields.
        """
        settings = self.settings
        return {
            "base_model_source": self.CHECKPOINT_SOURCE,
            "lora_rank": settings.lora_rank,
            "lora_alpha": settings.lora_alpha,
            "lora_module_path": self.LORA_MODULE_PATH,
            "max_seq_length": settings.max_seq_length,
        }

    def _path_spec(self) -> dict[str, Any]:
        """Return path-like compatibility fields (not used for experiment ID)."""
        settings = self.settings
        return {
            "model_ckpt_path": settings.model_ckpt_path,
            "tokenizer_path": settings.gemma_tokenizer_path,
        }

    def _package_version_spec(self) -> dict[str, Any]:
        """Return runtime package versions for reproducibility metadata."""
        return {
            "tunix_version": self._pkg_version("tunix"),
            "flax_version": self._pkg_version("flax"),
            "orbax_version": self._pkg_version("orbax-checkpoint"),
        }

    def _compat_spec(self) -> dict[str, Any]:
        """Build a compatibility signature for checkpoint reuse decisions.

        Args:
            None.

        Returns:
            Dict fingerprinting model/checkpoint/training compatibility inputs.
        """
        return {
            "parameters": self._parameter_spec(),
            "paths": self._path_spec(),
            "package_versions": self._package_version_spec(),
        }

    def _experiment_id_spec(self) -> dict[str, Any]:
        """Build the subset used for experiment ID hash.

        Returns:
            Parameter-only spec used to generate stable experiment IDs.
        """
        return self._parameter_spec().copy()

    def _experiment_id(self) -> str:
        """Create a stable short hash for the current compatibility spec.

        Args:
            None.

        Returns:
            Stable 12-character experiment identifier.
        """
        override = self.settings.experiment_id_override
        if override is not None and override.strip():
            return override.strip()

        payload = json.dumps(self._experiment_id_spec(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _checkpoint_root(self) -> str:
        """Resolve the experiment-specific checkpoint directory.

        Args:
            None.

        Returns:
            Absolute checkpoint directory path for the current compatibility spec.
        """
        return str(self._experiment_root() / "checkpoints")

    def _experiment_root(self) -> Path:
        """Resolve the root directory for one experiment.

        Args:
            None.

        Returns:
            Path to experiment root directory.
        """
        return Path(self.settings.experiments_dir) / self._experiment_id()

    def _tensorboard_root(self) -> str:
        """Resolve the experiment-specific tensorboard log directory.

        Args:
            None.

        Returns:
            Absolute tensorboard log directory path for the current experiment.
        """
        return str(self._experiment_root() / "tensorboard")


    def _compat_manifest_path(self) -> Path:
        """Return path to compatibility manifest for this experiment.

        Args:
            None.

        Returns:
            Filesystem path to ``compat.json`` under this experiment root.
        """
        return self._experiment_root() / "compat.json"

    def _write_compat_manifest(self) -> None:
        """Persist compatibility metadata next to checkpoints.

        Args:
            None.

        Returns:
            None.
        """
        root = self._experiment_root()
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment_id": self._experiment_id(),
            "compat_spec": self._compat_spec(),
        }
        self._compat_manifest_path().write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _read_compat_manifest(self) -> dict[str, Any] | None:
        """Read compatibility manifest when present.

        Args:
            None.

        Returns:
            Parsed manifest dict when present, otherwise ``None``.
        """
        path = self._compat_manifest_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_compatible_or_raise(self) -> None:
        """Validate checkpoint compatibility for strict restore policy.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If checkpoint metadata does not match current config.
        """
        manifest = self._read_compat_manifest()
        if manifest is None:
            return

        expected = self._experiment_id()
        found = manifest.get("experiment_id")
        if expected == found:
            return

        raise RuntimeError(
            "Checkpoint compatibility mismatch. "
            f"Expected experiment_id={expected}, found experiment_id={found}. "
            f"Checkpoint root: {self._checkpoint_root()}"
        )

    def _setup_dirs(self) -> None:
        """Create output folders and optionally clean stale checkpoints.

        Removes the checkpoint directory when clean_start is enabled, then
        ensures output, tensorboard, and dataset download directories exist.

        Returns:
            None.
        """
        settings = self.settings
        experiment_root = self._experiment_root()
        if settings.clean_start and experiment_root.exists():
            shutil.rmtree(experiment_root)
            logger.info("Cleaned existing experiment data in %s", experiment_root)

        os.makedirs(settings.output_dir, exist_ok=True)
        os.makedirs(self._checkpoint_root(), exist_ok=True)
        os.makedirs(self._tensorboard_root(), exist_ok=True)
        Path(settings.dataset_download_dir).mkdir(parents=True, exist_ok=True)


    def create_mesh(self) -> jax.sharding.Mesh:
        """Create a JAX mesh based on available local devices.

        Axis names and meaning:
            data: Data parallel axis. Different micro-batches are processed on
                different replicas.
            fsdp: Fully-sharded data parallel axis. Model and optimizer state
                are partitioned across devices.
            tp: Tensor parallel axis. Tensor operations inside layers are split
                across devices.

        Returns:
            A mesh over axes data, fsdp, and tp for 8+ devices, or fsdp/tp for
            smaller device counts.
        """
        num_devices = jax.local_device_count()
        logger.info("Detected %d local devices for JAX mesh creation.", num_devices)

        if num_devices >= 8:
            return jax.make_mesh(
                (1, 4, 2),
                ("data", "fsdp", "tp"),
                axis_types=(jax.sharding.AxisType.Auto,) * 3,
            )

        if num_devices >= 2:
            return jax.make_mesh(
                (num_devices, 1),
                ("fsdp", "tp"),
                axis_types=(jax.sharding.AxisType.Auto,) * 2,
            )

        return jax.make_mesh(
            (1, 1),
            ("fsdp", "tp"),
            axis_types=(jax.sharding.AxisType.Auto,) * 2,
        )

    @abstractmethod
    def load_base_model(self, mesh: jax.sharding.Mesh) -> None:
        """Load the selected base checkpoint into a sharded model instance.

        Args:
            mesh: JAX mesh used during model creation and checkpoint restore.

        Returns:
            None. Implementations must populate ``self.base_model``.
        """

    def build_lora_model(self, mesh: jax.sharding.Mesh) -> nnx.Module:
        """Apply LoRA adapters to the loaded base model and constrain sharding.

        Args:
            mesh: Mesh used for sharding constraints.

        Returns:
            LoRA-augmented model.

        Raises:
            RuntimeError: If ``self.base_model`` has not been initialized.
        """
        if self.base_model is None:
            raise RuntimeError("Base model was not initialized by load_base_model().")

        settings = self.settings
        lora_provider = qwix.LoraProvider(
            module_path=self.LORA_MODULE_PATH,
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
        )

        model_input = self.base_model.get_model_input()
        lora_model = qwix.apply_lora_to_model(self.base_model, lora_provider, **model_input)

        with mesh:
            state = nnx.state(lora_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(lora_model, sharded_state)

        return lora_model

    def _build_optimizer(self, max_steps: int) -> optax.GradientTransformation:
        """Build AdamW optimizer with warmup-cosine schedule.

        Args:
            max_steps: Total number of optimization steps used to shape
                warmup and cosine decay schedules.

        Returns:
            Configured optax GradientTransformation instance.
        """
        settings = self.settings
        warmup_steps = max(1, int(max_steps * settings.warmup_ratio))
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=settings.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max(1, max_steps),
            end_value=0.0,
        )

        return optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=settings.weight_decay,
        )

    def _build_trainer(
        self,
        lora_model: nnx.Module,
        optimizer: optax.GradientTransformation,
        max_steps: int,
        tokenizer: tokenizer_lib.Tokenizer,
    ) -> peft_trainer.PeftTrainer:
        """Build PeftTrainer (including training config) and attach input callback.

        Args:
            lora_model: LoRA-augmented model to optimize.
            optimizer: Optimizer instance used during training.
            max_steps: Maximum training steps used to shape training config.
            tokenizer: Tokenizer used when building model-input callback.

        Returns:
            Configured ``peft_trainer.PeftTrainer`` instance.
        """
        settings = self.settings
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=settings.save_interval_steps,
            max_to_keep=settings.max_to_keep,
        )
        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir=self._tensorboard_root(),
            flush_every_n_steps=20,
        )
        training_config = peft_trainer.TrainingConfig(
            eval_every_n_steps=settings.eval_every_n_steps,
            max_steps=max_steps,
            checkpoint_root_directory=self._checkpoint_root(),
            checkpointing_options=checkpointing_options,
            metrics_logging_options=metrics_logging_options,
        )
        trainer = peft_trainer.PeftTrainer(
            model=lora_model,
            optimizer=optimizer,
            training_config=training_config,
        )
        return trainer.with_gen_model_input_fn(self._gen_model_input_fn(tokenizer))

    def _build_tokenizer(self) -> tokenizer_lib.Tokenizer:
        """Create the tokenizer instance used by training and evaluation.

        Args:
            None.

        Returns:
            Initialized tokenizer using ``settings.gemma_tokenizer_path``.
        """
        return tokenizer_lib.Tokenizer(tokenizer_path=self.settings.gemma_tokenizer_path)

    def _build_image_processor(self) -> image_processor_lib.ImageProcessor:
        """Create the image processor instance used by training and evaluation.

        Args:
            None.

        Returns:
            Initialized image processor using current vision model config.
        """
        return image_processor_lib.ImageProcessor(config=self.model_config.vision_config)

    def _build_sampler(
        self,
        lora_model: nnx.Module,
        tokenizer: tokenizer_lib.Tokenizer,
        image_processor: image_processor_lib.ImageProcessor,
    ) -> sampler_lib.Sampler:
        """Create the sampler used for autoregressive generation.

        Args:
            lora_model: LoRA-augmented model used as sampler transformer.
            tokenizer: Tokenizer used to encode/decode text tokens.
            image_processor: Vision preprocessor for multimodal image inputs.

        Returns:
            Configured ``sampler_lib.Sampler`` instance.
        """
        cache_config = sampler_lib.CacheConfig(
            cache_size=max(self.settings.max_seq_length + 128, 1024),
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
        )
        return sampler_lib.Sampler(
            transformer=lora_model,
            tokenizer=tokenizer,
            cache_config=cache_config,
            image_processor=image_processor,
        )

    def _gen_model_input_fn(
        self,
        tokenizer: tokenizer_lib.Tokenizer,
    ) -> Callable[[peft_trainer.TrainingInput], dict[str, Any]]:
        """Build callback that maps TrainingInput batches to model kwargs.

        Args:
            tokenizer: Tokenizer used to detect padding when building masks,
                positions, and causal attention masks.

        Returns:
            Callable that converts one TrainingInput batch into model keyword
            arguments expected by the Gemma forward pass.
        """

        def gen_model_input_fn(x: peft_trainer.TrainingInput) -> dict[str, Any]:
            pad_mask = x.input_tokens != tokenizer.pad_id()
            positions = utils.build_positions_from_mask(pad_mask)
            attention_mask = utils.make_causal_attn_mask(pad_mask)
            return {
                "input_tokens": x.input_tokens,
                "input_mask": x.input_mask,
                "positions": positions,
                "attention_mask": attention_mask,
                "images": x.images,
            }

        return gen_model_input_fn

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove trailing generation markers from model output text.

        Args:
            text: Raw generated model output string.

        Returns:
            Output truncated before turn terminator and stripped.
        """
        if not isinstance(text, str):
            return ""
        return text.split("<end_of_turn>")[0].strip()

    def _restore_latest_checkpoint(self, model: nnx.Module) -> int:
        """Restore latest checkpoint from output checkpoint directory.

        Args:
            model: LoRA model instance to restore into.

        Returns:
            Restored checkpoint step.
        """
        self._assert_compatible_or_raise()

        if self.restore_policy == "permissive":
            restored_step = self._partial_restore(model=model, step=None)
            if restored_step == 0:
                raise RuntimeError(f"No checkpoint found under {self._checkpoint_root()}")
            return restored_step

        manager = sft_checkpoint_manager.CheckpointManager(root_directory=self._checkpoint_root())
        try:
            restored_step, _ = manager.maybe_restore(model=model, step=None)
        except Exception as exc:
            raise RuntimeError(
                "Strict checkpoint restore failed. "
                "If your model/checkpoint trees differ, use restore_policy='permissive' "
                "to restore only matching parameters."
            ) from exc
        finally:
            manager.close()

        if restored_step == 0:
            raise RuntimeError(f"No checkpoint found under {self._checkpoint_root()}")
        return restored_step

    def _partial_restore(self, model: nnx.Module, step: int | None) -> int:
        """Restore available checkpoint parameters while ignoring missing targets.

        Args:
            model: LoRA model instance to restore into.
            step: Optional explicit checkpoint step; latest when ``None``.

        Returns:
            Restored checkpoint step.

        Raises:
            RuntimeError: If no checkpoint exists in current checkpoint root.
        """
        manager = ocp.CheckpointManager(
            self._checkpoint_root(),
            item_handlers={"model_params": ocp.PyTreeCheckpointHandler()},
        )
        try:
            resolved_step = step if step is not None else manager.latest_step()
            if resolved_step is None:
                raise RuntimeError(f"No checkpoint found under {self._checkpoint_root()}")

            abstract_state = nnx.state(model)
            restore_args = ocp.checkpoint_utils.construct_restore_args(target=abstract_state)

            checkpoint = manager.restore(
                resolved_step,
                args=ocp.args.Composite(
                    model_params=ocp.args.PyTreeRestore(
                        item=abstract_state,
                        restore_args=restore_args,
                        partial_restore=True,
                    )
                ),
            )

            nnx.update(model, checkpoint.model_params)
            return resolved_step
        finally:
            manager.close()

    def eval(self, start_index: int, num_examples: int) -> list[dict[str, Any]]:
        """Run inference for a range of examples using the latest output checkpoint.

        Args:
            start_index: First dataset index to evaluate.
            num_examples: Number of sequential samples to evaluate.

        Returns:
            List of dict results containing index, prompt, ground_truth, and output.
        """
        if num_examples <= 0:
            raise ValueError("num_examples must be greater than 0")

        mesh = self.create_mesh()
        self.load_base_model(mesh)
        lora_model = self.build_lora_model(mesh=mesh)
        restored_step = self._restore_latest_checkpoint(lora_model)
        logger.info("Loaded checkpoint step: %d", restored_step)

        tokenizer = self._build_tokenizer()
        image_processor = self._build_image_processor()
        sampler = self._build_sampler(
            lora_model=lora_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
        )

        ds = load_from_disk(self.settings.dataset_dir)
        results: list[dict[str, Any]] = []

        for idx in range(start_index, start_index + num_examples):
            user_text, gt_text, image = self.dataset.load_eval_sample(ds, idx)
            prompt = self.dataset.build_eval_prompt(user_text)

            output = sampler(
                input_strings=[prompt],
                images=[np.asarray(image)],
                max_generation_steps=96,
                max_prompt_length=self.settings.max_seq_length,
                temperature=0.0,
            )

            prediction = self._clean_output(output.text[0])
            results.append({
                    "index": idx,
                    "prompt": prompt,
                    "ground_truth": gt_text,
                    "output": prediction,
            })

        return results

    def train(self) -> None:
        """Run the full training pipeline from data prep through trainer execution.

        Steps:
            1. Prepare output directories and optional checkpoint cleanup.
            2. Ensure the dataset is available locally (download if needed).
            3. Optionally emit one debug sample for inspection.
            4. Initialize tokenizer and image processor.
            5. Build train and validation dataset pipelines and compute max_steps.
            6. Create a device mesh for sharding.
            7. Load the base Gemma model checkpoint.
            8. Inject LoRA adapters into the model.
            9. Persist compatibility manifest metadata.
            10. Build optimizer and configured PeftTrainer.
            11. Execute training and close trainer resources.
            12. Run optional post-training export hooks.

        Returns:
            None.
        """
        settings = self.settings

        logger.info(
            "Experiment paths: root=%s checkpoints=%s tensorboard=%s",
            self._experiment_root(),
            self._checkpoint_root(),
            self._tensorboard_root(),
        )

        # 1) Prepare output directories and optional checkpoint cleanup.
        self._setup_dirs()

        # 2) Ensure dataset files are available locally.
        self.dataset.ensure_available()

        # 3) Optionally log and export a single debug sample.
        if settings.log_sample_debug:
            self.dataset.log_sample_debug()

        # 4) Initialize tokenizer and image processor.
        logger.info("Initializing tokenizer from %s...", settings.gemma_tokenizer_path)
        tokenizer = self._build_tokenizer()
        image_processor = self._build_image_processor()

        # 5) Build train/validation pipelines and derive max steps.
        train_ds, val_ds, max_steps = self.dataset.build(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

        # 6) Create mesh and 7) load base model.
        mesh = self.create_mesh()
        self.load_base_model(mesh)

        # 8) Apply LoRA adapters.
        lora_model = self.build_lora_model(mesh=mesh)

        # 9) Write manifest for checking compatibility
        self._write_compat_manifest()

        # 10) Build optimizer and trainer.
        optimizer = self._build_optimizer(max_steps=max_steps)
        trainer = self._build_trainer(
            lora_model=lora_model,
            optimizer=optimizer,
            max_steps=max_steps,
            tokenizer=tokenizer,
        )

        # 12) Run training loop and release trainer resources.
        logger.info("Starting LoRA training with PeftTrainer...")
        with mesh:
            trainer.train(train_ds=train_ds, eval_ds=val_ds)
        trainer.close()
        logger.info("Training complete.")

        self._post_train(lora_model)

    def _post_train(self, lora_model: nnx.Module) -> None:
        """Optional post-training hook for subclasses.

        Args:
            lora_model: Trained LoRA-augmented model instance.

        Returns:
            None.
        """
        del lora_model


def create_trainer(
    settings: Settings | None = None,
    model_config: Optional[gemma3_model_lib.ModelConfig] = None,
    restore_policy: Literal["strict", "permissive"] = "permissive",
    model_source: CheckpointModelSource | None = None,
) -> GemmaEarth:
    """Factory for source-specific trainer implementations.

    Args:
        settings: Optional settings override. Defaults to environment-loaded
            ``Settings``.
        model_config: Optional explicit model config override.
        restore_policy: Restore behavior for loading output checkpoints.
        model_source: Optional model source override. Uses settings default
            when omitted.

    Returns:
        Source-specific ``GemmaEarth`` trainer instance.

    Raises:
        ValueError: If source is unsupported or permissive restore is requested
            for Hugging Face source.
    """
    from .huggingface_checkpoint_trainer import HuggingFaceCheckpointTrainer
    from .tunix_checkpoint_trainer import TunixCheckpointTrainer

    resolved_settings = settings or Settings()
    source = (model_source or resolved_settings.base_model_checkpoint_source).strip().lower()
    if source == "huggingface":
        return HuggingFaceCheckpointTrainer(
            settings=resolved_settings,
            model_config=model_config,
            restore_policy=restore_policy,
        )
    if source == "tunix":
        return TunixCheckpointTrainer(
            settings=resolved_settings,
            model_config=model_config,
            restore_policy=restore_policy,
        )
    raise ValueError(f"Unsupported model source: {source}")
