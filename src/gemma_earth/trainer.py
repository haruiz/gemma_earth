from collections.abc import Callable
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Optional
import io
import json
import os
import re
import shutil

from . import logger
from .config import Settings
from .dataset import EarthDialDataset

from datasets import load_from_disk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import qwix
from flax import nnx
from PIL import Image
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as params_lib
from tunix.processors import image_processor as image_processor_lib
from tunix.sft import checkpoint_manager as sft_checkpoint_manager
from tunix.sft import metrics_logger, peft_trainer, utils


class GemmaEarth:
    """Encapsulates Gemma model setup, LoRA wrapping, and training execution."""

    LORA_MODULE_PATH = (
        r".*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*down_proj|"
        r".*up_proj|.*query_proj|.*key_proj|.*value_proj|.*out_proj|.*fc1|.*fc2"
    )

    def __init__(
        self,
        settings: Settings | None = None,
        model_config: Optional[gemma3_model_lib.ModelConfig] = None,
        restore_policy: Literal["strict", "permissive"] = "strict",
    ) -> None:
        """Create trainer state with settings, dataset, and selected model config.

        Args:
            model_config: Optional explicit ModelConfig override.

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
        self.restore_policy = restore_policy

    def _pkg_version(self, package_name: str) -> str:
        """Return installed package version or unknown when unavailable."""
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "unknown"

    def _compat_spec(self) -> dict[str, Any]:
        """Build a compatibility signature for checkpoint reuse decisions."""
        settings = self.settings
        return {
            "model_ckpt_path": settings.model_ckpt_path,
            "tokenizer_path": settings.gemma_tokenizer_path,
            "lora_rank": settings.lora_rank,
            "lora_alpha": settings.lora_alpha,
            "lora_module_path": self.LORA_MODULE_PATH,
            "max_seq_length": settings.max_seq_length,
            "model_config_repr": repr(self.model_config),
            "tunix_version": self._pkg_version("tunix"),
            "flax_version": self._pkg_version("flax"),
            "orbax_version": self._pkg_version("orbax-checkpoint"),
        }

    def _experiment_id(self) -> str:
        """Create a stable short hash for the current compatibility spec."""
        payload = json.dumps(self._compat_spec(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def _checkpoint_root(self) -> str:
        """Resolve experiment-specific checkpoint root directory."""
        return str(Path(self.settings.checkpoint_dir) / self._experiment_id())

    def _compat_manifest_path(self) -> Path:
        """Return path to compatibility manifest for this experiment."""
        return Path(self._checkpoint_root()) / "compat.json"

    def _write_compat_manifest(self) -> None:
        """Persist compatibility metadata next to checkpoints."""
        root = Path(self._checkpoint_root())
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
        """Read compatibility manifest when present."""
        path = self._compat_manifest_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _assert_compatible_or_raise(self) -> None:
        """Validate checkpoint compatibility for strict restore policy."""
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
        checkpoint_root = self._checkpoint_root()
        if settings.clean_start and os.path.exists(checkpoint_root):
            shutil.rmtree(checkpoint_root)
            logger.info("Cleaned existing checkpoints in %s", checkpoint_root)

        os.makedirs(settings.output_dir, exist_ok=True)
        os.makedirs(settings.tensorboard_log_dir, exist_ok=True)
        os.makedirs(checkpoint_root, exist_ok=True)
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

    def load_base_model(self, mesh: jax.sharding.Mesh) -> tuple[gemma3_model_lib.ModelConfig, nnx.Module]:
        """Load the selected base checkpoint into a sharded model instance.

        Args:
            mesh: JAX mesh used during checkpoint restoration.

        Returns:
            Tuple of selected model config and loaded base model.
        """
        settings = self.settings
        logger.info(
            "Loading model config %s with checkpoint from %s...",
            self.model_config.__class__.__qualname__,
            settings.model_ckpt_path,
        )

        with mesh:
            base_model = params_lib.create_model_from_checkpoint(
                settings.model_ckpt_path,
                self.model_config,
                mesh,
                dtype=jnp.bfloat16,
            )

        return self.model_config, base_model

    def build_lora_model(self, base_model: nnx.Module, mesh: jax.sharding.Mesh) -> nnx.Module:
        """Apply LoRA adapters to the base model and constrain sharding.

        Args:
            base_model: Loaded Gemma base model.
            mesh: Mesh used for sharding constraints.

        Returns:
            LoRA-augmented model.
        """
        settings = self.settings
        lora_provider = qwix.LoraProvider(
            module_path=self.LORA_MODULE_PATH,
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
        )

        model_input = base_model.get_model_input()
        lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)

        with mesh:
            state = nnx.state(lora_model)
            pspecs = nnx.get_partition_spec(state)
            nnx.update(lora_model, jax.lax.with_sharding_constraint(state, pspecs))

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

    def _build_training_config(self, max_steps: int) -> peft_trainer.TrainingConfig:
        """Build Tunix training config including checkpoint and metric options.

        Args:
            max_steps: Maximum number of training steps for the run.

        Returns:
            peft_trainer.TrainingConfig with checkpointing and logging options.
        """
        settings = self.settings
        checkpointing_options = ocp.CheckpointManagerOptions(
            save_interval_steps=settings.save_interval_steps,
            max_to_keep=settings.max_to_keep,
        )

        metrics_logging_options = metrics_logger.MetricsLoggerOptions(
            log_dir=settings.tensorboard_log_dir,
            flush_every_n_steps=20,
        )

        return peft_trainer.TrainingConfig(
            eval_every_n_steps=settings.eval_every_n_steps,
            max_steps=max_steps,
            checkpoint_root_directory=self._checkpoint_root(),
            checkpointing_options=checkpointing_options,
            metrics_logging_options=metrics_logging_options,
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
    def _clean_text(text: str) -> str:
        """Normalize text fields the same way training and inference do."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"^\[[^\]]+\]\s*", "", text)
        text = text.replace("<image>", "")
        return text.strip()

    @staticmethod
    def _clean_output(text: str) -> str:
        """Remove trailing generation markers from model output text."""
        if not isinstance(text, str):
            return ""
        return text.split("<end_of_turn>")[0].strip()

    @staticmethod
    def _decode_image(x: Any) -> Image.Image:
        """Decode supported dataset image payloads into RGB PIL images."""
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, dict):
            if x.get("bytes") is not None:
                return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
            if x.get("path") is not None:
                return Image.open(x["path"]).convert("RGB")
        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(x)).convert("RGB")
        if isinstance(x, np.ndarray):
            return Image.fromarray(x.astype("uint8")).convert("RGB")
        raise TypeError(f"Unsupported image payload type: {type(x)}")

    @staticmethod
    def _parse_conversations(raw: Any) -> list[dict[str, Any]]:
        """Parse the EarthDial conversations payload into validated turn dicts."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        if not isinstance(raw, list):
            return []
        return [turn for turn in raw if isinstance(turn, dict)]

    @staticmethod
    def _build_prompt(user_text: str) -> str:
        """Build one multimodal Gemma prompt from user text."""
        user_text = GemmaEarth._clean_text(user_text)
        return (
            "<start_of_turn>user\n"
            + "<start_of_image>"
            + "<img>" * 256
            + "<end_of_image>\n\n"
            + user_text
            + "<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )

    def _load_eval_sample(self, ds: Any, idx: int) -> tuple[str, str, Image.Image]:
        """Load one evaluation sample from an already-loaded dataset.

        Args:
            ds: Loaded EarthDial dataset object.
            idx: Sample index.

        Returns:
            Tuple of user_text, ground_truth_text, and decoded PIL image.
        """
        if idx < 0 or idx >= len(ds):
            raise ValueError(f"sample index must be in [0, {len(ds) - 1}], got {idx}")

        sample = ds[idx]
        conv = self._parse_conversations(sample.get("conversations", []))

        user_text = next(
            (self._clean_text(t.get("value", "")) for t in conv if t.get("from") in ("human", "user")),
            "",
        )
        gt_text = next(
            (
                self._clean_text(t.get("value", ""))
                for t in conv
                if t.get("from") in ("gpt", "assistant", "model")
            ),
            "",
        )

        return user_text, gt_text, self._decode_image(sample["jpg"])

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
            step: Optional explicit checkpoint step. Uses latest when None.

        Returns:
            Restored checkpoint step.

        Raises:
            RuntimeError: If no checkpoint is found under checkpoint_dir.
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
        _, base_model = self.load_base_model(mesh)
        lora_model = self.build_lora_model(base_model=base_model, mesh=mesh)
        restored_step = self._restore_latest_checkpoint(lora_model)
        logger.info("Loaded checkpoint step: %d", restored_step)

        tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=self.settings.gemma_tokenizer_path)
        image_processor = image_processor_lib.ImageProcessor(config=self.model_config.vision_config)

        cache_config = sampler_lib.CacheConfig(
            cache_size=max(self.settings.max_seq_length + 128, 1024),
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
        )
        sampler = sampler_lib.Sampler(
            transformer=lora_model,
            tokenizer=tokenizer,
            cache_config=cache_config,
            image_processor=image_processor,
        )

        ds = load_from_disk(self.settings.dataset_dir)
        results: list[dict[str, Any]] = []

        for idx in range(start_index, start_index + num_examples):
            user_text, gt_text, image = self._load_eval_sample(ds, idx)
            prompt = self._build_prompt(user_text)

            output = sampler(
                input_strings=[prompt],
                images=[np.asarray(image)],
                max_generation_steps=96,
                max_prompt_length=self.settings.max_seq_length,
                temperature=0.0,
            )

            prediction = self._clean_output(output.text[0])
            results.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "ground_truth": gt_text,
                    "output": prediction,
                }
            )

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
            9. Build optimizer and training configuration.
            10. Build a PeftTrainer and attach model-input callback.
            11. Execute training and close trainer resources.

        Returns:
            None.
        """
        settings = self.settings

        # 1) Prepare output directories and optional checkpoint cleanup.
        self._setup_dirs()

        # 2) Ensure dataset files are available locally.
        self.dataset.ensure_available()

        # 3) Optionally log and export a single debug sample.
        if settings.log_sample_debug:
            self.dataset.log_sample_debug()

        # 4) Initialize tokenizer and image processor.
        logger.info("Initializing tokenizer from %s...", settings.gemma_tokenizer_path)
        tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=settings.gemma_tokenizer_path)
        image_processor = image_processor_lib.ImageProcessor(config=self.model_config.vision_config)

        # 5) Build train/validation pipelines and derive max steps.
        train_ds, val_ds, max_steps = self.dataset.build(
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

        # 6) Create mesh and 7) load base model.
        mesh = self.create_mesh()
        _, base_model = self.load_base_model(mesh)

        # 8) Apply LoRA adapters.
        lora_model = self.build_lora_model(base_model=base_model, mesh=mesh)

        # 9) Build optimizer and training configuration.
        self._write_compat_manifest()
        optimizer = self._build_optimizer(max_steps=max_steps)
        training_config = self._build_training_config(max_steps=max_steps)

        # 10) Create trainer and bind model-input generation callback.
        trainer = peft_trainer.PeftTrainer(
            model=lora_model,
            optimizer=optimizer,
            training_config=training_config,
        )
        trainer = trainer.with_gen_model_input_fn(self._gen_model_input_fn(tokenizer))

        # 11) Run training loop and release trainer resources.
        logger.info("Starting LoRA training with PeftTrainer...")
        with mesh:
            trainer.train(train_ds=train_ds, eval_ds=val_ds)
        trainer.close()
        logger.info("Training complete.")
