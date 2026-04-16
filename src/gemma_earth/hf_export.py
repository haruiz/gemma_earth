"""Hugging Face-compatible safetensors export for Gemma3 LoRA models.

This module merges LoRA adapter weights from an in-memory Tunix/Flax model
into a base Hugging Face Gemma3 checkpoint directory and writes the result as
HF-compatible safetensors artifacts.

Supported base checkpoint layouts:
1. Single-file: ``model.safetensors``.
2. Sharded: ``model-xxxxx-of-yyyyy.safetensors`` + ``model.safetensors.index.json``.

Output format mirrors the input format:
- Single-file input -> single-file output.
- Sharded input -> sharded output with regenerated
  ``model.safetensors.index.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from flax import nnx
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as safe_np
from tunix.models.gemma3 import params as gemma3_params_lib


def _join_path(path: tuple[Any, ...]) -> str:
    """Convert an NNX graph path tuple to dotted string form.

    Args:
        path: NNX graph path elements.

    Returns:
        Dotted path string used by downstream mapping helpers.
    """
    return ".".join(str(field) for field in path)


def _collect_lora_layers(lora_model: Any) -> dict[str, list[Any]]:
    """Collect LoRA adapter pairs from an NNX model graph.

    Args:
        lora_model: Model containing ``nnx.LoRAParam`` nodes.

    Returns:
        Mapping ``layer_path -> [lora_a, lora_b]`` in discovery order.
        The returned layer keys use dotted path notation and exclude trailing
        ``lora_a``/``lora_b`` field names.
    """
    lora_layers: dict[str, list[Any]] = {}
    for path, value in nnx.iter_graph(lora_model):
        if isinstance(value, nnx.LoRAParam):
            path_str = _join_path(path[:-1])
            if path_str in lora_layers:
                lora_layers[path_str].append(value)
            else:
                lora_layers[path_str] = [value]
    return lora_layers


def _load_hf_safetensors(
    model_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[str], bool]:
    """Load base model tensors from a HF checkpoint directory.

    Args:
        model_dir: Directory that contains HF model files.

    Returns:
        A tuple of:
        1. ``base_state``: mapping ``weight_name -> ndarray`` for all tensors.
        2. ``weight_map``: mapping ``weight_name -> shard_filename``.
        3. ``shard_files``: ordered shard filenames used for writing.
        4. ``is_sharded``: whether the input was sharded.

    Raises:
        FileNotFoundError: If neither single-file nor sharded HF artifacts are
            present, or if an index references a missing shard file.
    """
    single_file = model_dir / "model.safetensors"
    index_file = model_dir / "model.safetensors.index.json"

    if single_file.exists():
        state = safe_np.load_file(str(single_file))
        weight_map = {k: "model.safetensors" for k in state.keys()}
        return state, weight_map, ["model.safetensors"], False

    if not index_file.exists():
        raise FileNotFoundError(
            f"Could not find {single_file} or {index_file} in {model_dir}"
        )

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    weight_map = dict(index_data["weight_map"])
    shard_files = list(dict.fromkeys(weight_map.values()))
    state: dict[str, np.ndarray] = {}
    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file listed in index: {shard_path}")
        state.update(safe_np.load_file(str(shard_path)))
    return state, weight_map, shard_files, True


def export_gemma3_lora_merged_hf_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
) -> None:
    """Export LoRA-merged Gemma3 weights as HF-compatible safetensors.

    This function:
    1. Loads base tensors from ``local_model_path`` (single-file or sharded).
    2. Collects LoRA adapter weights from ``lora_model``.
    3. Resolves each LoRA layer to the corresponding HF tensor key.
    4. Applies LoRA delta updates in-place to base tensors.
    5. Writes merged tensors to ``output_dir`` in HF-compatible layout.
    6. Copies non-weight model assets (config/tokenizer/etc.) to output.

    Args:
        local_model_path: Path to base HF checkpoint directory.
        output_dir: Output directory where merged model artifacts are written.
        lora_model: Tunix/Flax model containing trained LoRA parameters.
        rank: LoRA rank used during training.
        alpha: LoRA alpha scaling used during training.

    Raises:
        FileNotFoundError: If base checkpoint files are missing.
        KeyError: If one or more LoRA layers cannot be mapped to HF keys.
        ValueError: If a LoRA delta cannot be aligned to target tensor shape.
    """
    model_dir = Path(local_model_path)
    out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    base_state, weight_map, shard_files, is_sharded = _load_hf_safetensors(model_dir)

    lora_layers = _collect_lora_layers(lora_model)
    lora_layers = gemma3_params_lib._extract_gemma3_lora_layers(lora_layers)
    transform_key = gemma3_params_lib._gemma3_state_key_to_safetensors_key
    transpose_rules = gemma3_params_lib._GEMMA3_HUGGINGFACE_TRANSPOSE_RULES
    base_keys = set(base_state.keys())

    def _resolve_state_key(lora_path: str) -> str | None:
        """Resolve one LoRA path to an existing HF tensor key.

        Handles multiple naming variants used across Gemma3 checkpoint formats:
        - ``model.*`` vs ``language_model.model.*``.
        - Vision tower and multimodal projector key differences.

        Args:
            lora_path: Dotted internal LoRA layer path.

        Returns:
            Matched HF tensor key if found, else ``None``.
        """
        key = transform_key(lora_path)
        candidates = [key]

        # Newer HF Gemma3 checkpoints use `language_model.model.*` namespace.
        if key.startswith("model."):
            candidates.append("language_model." + key)

        normalized = lora_path[:-7] if lora_path.endswith(".kernel") else lora_path

        # Vision tower attention/MLP adapters in HF naming.
        m = re.fullmatch(
            r"vision_encoder\.siglip_encoder\.transformer\.blocks\.(\d+)\.attn\.(query_proj|key_proj|value_proj|out_proj)",
            normalized,
        )
        if m:
            layer, proj = m.groups()
            proj_map = {
                "query_proj": "q_proj",
                "key_proj": "k_proj",
                "value_proj": "v_proj",
                "out_proj": "out_proj",
            }
            candidates.append(
                f"vision_tower.vision_model.encoder.layers.{layer}.self_attn.{proj_map[proj]}.weight"
            )

        m = re.fullmatch(
            r"vision_encoder\.siglip_encoder\.transformer\.blocks\.(\d+)\.mlp\.(fc1|fc2)",
            normalized,
        )
        if m:
            layer, fc = m.groups()
            candidates.append(
                f"vision_tower.vision_model.encoder.layers.{layer}.mlp.{fc}.weight"
            )

        if normalized == "embedder.mm_input_projection":
            candidates.append("multi_modal_projector.mm_input_projection_weight")

        for candidate in candidates:
            if candidate in base_keys:
                return candidate
        return None

    missing_keys: list[tuple[str, str]] = []
    for path, (lora_a, lora_b) in lora_layers.items():
        state_key = _resolve_state_key(path)
        if state_key is None:
            missing_keys.append((path, transform_key(path)))
            continue

        lora_a_val = jnp.asarray(getattr(lora_a, "value", lora_a))
        lora_b_val = jnp.asarray(getattr(lora_b, "value", lora_b))

        # Some LoRA params may be stored in packed 3D forms; flatten to 2D.
        if lora_a_val.ndim == 3:
            d0, d1, d2 = lora_a_val.shape
            lora_a_val = lora_a_val.reshape(d0 * d1, d2)
        if lora_b_val.ndim == 3:
            d0, d1, d2 = lora_b_val.shape
            lora_b_val = lora_b_val.reshape(d0, d1 * d2)

        delta = (lora_a_val @ lora_b_val) * (alpha / rank)
        for key_fragment, rule in transpose_rules.items():
            if key_fragment in state_key:
                delta = delta.transpose(rule)
                break

        base = base_state[state_key]
        delta_np = np.asarray(delta, dtype=base.dtype)
        if delta_np.shape != base.shape:
            # Some checkpoints use opposite kernel orientation.
            # Prefer exact match, fallback to 2D transpose when valid.
            if delta_np.ndim == 2 and delta_np.T.shape == base.shape:
                delta_np = delta_np.T
            else:
                raise ValueError(
                    f"Delta/base shape mismatch for {state_key}: "
                    f"delta={delta_np.shape}, base={base.shape}"
                )
        base_state[state_key] = base + delta_np

    if missing_keys:
        preview = ", ".join(
            f"{path}->{fallback}" for path, fallback in missing_keys[:8]
        )
        raise KeyError(
            "Some LoRA layers could not be mapped to HF tensor keys. "
            f"Missing: {len(missing_keys)}. First entries: {preview}"
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-weight assets so output is directly loadable by HF APIs.
    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors"):
            continue
        if filename == "model.safetensors.index.json" and is_sharded:
            continue
        src = model_dir / filename
        if src.is_file():
            shutil.copy(src, out_dir / filename)

    if not is_sharded:
        safe_np.save_file(base_state, str(out_dir / "model.safetensors"))
        return

    # Rebuild shard assignment from original index and write shards back out.
    shard_to_keys: dict[str, list[str]] = {name: [] for name in shard_files}
    for key, shard_name in weight_map.items():
        if shard_name not in shard_to_keys:
            shard_to_keys[shard_name] = []
            shard_files.append(shard_name)
        shard_to_keys[shard_name].append(key)

    fallback_shard = shard_files[0]
    for key in base_state.keys():
        if key not in weight_map:
            weight_map[key] = fallback_shard
            shard_to_keys[fallback_shard].append(key)

    for shard_name in shard_files:
        shard_state = {k: base_state[k] for k in shard_to_keys[shard_name] if k in base_state}
        safe_np.save_file(shard_state, str(out_dir / shard_name))

    total_size = int(sum(arr.nbytes for arr in base_state.values()))
    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
