import fnmatch
import json
import re
from pathlib import Path
from typing import Any, Literal

from . import logger
from .config import Settings
from .utils import decode_image, is_huggingface_authenticated
from huggingface_hub import snapshot_download

import grain
import numpy as np
from datasets import load_from_disk
from huggingface_hub import HfApi
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.sft import peft_trainer


class EarthDialDataset:
    """Loads EarthDial data and builds train and validation iterables for Tunix."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create dataset helper with configuration state.

        Args:
            settings: Optional Settings instance. If not provided, a new
                Settings object is created and stored in self.settings.

        Returns:
            None.
        """
        self.settings = settings or Settings()

    def get_dataset_size_bytes(self) -> None | int | Literal[0]:
        """Return total size in bytes of EarthDial dataset files matching the allow pattern.

        Queries Hugging Face dataset metadata and sums the sizes of all sibling
        files whose path matches self.settings.hf_dataset_allow_pattern.

        Returns:
            Total size in bytes, or -1 if the metadata lookup fails.
        """
        settings = self.settings
        api = HfApi()
        try:
            dataset_info = api.dataset_info(
                repo_id=settings.hf_dataset_repo_id,
                files_metadata=True,
            )
            sizes = [
                s.size
                for s in (dataset_info.siblings or [])
                if s.size is not None
                and fnmatch.fnmatch(s.rfilename, settings.hf_dataset_allow_pattern)
            ]
            return sum(sizes)
        except Exception as exc:
            logger.error("Error fetching EarthDial dataset info: %s", exc)
            return -1

    def ensure_available(self) -> None:
        """Ensure EarthDial files are present locally, downloading if needed.

        When a download is required, queries the Hugging Face API first to log
        the expected download size. Skips the size check when the dataset is
        already present on disk.

        Returns:
            None.
        """
        settings = self.settings
        dataset_dir = Path(settings.dataset_dir)
        download_dir = Path(settings.dataset_download_dir)

        if settings.force_download or not dataset_dir.exists():
            is_auth, message = is_huggingface_authenticated()
            if not is_auth:
                raise RuntimeError(message)

            logger.info("Downloading EarthDial dataset from Hugging Face...")
            self.download_earthdial_dataset(output_dir=str(download_dir))
        else:
            logger.info("Dataset already on disk, skipping download.")

        dataset_size_bytes = self.get_dataset_size_bytes()
        if dataset_size_bytes < 0:
            logger.error("Could not determine EarthDial dataset size. Aborting download.")
            return
        dataset_size_gb = dataset_size_bytes / (1024 ** 3)
        logger.info("EarthDial dataset size: %.2f GB", dataset_size_gb)

            

    def download_earthdial_dataset(self, output_dir: str) -> None:
        """Download EarthDial dataset shard files required for training.

        Uses huggingface_hub.snapshot_download to fetch only the files
        matching self.settings.hf_dataset_allow_pattern.

        Args:
            output_dir: Local directory path where downloaded files will be saved.
        """
        settings = self.settings
        snapshot_download(
            repo_id=settings.hf_dataset_repo_id,
            repo_type="dataset",
            allow_patterns=settings.hf_dataset_allow_pattern,
            local_dir=output_dir,
        )

    def log_sample_debug(self) -> None:
        """Log and export one sample to confirm dataset decoding works.

        Returns:
            None.
        """
        settings = self.settings
        ds = load_from_disk(settings.dataset_dir)
        logger.info("Dataset loaded with %d samples.", len(ds))
        image_path, prompt_path = self.save_example_image_and_prompt(index=0)
        logger.info("Sample image saved to %s", image_path)
        logger.info("Sample prompt saved to %s", prompt_path)
        logger.info("Sample conversations: %s", ds[0].get("conversations"))

    def save_example_image_and_prompt(
        self,
        index: int = 0,
        output_dir: str | Path = ".",
        image_filename: str = "sample_image.jpg",
        prompt_filename: str = "sample_prompt.txt",
    ) -> tuple[Path, Path]:
        """Save one dataset example image and formatted prompt to disk.

        Args:
            index: Dataset row index to export.
            output_dir: Directory where files will be written.
            image_filename: Output image file name.
            prompt_filename: Output prompt text file name.

        Returns:
            Tuple ``(image_path, prompt_path)``.

        Raises:
            IndexError: If ``index`` is outside dataset bounds.
        """
        settings = self.settings
        ds = load_from_disk(settings.dataset_dir)
        if index < 0 or index >= len(ds):
            raise IndexError(f"index must be in [0, {len(ds) - 1}], got {index}")

        sample = ds[index]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        image = decode_image(sample["jpg"]).convert("RGB")
        image_path = out / image_filename
        image.save(image_path)

        conversations = self._parse_conversations(sample.get("conversations", []))
        prompt, _, _, _ = self._format_prompt_and_response(conversations)
        prompt_path = out / prompt_filename
        prompt_path.write_text(prompt, encoding="utf-8")

        return image_path, prompt_path

    def _compute_validation_size(self, sample_limit: int) -> int:
        """Compute the number of examples to reserve for validation.

        The size is derived from settings.val_split_ratio then clamped to
        the [settings.min_validation_samples, settings.max_validation_samples]
        range and further bounded so that at least one training batch remains.

        Args:
            sample_limit: Total number of samples available after applying the
                settings.num_samples cap.

        Returns:
            Number of examples to use for the validation split.

        Raises:
            ValueError: If there are not enough samples to form at least one
                training batch with the configured self.settings.batch_size.
        """
        settings = self.settings
        ratio_size = int(sample_limit * settings.val_split_ratio)
        validation_size = max(settings.min_validation_samples, ratio_size)
        validation_size = min(validation_size, settings.max_validation_samples)

        max_allowed = sample_limit - settings.batch_size
        if max_allowed <= 0:
            raise ValueError(
                f"Not enough samples for train and validation with batch_size={settings.batch_size}."
            )

        validation_size = min(validation_size, max_allowed)
        validation_size = max(1, validation_size)
        return validation_size

    @staticmethod
    def _parse_conversations(raw_conversations: Any) -> list[dict[str, Any]]:
        """Convert a conversation payload into a list of validated turn dicts.

        Args:
            raw_conversations: Raw conversation data from the dataset row. May be
                a JSON-encoded string, a list of dicts, or any other value.

        Returns:
            A list of conversation-turn dicts. Returns an empty list when the
            input cannot be parsed or does not contain a list of dicts.
        """
        if isinstance(raw_conversations, str):
            try:
                raw_conversations = json.loads(raw_conversations)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed conversations payload.")
                return []

        if not isinstance(raw_conversations, list):
            return []

        return [turn for turn in raw_conversations if isinstance(turn, dict)]

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize a conversation turn's text.

        Strips optional leading bracket annotations (e.g. [ImageDescription]),
        removes <image> placeholder tokens, and trims surrounding whitespace.

        Args:
            text: Raw turn text from the conversation payload.

        Returns:
            Cleaned, stripped text string.
        """
        return re.sub(r"^\[[^\]]+\]\s*", "", text).replace("<image>", "").strip()

    def _format_prompt_and_response(
        self,
        conversations: list[dict[str, Any]],
    ) -> tuple[str, str, bool, bool]:
        """Format EarthDial conversations into Gemma turn-based instruction text.

        When settings.preserve_multi_turn is True all conversation turns are
        interleaved; otherwise only the first human/model pair is used.
        The image placeholder (256 <img> tokens) is always inserted in the
        first user turn.

        Args:
            conversations: List of turn dicts, each containing "from" (role)
                and "value" (text) keys. An empty list produces a default
                fallback prompt and response.

        Returns:
            A four-tuple of (prompt, chosen_response, human_empty, model_empty).
            prompt is the formatted instruction text ending with the opening
            <start_of_turn>model tag. chosen_response is the model response
            text ending with <end_of_turn>. human_empty is True when the
            human/user turn is empty. model_empty is True when the
            model/assistant response is empty.
        """
        settings = self.settings
        if not conversations:
            return (
                "<start_of_turn>user\n"
                + "<start_of_image>"
                + "<img>" * 256
                + "<end_of_image>\n\nDescribe the image.<end_of_turn>\n<start_of_turn>model\n",
                "Unknown.<end_of_turn>",
                True,
                True,
            )

        if settings.preserve_multi_turn:
            prompt_parts: list[str] = []
            last_model_text = ""
            human_texts: list[str] = []

            for i, turn in enumerate(conversations):
                role = turn.get("from", "")
                value = self._clean_text(turn.get("value", ""))

                if role in ("human", "user"):
                    human_texts.append(value)
                    prompt_parts.append(
                        "<start_of_turn>user\n"
                        + "<start_of_image>"
                        + "<img>" * 256
                        + "<end_of_image>\n\n"
                        + value
                        + "<end_of_turn>\n"
                    )
                elif role in ("gpt", "assistant", "model"):
                    last_model_text = value
                    if i != len(conversations) - 1:
                        prompt_parts.append(
                            "<start_of_turn>model\n"
                            + value
                            + "<end_of_turn>\n"
                        )

            prompt = "".join(prompt_parts)
            if not prompt.endswith("<start_of_turn>model\n"):
                prompt += "<start_of_turn>model\n"

            chosen_response = (
                last_model_text if last_model_text.endswith("<end_of_turn>")
                else f"{last_model_text}<end_of_turn>"
            )

            return (
                prompt,
                chosen_response,
                len(" ".join(human_texts).strip()) == 0,
                len(last_model_text.strip()) == 0,
            )

        human_text = next(
            (
                self._clean_text(turn.get("value", ""))
                for turn in conversations
                if turn.get("from") in ("human", "user")
            ),
            "",
        )
        model_text = next(
            (
                self._clean_text(turn.get("value", ""))
                for turn in conversations
                if turn.get("from") in ("gpt", "assistant", "model")
            ),
            "",
        )

        prompt = (
            "<start_of_turn>user\n"
            + "<start_of_image>"
            + "<img>" * 256
            + "<end_of_image>\n\n"
            + human_text
            + "<end_of_turn>\n"
            + "<start_of_turn>model\n"
        )

        chosen_response = (
            model_text if model_text.endswith("<end_of_turn>")
            else f"{model_text}<end_of_turn>"
        )

        return prompt, chosen_response, len(human_text) == 0, len(model_text) == 0

    def build_eval_prompt(self, user_text: str) -> str:
        """Build an evaluation prompt from one user text instruction.

        Args:
            user_text: Raw user instruction/content.

        Returns:
            Gemma-formatted multimodal prompt ending with `<start_of_turn>model`.
        """
        prompt, _, _, _ = self._format_prompt_and_response(
            conversations=[{"from": "user", "value": user_text}]
        )
        return prompt

    def load_eval_sample(self, ds: Any, idx: int) -> tuple[str, str, Any]:
        """Load one evaluation sample from an already-loaded dataset.

        Args:
            ds: Loaded EarthDial dataset object.
            idx: Sample index.

        Returns:
            Tuple of `(user_text, ground_truth_text, image)` where `image` is an
            RGB PIL image decoded from dataset payload.

        Raises:
            ValueError: If `idx` is outside dataset bounds.
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

        return user_text, gt_text, decode_image(sample["jpg"])

    def _to_training_example(self, x: dict[str, Any], image_processor: Any) -> dict[str, Any]:
        """Transform a raw dataset row into text and image fields for SFT training.

        Args:
            x: Raw dataset row containing at minimum a conversations field
               (list or JSON string) and a jpg field (image payload supported
               by decode_image).
            image_processor: Tunix ImageProcessor used to convert the decoded
               PIL image into a model-compatible float array.

        Returns:
            A dict with keys: prompts (formatted prompt string),
            chosen_responses (formatted model-response string),
            images (numpy.ndarray of the processed image),
            human_empty (whether the human turn was empty), and
            model_empty (whether the model turn was empty).
        """
        conversations = self._parse_conversations(x.get("conversations", []))
        prompt, chosen_response, human_empty, model_empty = self._format_prompt_and_response(conversations)

        image = decode_image(x["jpg"]).convert("RGB")

        return {
            "prompts": prompt,
            "chosen_responses": chosen_response,
            "images": np.array(image_processor(image)[0]),
            "human_empty": human_empty,
            "model_empty": model_empty,
        }

    def _to_training_input(
        self,
        x: dict[str, Any],
        tokenizer: tokenizer_lib.Tokenizer,
    ) -> peft_trainer.TrainingInput:
        """Tokenize and pad one mapped sample into TrainingInput format.

        Concatenates prompt and response token arrays, builds a boolean loss mask
        (True only for response positions), truncates to settings.max_seq_length
        if necessary, and right-pads to a fixed length with the tokenizer pad ID.

        Args:
            x: Mapped sample dict produced by _to_training_example.
            tokenizer: Tunix tokenizer used to encode the prompt and response
                strings and to supply the pad and EOS token IDs.

        Returns:
            A peft_trainer.TrainingInput with padded int32 token IDs,
            a boolean loss mask, and the processed image array.
        """
        settings = self.settings
        prompt_tokens = tokenizer.tokenize(x["prompts"], add_eos=False)
        response_tokens = tokenizer.tokenize(x["chosen_responses"], add_eos=False)

        if response_tokens.size == 0:
            response_tokens = np.array([tokenizer.eos_id()], dtype=np.int32)

        tokens = np.concatenate([prompt_tokens, response_tokens], axis=0)
        prompt_mask = np.zeros(prompt_tokens.shape[0], dtype=np.bool_)
        response_mask = np.ones(response_tokens.shape[0], dtype=np.bool_)
        mask = np.concatenate([prompt_mask, response_mask], axis=0)

        if tokens.shape[0] > settings.max_seq_length:
            tokens = tokens[: settings.max_seq_length]
            mask = mask[: settings.max_seq_length]

        pad_len = settings.max_seq_length - tokens.shape[0]
        if pad_len > 0:
            tokens = np.pad(tokens, (0, pad_len), constant_values=tokenizer.pad_id())
            mask = np.pad(mask, (0, pad_len), constant_values=False)

        return peft_trainer.TrainingInput(
            input_tokens=tokens.astype(np.int32),
            input_mask=mask.astype(np.bool_),
            images=np.asarray(x["images"], dtype=np.float32),
        )

    def _build_pipeline(
        self,
        split: Any,
        image_processor: Any,
        tokenizer: tokenizer_lib.Tokenizer,
        num_epochs: int | None,
        split_name: str,
    ) -> Any:
        """Construct a shuffled, mapped, and batched Grain iterable pipeline.

        The pipeline applies the following operations in order: shuffle, map to
        training example, map to training input (tokenize + pad), batch, repeat,
        and convert to an iterable dataset.

        Args:
            split: HuggingFace Dataset split used as the Grain data source.
            image_processor: Tunix ImageProcessor forwarded to
                _to_training_example.
            tokenizer: Tunix tokenizer forwarded to _to_training_input.
            num_epochs: Number of times to repeat the dataset. Pass 1 (or
                None) for a single pass, typically used for validation.
            split_name: Human-readable label ("train" or "validation")
                used in progress log messages.

        Returns:
            A Grain iterable dataset ready for use as train_ds or
            eval_ds in GemmaEarth.train.
        """
        settings = self.settings
        logger.info("Building %s pipeline with %d rows...", split_name, len(split))
        return (
            grain.MapDataset.source(split)
            .shuffle(seed=settings.shuffle_seed)
            .map(lambda x: self._to_training_example(x, image_processor))
            .map(lambda x: self._to_training_input(x, tokenizer))
            .batch(settings.batch_size, drop_remainder=True)
            .repeat(num_epochs)
            .to_iter_dataset()
        )

    def build(
        self,
        image_processor: Any,
        tokenizer: tokenizer_lib.Tokenizer,
    ) -> tuple[Any, Any, int]:
        """Build shuffled train and validation Grain datasets from disk.

        Loads the dataset from ``settings.dataset_dir``, caps the total
        sample count at ``settings.num_samples``, splits into train and
        validation subsets, and constructs a Grain pipeline for each.

        Args:
            image_processor: Tunix ImageProcessor passed through to
                _build_pipeline.
            tokenizer: Tunix tokenizer passed through to _build_pipeline.

        Returns:
            A three-tuple of (train_dataset, val_dataset, max_steps).
            train_dataset is a Grain iterable for training. val_dataset is a
            Grain iterable for validation. max_steps is the total training
            steps computed as (train_size // batch_size) * num_epochs.
        """
        settings = self.settings
        ds = load_from_disk(settings.dataset_dir)
        logger.info("Dataset loaded with %d samples.", len(ds))

        sample_limit = len(ds) if settings.num_samples is None else min(len(ds), settings.num_samples)
        ds = ds.shuffle(seed=settings.shuffle_seed).select(range(sample_limit))

        validation_size = self._compute_validation_size(sample_limit)
        split_ds = ds.train_test_split(
            test_size=validation_size,
            seed=settings.shuffle_seed,
            shuffle=True,
        )

        train_split = split_ds["train"]
        val_split = split_ds["test"]

        logger.info(
            "Using %d sampled rows: %d train / %d validation.",
            sample_limit,
            len(train_split),
            len(val_split),
        )

        train_dataset = self._build_pipeline(
            split=train_split,
            image_processor=image_processor,
            tokenizer=tokenizer,
            num_epochs=settings.num_epochs,
            split_name="train",
        )

        val_dataset = self._build_pipeline(
            split=val_split,
            image_processor=image_processor,
            tokenizer=tokenizer,
            num_epochs=1,
            split_name="validation",
        )

        steps_per_epoch = len(train_split) // settings.batch_size
        max_steps = max(1, steps_per_epoch * settings.num_epochs)

        logger.info("Steps per epoch: %d", steps_per_epoch)
        logger.info("Max training steps: %d", max_steps)

        return train_dataset, val_dataset, max_steps
