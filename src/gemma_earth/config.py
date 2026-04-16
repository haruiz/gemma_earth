from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Dataset and model artifacts
    hf_dataset_repo_id: str = Field(default="akshaydudhane/EarthDial-Dataset")
    hf_dataset_allow_pattern: str = Field(default="training_set/**")
    gemma_tokenizer_path: str = Field(default="gs://gemma-data/tokenizers/tokenizer_gemma3.model")
    model_ckpt_path: str = Field(default="gs://gemma-data/checkpoints/gemma3-4b-it")
    base_model_checkpoint_source: str = Field(default="tunix")


    # Dataset split and sampling
    num_samples: int | None = Field(default=20000)
    min_validation_samples: int = Field(default=200)
    max_validation_samples: int = Field(default=500)
    val_split_ratio: float = Field(default=0.1)
    shuffle_seed: int = Field(default=42)
    preserve_multi_turn: bool = Field(default=True)


    #huggingface checkpoint source settings
    hf_model_id: str = Field(default="google/gemma-3-4b-it")
    hf_ignore_patterns: str = Field(default="*.pth")

    # LoRA and model
    lora_rank: int = Field(default=16)
    lora_alpha: float = Field(default=32.0)
    max_seq_length: int = Field(default=768)

    # Optimization
    learning_rate: float = Field(default=5e-5)
    weight_decay: float = Field(default=1e-4)
    warmup_ratio: float = Field(default=0.1)

    # Training
    batch_size: int = Field(default=4)
    num_epochs: int = Field(default=2)
    eval_every_n_steps: int = Field(default=50)
    save_interval_steps: int = Field(default=200)
    max_to_keep: int = Field(default=3)

    # Paths and directories
    output_dir: str = Field(default="/mnt/disks/data/gemma_earth_output")
    dataset_download_dir: str = Field(default="/mnt/disks/data/earthdial-dataset")
    dataset_relative_dir: str = Field(
        default="training_set/Classification_Shards_corrected/BigEarthNet_FINAL_RGB/BigEarthNet_train"
    )

    # Runtime behavior
    clean_start: bool = Field(default=True)
    force_download: bool = Field(default=False)
    log_sample_debug: bool = Field(default=True)
    experiment_id_override: str | None = Field(default=None)
    include_runtime_versions_in_experiment_id: bool = Field(default=False)

    @computed_field
    @property
    def dataset_dir(self) -> str:
        """Resolved dataset directory."""
        return str(Path(self.dataset_download_dir) / self.dataset_relative_dir)

    @computed_field
    @property
    def experiments_dir(self) -> str:
        """Resolved experiments directory."""
        return str(Path(self.output_dir) / "experiments")

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be greater than 0")
        return v

    @field_validator("num_epochs")
    @classmethod
    def validate_num_epochs(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_epochs must be greater than 0")
        return v

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_seq_length must be greater than 0")
        return v

    @field_validator("val_split_ratio")
    @classmethod
    def validate_val_split_ratio(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("val_split_ratio must be between 0 and 1")
        return v

    @field_validator("warmup_ratio")
    @classmethod
    def validate_warmup_ratio(cls, v: float) -> float:
        if not 0 <= v < 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        return v

    @field_validator("max_validation_samples")
    @classmethod
    def validate_max_validation_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_validation_samples must be greater than 0")
        return v

    @field_validator("min_validation_samples")
    @classmethod
    def validate_min_validation_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("min_validation_samples must be greater than 0")
        return v

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("learning_rate must be greater than 0")
        return v

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight_decay must be non-negative")
        return v

    @field_validator("base_model_checkpoint_source")
    @classmethod
    def validate_base_model_checkpoint_source(cls, v: str) -> str:
        normalized = v.strip().lower()
        if normalized not in {"tunix", "huggingface"}:
            raise ValueError("base_model_checkpoint_source must be one of: tunix, huggingface")
        return normalized
