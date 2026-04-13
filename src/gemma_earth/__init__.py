import logging
import sys

from absl import logging as absl_logging
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=Console(file=sys.stdout), rich_tracebacks=True)],
    force=True,
)

app_logger = logging.getLogger("gemma_earth")
app_logger.setLevel(logging.INFO)

for name in [
    "orbax",
    "orbax.checkpoint",
    "orbax.checkpoint._src.multihost.multihost",
    "orbax.checkpoint._src.checkpointers.async_checkpointer",
    "jax",
    "gcsfs",
    "absl",
    "etils",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

absl_logging.set_verbosity("error")
absl_logging.set_stderrthreshold("error")

logger = logging.getLogger("gemma_earth")

from .trainers import (
    CheckpointModelSource,
    GemmaEarth,
    HuggingFaceCheckpointTrainer,
    TunixCheckpointTrainer,
    create_trainer,
)

__all__ = [
    "logger",
    "CheckpointModelSource",
    "GemmaEarth",
    "TunixCheckpointTrainer",
    "HuggingFaceCheckpointTrainer",
    "create_trainer",
]
