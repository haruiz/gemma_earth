import io
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, LocalTokenNotFoundError
from PIL import Image


def is_huggingface_authenticated() -> tuple[bool, str]:
    """Validate Hugging Face authentication from env vars or cached CLI token.

    Checks HF_TOKEN / HUGGING_FACE_HUB_TOKEN environment variables first,
    then falls back to the locally cached CLI token produced by hf auth login.

    Returns:
        A two-tuple (is_authenticated, message) where is_authenticated is
        True on success and message describes the outcome or the error.
    """
    load_dotenv()

    api = HfApi()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    try:
        if token:
            api.whoami(token=token)
        else:
            api.whoami()
        return True, "Authenticated with Hugging Face"
    except LocalTokenNotFoundError:
        return False, "No Hugging Face token found. Run: hf auth login"
    except HfHubHTTPError:
        return False, "Hugging Face token is invalid or expired"
    except Exception as exc:
        return False, f"Could not verify Hugging Face authentication: {exc}"


def decode_image(x: Any) -> Image.Image:
    """Decode different image payload formats into an RGB PIL image.

    Args:
        x: Image payload. Accepted types are PIL.Image.Image (converted to RGB
            in place), dict with a bytes key (decoded from in-memory bytes),
            dict with a path key (opened from the filesystem), bytes or
            bytearray (decoded from raw bytes), and numpy.ndarray (interpreted
            as uint8 pixel data).

    Returns:
        An RGB PIL.Image.Image.

    Raises:
        TypeError: If x is not one of the supported payload types.
    """
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
