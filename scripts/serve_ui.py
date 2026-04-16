#!/usr/bin/env python3
"""Chainlit UI for GemmaEarth inference API.

Serves a chat interface that calls the /predict endpoint of serve_fastapi.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import chainlit as cl
import httpx
from dotenv import load_dotenv

# Load environment variables at module level (before auth callback is registered)
load_dotenv()


class APIClient:
    """Client for the GemmaEarth inference API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=300.0)

    async def health(self) -> dict:
        """Check server health."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    async def predict(
        self,
        message: str,
        image_data: bytes | None = None,
        temperature: float = 0.0,
        max_generation_steps: int = 96,
        max_prompt_length: int = 768,
    ) -> dict:
        """Send prediction request to the API."""
        data = {
            "message": message,
            "temperature": temperature,
            "max_generation_steps": max_generation_steps,
            "max_prompt_length": max_prompt_length,
        }

        files = {}
        if image_data:
            files["image"] = ("image.jpg", image_data, "image/jpeg")

        response = await self.client.post(
            f"{self.base_url}/predict",
            data=data,
            files=files if files else None,
        )
        response.raise_for_status()
        return response.json()


# Global API client
api_client = None


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    global api_client

    settings = cl.user_session.get("settings", {})
    base_url = settings.get("base_url", "http://localhost:8000")
    api_client = APIClient(base_url=base_url)

    # Check server health
    health = await api_client.health()
    if health.get("status") == "ok":
        status_msg = "Connected to inference server"
    else:
        status_msg = f"Server status: {health.get('status', 'unknown')}"

    await cl.Message(
        content=(
            f"{status_msg}\n\n"
            "Welcome to GemmaEarth\n\n"
            "I can classify remote sensing scenes. You can:\n"
            "- Type a classification prompt\n"
            "- Upload satellite images for analysis\n"
            "- Adjust generation parameters\n\n"
            "Try something like:\n"
            '"Classify this remote sensing scene"\n'
            '"What type of land cover is shown?"'
        ),
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and image uploads."""
    global api_client

    if not api_client:
        await cl.Message(content="API client not initialized").send()
        return

    # Extract text and images
    text = message.content
    images = [
        file for file in message.elements if getattr(file, "mime", "").startswith("image/")
    ]

    if not text and not images:
        await cl.Message(content="Please provide a message or upload an image.").send()
        return

    # Prepare request
    image_data = None
    if images:
        image_file = images[0]
        image_path = getattr(image_file, "path", None)
        if image_path:
            image_data = Path(image_path).read_bytes()

    # Default message if only image provided
    if not text:
        text = "Classify this remote sensing scene"

    # Show processing message
    msg = cl.Message(content="Processing your request...")
    await msg.send()

    try:
        # Call API
        result = await api_client.predict(
            message=text,
            image_data=image_data,
            temperature=0.0,
            max_generation_steps=96,
            max_prompt_length=768,
        )

        # Build response
        prediction = result.get("prediction", "No prediction")
        has_image = result.get("has_image", False)

        response_content = (
            f"**Classification Result:**\n\n"
            f"{prediction}\n\n"
            f"{'Image included in analysis' if has_image else 'Text-only analysis'}"
        )

        # Update message
        msg.content = response_content
        await msg.update()

        # Display image if provided
        if images:
            img_msg = cl.Message(
                content="",
                elements=[
                    cl.Image(
                        name="uploaded_image",
                        path=images[0].path,
                    )
                ],
            )
            await img_msg.send()

    except httpx.ConnectError:
        msg.content = (
            "Cannot connect to API server. "
            "Make sure serve_fastapi.py is running on http://localhost:8000"
        )
        await msg.update()
    except httpx.HTTPStatusError as e:
        error_detail = "Unknown error"
        try:
            error_data = e.response.json()
            error_detail = error_data.get("detail", str(e))
        except Exception:
            error_detail = str(e)

        msg.content = f"API Error: {error_detail}"
        await msg.update()
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate using credentials from .env file."""
    valid_user = os.getenv("CHAINLIT_USER", "admin")
    valid_pass = os.getenv("CHAINLIT_PASSWORD", "password")

    if username == valid_user and password == valid_pass:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


def main_cli():
    """Run the Chainlit app."""
    parser = argparse.ArgumentParser(
        description="Serve GemmaEarth UI with Chainlit"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the inference API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Chainlit server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Chainlit server port (default: 8501)",
    )

    args = parser.parse_args()

    # Set environment variables for Chainlit
    os.environ["CHAINLIT_HOST"] = args.host
    os.environ["CHAINLIT_PORT"] = str(args.port)

    # Run Chainlit
    cl.run()


if __name__ == "__main__":
    main_cli()
