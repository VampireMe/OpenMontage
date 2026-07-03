"""Generic OpenAI-compatible image generation provider.

Talks to any backend that implements the OpenAI /v1/images/generations
contract (remote gateways, self-hosted servers, etc.) via a configurable
base_url. Swapping backends is a .env change
(OM_IMAGE_BASE_URL/OM_IMAGE_API_KEY/OM_IMAGE_MODEL), not a code change.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class OpenAICompatibleImage(BaseTool):
    name = "openai_compatible_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "openai_compatible"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:OM_IMAGE_BASE_URL", "env:OM_IMAGE_API_KEY"]
    install_instructions = (
        "Set these to any OpenAI-compatible image gateway, e.g.:\n"
        "  OM_IMAGE_BASE_URL=https://www.packyapi.com/v1\n"
        "  OM_IMAGE_API_KEY=your_key_here\n"
        "  OM_IMAGE_MODEL=gpt-image-2\n"
        "Works with any backend implementing POST {base_url}/images/generations."
    )
    fallback_tools = ["openai_image", "flux_image", "google_imagen", "recraft_image"]
    agent_skills = ["flux-best-practices"]

    capabilities = ["generate_image", "generate_illustration", "text_to_image"]
    supports = {
        "complex_instructions": True,
        "text_in_image": True,
        "multiple_outputs": True,
        "configurable_endpoint": True,
    }
    best_for = [
        "swapping image generation backends without code changes",
        "third-party OpenAI-compatible image gateways",
    ]
    not_good_for = ["backends that don't implement the OpenAI images.generate shape"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "description": "Model name understood by the backend. Defaults to OM_IMAGE_MODEL.",
            },
            "size": {
                "type": "string",
                "default": "1024x1024",
                "description": "Backend-specific size string, e.g. 1024x1024.",
            },
            "quality": {
                "type": "string",
                "default": "high",
                "description": "Backend-specific quality tier, if supported.",
            },
            "output_format": {
                "type": "string",
                "enum": ["png", "jpeg", "webp"],
                "default": "png",
            },
            "n": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "size", "quality", "model"]
    side_effects = ["writes image file to output_path", "calls a configured OpenAI-compatible image endpoint"]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def get_status(self) -> ToolStatus:
        if os.environ.get("OM_IMAGE_BASE_URL") and os.environ.get("OM_IMAGE_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        try:
            per_image = float(os.environ.get("OM_IMAGE_COST_PER_IMAGE", "0"))
        except ValueError:
            per_image = 0.0
        return per_image * inputs.get("n", 1)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        base_url = os.environ.get("OM_IMAGE_BASE_URL")
        api_key = os.environ.get("OM_IMAGE_API_KEY")
        if not base_url or not api_key:
            return ToolResult(success=False, error="OM_IMAGE_BASE_URL/OM_IMAGE_API_KEY not set. " + self.install_instructions)

        from openai import OpenAI

        start = time.time()
        client = OpenAI(base_url=base_url, api_key=api_key)
        model = inputs.get("model") or os.environ.get("OM_IMAGE_MODEL", "gpt-image-2")
        prompt = inputs["prompt"]
        size = inputs.get("size", "1024x1024")
        quality = inputs.get("quality", "high")
        output_format = inputs.get("output_format", "png")
        n = inputs.get("n", 1)

        try:
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                output_format=output_format,
                n=n,
            )
            image_data = base64.b64decode(response.data[0].b64_json)
            output_path = Path(inputs.get("output_path", f"generated_image.{output_format}"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_data)
        except Exception as e:
            return ToolResult(success=False, error=f"OpenAI-compatible image generation failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "base_url": base_url,
                "model": model,
                "prompt": prompt,
                "output": str(output_path),
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
