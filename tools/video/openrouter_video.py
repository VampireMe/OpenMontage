"""OpenRouter video generation provider.

Uses OpenRouter's async video generation API with a configurable model, default
ing to ByteDance Seedance 2.0. The provider is selector-friendly: switching the
model later is an env change, not a selector edit.
"""

from __future__ import annotations

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


class OpenRouterVideo(BaseTool):
    name = "openrouter_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "openrouter"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set OPENROUTER_API_KEY to your OpenRouter API key.\n"
        "Optional overrides:\n"
        "  OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n"
        "  OM_VIDEO_OPENROUTER_MODEL=bytedance/seedance-2.0\n"
        "  OM_VIDEO_PREFERRED_PROVIDER=openrouter"
    )
    agent_skills = ["ai-video-gen", "seedance-2-0"]

    capabilities = ["text_to_video", "provider_selection"]
    supports = {
        "text_to_video": True,
        "aspect_ratio": True,
        "duration_control": True,
        "seed": True,
        "configurable_model": True,
    }
    best_for = [
        "OpenRouter-routed video generation with a config-driven model choice",
        "Seedance 2.0 through OpenRouter without hardcoding selector logic",
        "teams that want one gateway key and future model swaps via env vars",
    ]
    not_good_for = [
        "offline generation",
        "image-to-video until an OpenRouter-backed provider path is explicitly added",
    ]
    fallback_tools = ["seedance_video", "veo_video", "kling_video", "minimax_video"]
    quality_score = 0.92

    setup_offer = {
        "env_var": "OPENROUTER_API_KEY",
        "docs_url": "https://openrouter.ai/settings/keys",
        "default_model": "bytedance/seedance-2.0",
    }

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["text_to_video"],
                "default": "text_to_video",
            },
            "model": {
                "type": "string",
                "description": "Optional override. Defaults to OM_VIDEO_OPENROUTER_MODEL.",
            },
            "duration": {
                "type": "integer",
                "minimum": 1,
                "maximum": 60,
                "default": 5,
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                "default": "16:9",
            },
            "resolution": {
                "type": "string",
                "description": "Optional resolution hint passed through to the provider.",
            },
            "seed": {
                "type": "integer",
                "description": "Optional seed for reproducibility when the provider supports it.",
            },
            "output_path": {"type": "string"},
            "poll_interval_seconds": {"type": "integer", "minimum": 2, "default": 5},
            "timeout_seconds": {"type": "integer", "minimum": 30, "default": 900},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "duration", "aspect_ratio", "seed"]
    side_effects = ["writes video file to output_path", "calls OpenRouter's video API"]
    user_visible_verification = [
        "Watch generated clip for motion coherence and prompt fidelity"
    ]

    def get_info(self) -> dict[str, Any]:
        info = super().get_info()
        info["setup_offer"] = self.setup_offer
        return info

    def get_status(self) -> ToolStatus:
        if os.environ.get("OPENROUTER_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        try:
            per_second = float(os.environ.get("OM_VIDEO_OPENROUTER_COST_PER_SECOND", "0"))
        except ValueError:
            per_second = 0.0
        return round(per_second * int(inputs.get("duration", 5)), 4)

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return 90.0 + int(inputs.get("duration", 5)) * 6.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="OPENROUTER_API_KEY not set. " + self.install_instructions,
            )

        import requests

        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        model = inputs.get("model") or os.environ.get("OM_VIDEO_OPENROUTER_MODEL", "bytedance/seedance-2.0")
        payload: dict[str, Any] = {
            "model": model,
            "prompt": inputs["prompt"],
        }
        if inputs.get("duration") is not None:
            payload["duration"] = int(inputs.get("duration", 5))
        if inputs.get("aspect_ratio"):
            payload["aspect_ratio"] = inputs["aspect_ratio"]
        if inputs.get("resolution"):
            payload["resolution"] = inputs["resolution"]
        if inputs.get("seed") is not None:
            payload["seed"] = inputs["seed"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        start = time.time()
        try:
            create_resp = requests.post(
                f"{base_url}/videos",
                headers=headers,
                json=payload,
                timeout=60,
            )
            create_resp.raise_for_status()
            create_data = create_resp.json()
            generation_id = self._extract_generation_id(create_data)
            if not generation_id:
                raise RuntimeError(f"OpenRouter response missing generation id: {create_data}")

            poll_interval = int(inputs.get("poll_interval_seconds", 5))
            deadline = time.time() + int(inputs.get("timeout_seconds", 900))
            status_data = create_data
            while time.time() < deadline:
                status_resp = requests.get(
                    f"{base_url}/videos/{generation_id}",
                    headers={"Authorization": headers["Authorization"]},
                    timeout=30,
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = (self._extract_status(status_data) or "").lower()
                if status in {"completed", "succeeded", "done"}:
                    break
                if status in {"failed", "cancelled", "canceled", "error", "expired"}:
                    detail = self._extract_error(status_data) or status
                    return ToolResult(success=False, error=f"OpenRouter video generation failed: {detail}")
                time.sleep(poll_interval)
            else:
                return ToolResult(success=False, error="OpenRouter video generation timed out.")

            video_url = self._extract_video_url(status_data)
            if not video_url:
                raise RuntimeError(f"OpenRouter response missing video URL: {status_data}")

            video_resp = requests.get(video_url, timeout=180)
            video_resp.raise_for_status()
            output_path = Path(inputs.get("output_path", "openrouter_video.mp4"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(video_resp.content)

        except Exception as exc:
            return ToolResult(success=False, error=f"OpenRouter video generation failed: {exc}")

        from tools.video._shared import probe_output

        probed = probe_output(output_path)
        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "prompt": inputs["prompt"],
                "duration": payload.get("duration"),
                "aspect_ratio": payload.get("aspect_ratio"),
                "resolution": payload.get("resolution"),
                "seed": payload.get("seed"),
                "generation_id": generation_id,
                "output": str(output_path),
                "output_path": str(output_path),
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )

    @staticmethod
    def _extract_generation_id(data: dict[str, Any]) -> str | None:
        for key in ("id", "generation_id", "video_id"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("id", "generation_id", "video_id"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _extract_status(data: dict[str, Any]) -> str | None:
        for key in ("status", "state"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("status", "state"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _extract_error(data: dict[str, Any]) -> str | None:
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("error", "message", "detail"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _extract_video_url(data: dict[str, Any]) -> str | None:
        def _from_mapping(mapping: dict[str, Any]) -> str | None:
            for key in ("video_url", "url"):
                value = mapping.get(key)
                if isinstance(value, str) and value:
                    return value
            video = mapping.get("video")
            if isinstance(video, dict):
                for key in ("url", "video_url"):
                    value = video.get(key)
                    if isinstance(value, str) and value:
                        return value
            for key in ("output", "outputs", "data", "result", "results"):
                nested = mapping.get(key)
                if isinstance(nested, dict):
                    found = _from_mapping(nested)
                    if found:
                        return found
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict):
                            found = _from_mapping(item)
                            if found:
                                return found
            return None

        return _from_mapping(data)
