"""Configurable music generation entrypoint.

Default path targets a local SongGeneration-MLX script. The backend is selected
by capability-named environment variables so future swaps are config changes,
not code changes.
"""

from __future__ import annotations

import base64
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.analysis.audio_probe import probe_duration
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

_DEFAULT_LOCAL_SCRIPT_PATH = (
    Path.home() / "Documents/hf/repo/common-test/src/song/generate_songgeneration_mlx.sh"
)
_DEFAULT_MODEL_LABEL = "SongGeneration-v2-medium"
_SUPPORTED_OUTPUT_FORMATS = {"mp3", "wav", "flac", "aac", "opus"}
_DIRECT_OUTPUT_FORMATS = {"wav", "flac"}
_LOCAL_BACKEND_NAMES = {
    "local",
    "songgeneration",
    "songgeneration_mlx",
    "songgeneration-mlx",
}


class MusicGen(BaseTool):
    name = "music_gen"
    version = "0.3.0"
    tier = ToolTier.GENERATE
    capability = "music_generation"
    provider = "songgeneration_mlx"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.HYBRID

    dependencies = []  # checked dynamically from selected backend
    install_instructions = (
        "Preferred local backend (SongGeneration-MLX on Apple Silicon):\n"
        "  OM_MUSIC_PROVIDER=songgeneration_mlx\n"
        f"  OM_MUSIC_SCRIPT_PATH={_DEFAULT_LOCAL_SCRIPT_PATH}\n"
        "  OM_MUSIC_MODEL=SongGeneration-v2-medium\n"
        "Optional local overrides:\n"
        "  OM_MUSIC_MODEL_PATH=/path/to/SongGeneration-v2-medium\n"
        "  OM_MUSIC_SONGGEN_REPO=/path/to/SongGeneration-MLX\n"
        "  OM_MUSIC_DECODER_PYTHON=/path/to/.venv-decoder312/bin/python\n"
        "  OM_MUSIC_OFFICIAL_REPO=/path/to/SongGeneration\n"
        "  OM_MUSIC_DECODER_DEVICE=mps\n"
        "  OM_MUSIC_GEN_TYPE=bgm\n"
        "  OM_MUSIC_TOP_K=50\n"
        "  OM_MUSIC_TEMPERATURE=0.9\n"
        "  OM_MUSIC_RESPONSE_FORMAT=flac\n"
        "  OM_MUSIC_TIMEOUT_SECONDS=1200\n"
        "\n"
        "Legacy local gateway backend (omlx HTTP):\n"
        "  OM_MUSIC_PROVIDER=omlx\n"
        "  OM_MUSIC_BASE_URL=http://<host>:8999/v1\n"
        "  OM_MUSIC_API_KEY=your_local_key\n"
        "  OM_MUSIC_MODEL=SongGeneration-v2-medium\n"
        "Optional legacy endpoint override:\n"
        "  OM_MUSIC_ENDPOINT=/music\n"
        "\n"
        "Fallback cloud backend (ElevenLabs):\n"
        "  OM_MUSIC_PROVIDER=elevenlabs\n"
        "  ELEVENLABS_API_KEY=your_key_here"
    )

    agent_skills = ["music"]

    capabilities = [
        "generate_background_music",
        "generate_sfx",
    ]
    supports = {
        "configurable_endpoint": True,
        "local_model": True,
        "duration_control": True,
        "lyrics_control": True,
        "local_backend_swapping": True,
    }
    best_for = [
        "local SongGeneration-MLX generation with future-safe env-based swaps",
        "background music generation without changing pipeline call sites",
        "switching between local script, local gateway, and cloud backends without code changes",
    ]
    fallback_tools = ["suno_music", "pixabay_music", "freesound_music"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Music description (mood, genre, instruments, tempo)",
            },
            "duration_seconds": {
                "type": "number",
                "minimum": 3,
                "maximum": 600,
                "description": (
                    "Target duration in seconds. Must match the approved video runtime; "
                    "silent defaults are not permitted."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override. Defaults to OM_MUSIC_MODEL.",
            },
            "lyrics": {
                "type": "string",
                "description": "Optional structured lyrics. Required for vocal or mixed generation.",
            },
            "gen_type": {
                "type": "string",
                "enum": ["mixed", "vocal", "bgm"],
                "description": "SongGeneration output type. Defaults to bgm.",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "description": "SongGeneration sampling top-k.",
            },
            "temperature": {
                "type": "number",
                "minimum": 0,
                "description": "SongGeneration sampling temperature.",
            },
            "decoder_device": {
                "type": "string",
                "enum": ["mps", "cpu", "cuda"],
                "description": "SongGeneration decoder device.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Optional timeout for local SongGeneration execution.",
            },
            "response_format": {
                "type": "string",
                "enum": ["mp3", "wav", "flac", "aac", "opus"],
                "default": "flac",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=8192, vram_mb=6144, disk_mb=4096, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=0, retryable_errors=[])
    idempotency_key_fields = [
        "prompt",
        "duration_seconds",
        "model",
        "lyrics",
        "gen_type",
        "top_k",
        "temperature",
        "response_format",
    ]
    side_effects = [
        "writes audio file to output_path",
        "executes the configured music backend",
        "may load a local MLX model and spawn a decoder subprocess",
    ]
    user_visible_verification = [
        "Listen to generated music for mood and quality",
    ]

    def _backend_name(self) -> str:
        explicit = self._normalize_backend_name(
            (os.environ.get("OM_MUSIC_PROVIDER") or "").strip().lower()
        )
        if explicit:
            return explicit
        if os.environ.get("OM_MUSIC_BASE_URL"):
            return "omlx"
        if os.environ.get("ELEVENLABS_API_KEY"):
            return "elevenlabs"
        return "songgeneration_mlx"

    def get_status(self) -> ToolStatus:
        backend = self._backend_name()
        if backend == "songgeneration_mlx":
            if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
                return ToolStatus.UNAVAILABLE
            if shutil.which("bash") is None or shutil.which("uv") is None:
                return ToolStatus.UNAVAILABLE
            if not self._local_script_path().is_file():
                return ToolStatus.UNAVAILABLE
            if not self._configured_songgen_repo_path().exists():
                return ToolStatus.UNAVAILABLE
            if not self._configured_decoder_python_path().exists():
                return ToolStatus.UNAVAILABLE
            if not self._configured_official_repo_path().exists():
                return ToolStatus.UNAVAILABLE
            model_path = os.environ.get("OM_MUSIC_MODEL_PATH")
            if model_path and not Path(model_path).expanduser().exists():
                return ToolStatus.UNAVAILABLE
            return ToolStatus.AVAILABLE
        if backend == "elevenlabs":
            return ToolStatus.AVAILABLE if os.environ.get("ELEVENLABS_API_KEY") else ToolStatus.UNAVAILABLE
        if os.environ.get("OM_MUSIC_BASE_URL") and os.environ.get("OM_MUSIC_API_KEY"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        duration = inputs.get("duration_seconds")
        if duration is None:
            return 0.0

        if self._backend_name() == "elevenlabs":
            return round(float(duration) / 30 * 0.05, 4)

        try:
            cost_per_second = float(os.environ.get("OM_MUSIC_COST_PER_SECOND", "0"))
        except ValueError:
            cost_per_second = 0.0
        return round(float(duration) * cost_per_second, 4)

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        start = time.time()
        try:
            backend = self._backend_name()
            if backend == "elevenlabs":
                result = self._generate_elevenlabs(inputs, os.environ.get("ELEVENLABS_API_KEY"))
            elif backend == "omlx":
                result = self._generate_omlx(inputs)
            elif backend == "songgeneration_mlx":
                result = self._generate_songgeneration_mlx(inputs)
            else:
                return ToolResult(
                    success=False,
                    error=(
                        f"Unsupported music backend {backend!r}. "
                        "Set OM_MUSIC_PROVIDER to songgeneration_mlx, omlx, or elevenlabs."
                    ),
                )
        except Exception as exc:
            return ToolResult(success=False, error=f"Music generation failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _require_duration(self, inputs: dict[str, Any]) -> float:
        duration = inputs.get("duration_seconds")
        if duration is None:
            raise ValueError(
                "music_gen: duration_seconds is required. "
                "Derive it from the approved target runtime in the script/proposal. "
                "Silent defaults are not permitted."
            )
        return float(duration)

    def _generate_elevenlabs(self, inputs: dict[str, Any], api_key: str | None) -> ToolResult:
        if not api_key:
            return ToolResult(
                success=False,
                error="No ElevenLabs API key. " + self.install_instructions,
            )

        import requests

        prompt = inputs["prompt"]
        duration = self._require_duration(inputs)
        url = "https://api.elevenlabs.io/v1/music"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "music_length_ms": int(duration * 1000),
        }

        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()

        fmt, output_path = self._resolve_output_target(inputs, default_format="mp3")
        self._write_music_response(response, output_path)
        audio_duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": "elevenlabs",
                "model": "elevenlabs/music",
                "prompt": prompt,
                "duration_seconds": duration,
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "duration_probe_warning": None if audio_duration else "ffprobe unavailable; audio_duration_seconds is None",
                "output": str(output_path),
                "format": fmt,
            },
            artifacts=[str(output_path)],
            model="elevenlabs/music",
        )

    def _generate_songgeneration_mlx(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error="Local SongGeneration backend not available. " + self.install_instructions,
            )

        prompt = str(inputs["prompt"]).strip()
        if not prompt:
            raise ValueError("prompt 不能为空。")

        duration = self._require_duration(inputs)
        gen_type = self._resolve_gen_type(inputs)
        lyrics, lyrics_source = self._resolve_lyrics(inputs, duration, gen_type)
        top_k = self._resolve_positive_int(
            inputs.get("top_k") or os.environ.get("OM_MUSIC_TOP_K"),
            "top_k",
            50,
        )
        temperature = self._resolve_non_negative_float(
            inputs.get("temperature") or os.environ.get("OM_MUSIC_TEMPERATURE"),
            "temperature",
            0.9,
        )
        decoder_device = self._resolve_decoder_device(inputs)
        timeout_seconds = self._resolve_positive_int(
            inputs.get("timeout_seconds") or os.environ.get("OM_MUSIC_TIMEOUT_SECONDS"),
            "timeout_seconds",
            1200,
        )
        requested_format, output_path = self._resolve_output_target(
            inputs,
            default_format="flac",
        )
        if requested_format not in _DIRECT_OUTPUT_FORMATS and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. SongGeneration 本地脚本原生输出 wav/flac；"
                f"若要输出 {requested_format}，需要先安装 ffmpeg。"
            )

        temp_dir = tempfile.TemporaryDirectory(prefix="music-gen-")
        try:
            tokens_output = Path(temp_dir.name) / "songgeneration_tokens.npz"
            raw_output_path = (
                output_path
                if requested_format in _DIRECT_OUTPUT_FORMATS
                else Path(temp_dir.name) / "songgeneration-output.flac"
            )

            env = os.environ.copy()
            env["SONGGEN_REPO"] = str(self._configured_songgen_repo_path())
            env["SONGGEN_DECODER_PYTHON"] = str(self._configured_decoder_python_path())
            env["SONGGEN_OUTPUT_AUDIO"] = str(raw_output_path)

            command = [
                shutil.which("bash") or "bash",
                str(self._local_script_path()),
                "--duration",
                str(duration),
                "--top-k",
                str(top_k),
                "--temperature",
                str(temperature),
                "--lyrics",
                lyrics,
                "--description",
                prompt,
                "--decoder-device",
                decoder_device,
                "--gen-type",
                gen_type,
                "--tokens-output",
                str(tokens_output),
                "--official-repo",
                str(self._configured_official_repo_path()),
            ]

            model_path = self._resolve_local_model_path(inputs)
            if model_path is not None:
                command.extend(["--model", str(model_path)])

            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                env=env,
            )
            if proc.returncode != 0:
                details = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"SongGeneration local run failed (exit {proc.returncode}): {details[-2000:]}"
                )

            if not raw_output_path.is_file():
                raise RuntimeError(
                    f"SongGeneration local run produced no audio file: {raw_output_path}"
                )

            if requested_format not in _DIRECT_OUTPUT_FORMATS:
                self._transcode_audio(raw_output_path, output_path, requested_format)

            audio_duration = probe_duration(output_path)
            model_name = str(
                inputs.get("model")
                or os.environ.get("OM_MUSIC_MODEL")
                or _DEFAULT_MODEL_LABEL
            )
            return ToolResult(
                success=True,
                data={
                    "provider": "songgeneration_mlx",
                    "backend": "songgeneration_mlx",
                    "model": model_name,
                    "prompt": prompt,
                    "lyrics": lyrics,
                    "lyrics_source": lyrics_source,
                    "gen_type": gen_type,
                    "duration_seconds": duration,
                    "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                    "top_k": top_k,
                    "temperature": temperature,
                    "decoder_device": decoder_device,
                    "output": str(output_path),
                    "format": requested_format,
                },
                artifacts=[str(output_path)],
                model=model_name,
            )
        except subprocess.TimeoutExpired as exc:
            details = ((exc.stderr or "") or (exc.stdout or "")).strip()
            raise RuntimeError(
                f"SongGeneration local run timed out after {timeout_seconds}s. {details[-1000:]}"
            )
        finally:
            temp_dir.cleanup()

    def _generate_omlx(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        base_url = (os.environ.get("OM_MUSIC_BASE_URL") or "").rstrip("/")
        api_key = os.environ.get("OM_MUSIC_API_KEY")
        if not base_url or not api_key:
            return ToolResult(
                success=False,
                error="OM_MUSIC_BASE_URL/OM_MUSIC_API_KEY not set. " + self.install_instructions,
            )

        prompt = inputs["prompt"]
        duration = self._require_duration(inputs)
        model = inputs.get("model") or os.environ.get("OM_MUSIC_MODEL", "SongGeneration-v2-medium")
        response_format, output_path = self._resolve_output_target(inputs, default_format="mp3")
        endpoint = os.environ.get("OM_MUSIC_ENDPOINT", "/music")
        endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "duration_seconds": duration,
            "response_format": response_format,
        }

        response = requests.post(
            f"{base_url}{endpoint}",
            headers=headers,
            json=payload,
            timeout=int(os.environ.get("OM_MUSIC_TIMEOUT_SECONDS", "600")),
        )
        response.raise_for_status()

        self._write_music_response(response, output_path)
        audio_duration = probe_duration(output_path)

        return ToolResult(
            success=True,
            data={
                "provider": "omlx",
                "base_url": base_url,
                "endpoint": endpoint,
                "model": model,
                "prompt": prompt,
                "duration_seconds": duration,
                "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                "duration_probe_warning": None if audio_duration else "ffprobe unavailable; audio_duration_seconds is None",
                "output": str(output_path),
                "format": output_path.suffix.lstrip(".") or response_format,
            },
            artifacts=[str(output_path)],
            model=model,
        )

    @staticmethod
    def _normalize_backend_name(raw: str) -> str:
        if raw in _LOCAL_BACKEND_NAMES:
            return "songgeneration_mlx"
        return raw

    @staticmethod
    def _local_script_path() -> Path:
        return Path(
            os.environ.get("OM_MUSIC_SCRIPT_PATH") or _DEFAULT_LOCAL_SCRIPT_PATH
        ).expanduser()

    @classmethod
    def _common_test_root(cls) -> Path:
        script_path = cls._local_script_path()
        parents = script_path.parents
        return parents[2] if len(parents) > 2 else script_path.parent

    @classmethod
    def _configured_songgen_repo_path(cls) -> Path:
        return Path(
            os.environ.get("OM_MUSIC_SONGGEN_REPO")
            or cls._common_test_root() / "ref" / "SongGeneration-MLX"
        ).expanduser()

    @classmethod
    def _configured_decoder_python_path(cls) -> Path:
        return Path(
            os.environ.get("OM_MUSIC_DECODER_PYTHON")
            or cls._configured_songgen_repo_path() / ".venv-decoder312" / "bin" / "python"
        ).expanduser()

    @classmethod
    def _configured_official_repo_path(cls) -> Path:
        return Path(
            os.environ.get("OM_MUSIC_OFFICIAL_REPO")
            or cls._configured_songgen_repo_path() / "third_party" / "SongGeneration"
        ).expanduser()

    def _resolve_output_target(
        self,
        inputs: dict[str, Any],
        default_format: str,
    ) -> tuple[str, Path]:
        explicit_format = (
            inputs.get("response_format")
            or inputs.get("format")
            or inputs.get("output_format")
            or os.environ.get("OM_MUSIC_RESPONSE_FORMAT")
        )
        output_path_raw = inputs.get("output_path")
        if output_path_raw:
            output_path = Path(str(output_path_raw)).expanduser()
        else:
            output_path = Path(f"music_output.{explicit_format or default_format}")

        suffix = output_path.suffix.lstrip(".").lower()
        if explicit_format in (None, ""):
            requested_format = suffix if suffix in _SUPPORTED_OUTPUT_FORMATS else default_format
        else:
            requested_format = str(explicit_format).strip().lower()
            if requested_format == "wav" and suffix in _SUPPORTED_OUTPUT_FORMATS:
                requested_format = suffix

        if requested_format not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported response_format {requested_format!r}. "
                f"Supported: {sorted(_SUPPORTED_OUTPUT_FORMATS)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return requested_format, output_path

    def _resolve_local_model_path(self, inputs: dict[str, Any]) -> Path | None:
        explicit = inputs.get("model")
        if explicit:
            candidate = Path(str(explicit)).expanduser()
            if candidate.exists():
                return candidate.resolve()
            if any(sep in str(explicit) for sep in ("/", "\\")) or str(explicit).startswith(("~", ".")):
                raise FileNotFoundError(f"Local SongGeneration model path not found: {candidate}")

        env_path = os.environ.get("OM_MUSIC_MODEL_PATH")
        if env_path:
            candidate = Path(env_path).expanduser()
            if not candidate.exists():
                raise FileNotFoundError(f"Local SongGeneration model path not found: {candidate}")
            return candidate.resolve()

        return None

    def _resolve_lyrics(
        self,
        inputs: dict[str, Any],
        duration: float,
        gen_type: str,
    ) -> tuple[str, str]:
        raw = inputs.get("lyrics")
        if raw is not None:
            lyrics = str(raw).strip()
            if not lyrics:
                raise ValueError("lyrics 不能为空。")
            return lyrics, "user"

        if gen_type != "bgm":
            raise ValueError("gen_type 为 mixed 或 vocal 时必须显式提供 lyrics。")

        return self._default_instrumental_lyrics(duration), "instrumental_template"

    @staticmethod
    def _default_instrumental_lyrics(duration: float) -> str:
        sections = ["[intro-short]"]
        remaining = max(duration - 6.0, 0.0)
        while remaining >= 8.0:
            if remaining >= 15.0:
                sections.append("[inst-medium]")
                remaining -= 15.0
            else:
                sections.append("[inst-short]")
                remaining -= 8.0
        sections.append("[outro-short]")
        return "; ".join(sections)

    @staticmethod
    def _resolve_gen_type(inputs: dict[str, Any]) -> str:
        value = str(
            inputs.get("gen_type")
            or os.environ.get("OM_MUSIC_GEN_TYPE")
            or "bgm"
        ).strip().lower()
        if value not in {"mixed", "vocal", "bgm"}:
            raise ValueError("gen_type 只支持 mixed、vocal、bgm。")
        return value

    @staticmethod
    def _resolve_decoder_device(inputs: dict[str, Any]) -> str:
        value = str(
            inputs.get("decoder_device")
            or os.environ.get("OM_MUSIC_DECODER_DEVICE")
            or "mps"
        ).strip().lower()
        if value not in {"mps", "cpu", "cuda"}:
            raise ValueError("decoder_device 只支持 mps、cpu、cuda。")
        return value

    @staticmethod
    def _resolve_positive_int(raw: Any, key: str, default: int) -> int:
        if raw in (None, ""):
            return default
        value = int(raw)
        if value < 1:
            raise ValueError(f"{key} 必须大于等于 1。")
        return value

    @staticmethod
    def _resolve_non_negative_float(raw: Any, key: str, default: float) -> float:
        if raw in (None, ""):
            return default
        value = float(raw)
        if value < 0:
            raise ValueError(f"{key} 不能小于 0。")
        return value

    def _write_music_response(self, response: Any, output_path: Path) -> None:
        content_type = (response.headers.get("content-type") or "").lower()
        if content_type.startswith("audio/"):
            output_path.write_bytes(response.content)
            return

        data = response.json()
        failures: list[str] = []
        for candidate in self._iter_audio_candidates(data):
            if not isinstance(candidate, str):
                continue
            if candidate.startswith(("http://", "https://")):
                try:
                    self._download_to_path(candidate, output_path)
                    return
                except Exception as exc:
                    failures.append(f"url({candidate[:60]}): {exc}")
                    continue
            try:
                output_path.write_bytes(base64.b64decode(candidate))
                return
            except Exception as exc:
                failures.append(f"base64({candidate[:60]}): {exc}")
                continue

        raise RuntimeError(
            "Configured music backend returned no downloadable audio payload. "
            f"Candidates tried: {len(failures)}. Details: {'; '.join(failures) or 'none'}"
        )

    def _iter_audio_candidates(self, data: Any) -> list[str]:
        candidates: list[str] = []
        if isinstance(data, dict):
            for key in ("audio", "audio_base64", "b64_json", "url", "audio_url"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)

            nested_items = []
            for key in ("data", "output", "outputs", "result", "results", "music"):
                value = data.get(key)
                if isinstance(value, list):
                    nested_items.extend(value)
                elif isinstance(value, dict):
                    nested_items.append(value)

            for item in nested_items:
                if not isinstance(item, dict):
                    continue
                for key in ("audio", "audio_base64", "b64_json", "url", "audio_url"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        candidates.append(value)
        return candidates

    @staticmethod
    def _download_to_path(url: str, output_path: Path) -> None:
        import requests

        download = requests.get(url, timeout=180)
        download.raise_for_status()
        output_path.write_bytes(download.content)

    @staticmethod
    def _transcode_audio(input_path: Path, output_path: Path, output_format: str) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found on PATH. SongGeneration 本地脚本原生输出 wav/flac；"
                "若要输出 mp3/aac/opus，需要先安装 ffmpeg。"
            )

        cmd = [ffmpeg, "-y", "-i", str(input_path)]
        if output_format == "opus":
            cmd.extend(["-c:a", "libopus"])
        cmd.append(str(output_path))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg transcode failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
