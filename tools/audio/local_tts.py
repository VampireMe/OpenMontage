"""Local Python TTS provider with a backend-selected execution path.

The current backend is OmniVoice on Apple Silicon via mlx-audio. The contract
stays capability-named (`OM_TTS_*`) so future local backends can be swapped
without changing selector call sites.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import platform
import shutil
import subprocess
import tempfile
import time
import warnings
import wave
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


_DEFAULT_MODEL = "mlx-community/OmniVoice-bf16"
_DEFAULT_LANGUAGE = "zh"
_DEFAULT_NUM_STEPS = 32
_DEFAULT_GUIDANCE_SCALE = 2.0
_DEFAULT_REF_AUDIO_MAX_DURATION = 15.0
_SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3", "flac", "aac", "opus"}
_TOKENIZER_SOURCE_ID = "k2-fsa/OmniVoice"
_TOKENIZER_FILES = [
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
]
_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
_TOKENIZER_REPO_CACHE = _HF_HOME / "hub" / "models--k2-fsa--OmniVoice"
_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "english": "en",
    "en-us": "en",
    "japanese": "ja",
}


class LocalTTS(BaseTool):
    name = "local_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "local_python"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = []
    install_instructions = (
        "Preferred local backend (OmniVoice on Apple Silicon):\n"
        "  pip install -r requirements-gpu.txt\n"
        "  OM_TTS_PROVIDER=omnivoice\n"
        "  OM_TTS_MODEL=mlx-community/OmniVoice-bf16\n"
        "Optional defaults:\n"
        "  OM_TTS_LANGUAGE=zh\n"
        "  OM_TTS_INSTRUCT=female, young adult\n"
        "  OM_TTS_REF_AUDIO=/path/to/reference.wav\n"
        "  OM_TTS_REF_TEXT=参考音频对应的文字稿\n"
        "  OM_TTS_MEMORY_LIMIT_GB=10\n"
        "  OM_TTS_CACHE_LIMIT_GB=2\n"
        "This backend runs Python locally and does not use an OpenAI-compatible HTTP endpoint."
    )
    fallback_tools = ["elevenlabs_tts", "openai_tts", "doubao_tts", "google_tts", "piper_tts"]
    agent_skills = ["text-to-speech"]

    capabilities = [
        "text_to_speech",
        "voice_design",
        "voice_cloning",
        "offline_generation",
    ]
    supports = {
        "voice_cloning": True,
        "voice_design": True,
        "multilingual": True,
        "offline": True,
        "native_audio": True,
        "local_backend_swapping": True,
    }
    best_for = [
        "local Apple Silicon narration with OmniVoice",
        "voice design and voice cloning without an HTTP TTS gateway",
        "future-safe local backend swaps behind capability-named env vars",
    ]
    not_good_for = [
        "non-Apple Silicon machines without MLX support",
        "environments missing mlx-audio model runtime dependencies",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize."},
            "model": {
                "type": "string",
                "description": "Optional model override. Defaults to OM_TTS_MODEL.",
            },
            "voice": {
                "type": "string",
                "description": "Reserved future speaker hint. OmniVoice currently ignores fixed voice IDs.",
            },
            "language": {
                "type": "string",
                "description": "Language tag such as zh, en, ja. Defaults to OM_TTS_LANGUAGE or zh.",
            },
            "instruct": {
                "type": "string",
                "description": "OmniVoice voice-design prompt.",
            },
            "instructions": {
                "type": "string",
                "description": "Alias for instruct.",
            },
            "ref_audio": {
                "type": "string",
                "description": "Local reference audio path for voice cloning.",
            },
            "ref_text": {
                "type": "string",
                "description": "Transcript for ref_audio when available.",
            },
            "ref_audio_max_duration": {
                "type": "number",
                "minimum": 0.1,
                "description": "Max duration in seconds for reference audio when voice cloning.",
            },
            "duration_seconds": {
                "type": "number",
                "minimum": 0.1,
                "description": "Optional target duration upper bound in seconds.",
            },
            "num_steps": {
                "type": "integer",
                "minimum": 1,
                "description": "Diffusion steps for OmniVoice.",
            },
            "guidance_scale": {
                "type": "number",
                "minimum": 0.0,
                "description": "Classifier-free guidance scale.",
            },
            "memory_limit_gb": {
                "type": "number",
                "minimum": 0.1,
                "description": "Optional MLX Metal memory cap in GB.",
            },
            "cache_limit_gb": {
                "type": "number",
                "minimum": 0.0,
                "description": "Optional MLX Metal cache cap in GB.",
            },
            "response_format": {
                "type": "string",
                "default": "wav",
                "enum": sorted(_SUPPORTED_OUTPUT_FORMATS),
            },
            "format": {
                "type": "string",
                "default": "wav",
                "enum": sorted(_SUPPORTED_OUTPUT_FORMATS),
                "description": "Backward-compatible alias for response_format.",
            },
            "output_format": {
                "type": "string",
                "default": "wav",
                "enum": sorted(_SUPPORTED_OUTPUT_FORMATS),
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=4, ram_mb=6144, vram_mb=6144, disk_mb=4096, network_required=False
    )
    retry_policy = RetryPolicy(max_retries=0, retryable_errors=[])
    idempotency_key_fields = [
        "text",
        "model",
        "language",
        "instruct",
        "instructions",
        "ref_audio",
        "ref_text",
        "duration_seconds",
        "num_steps",
        "guidance_scale",
        "response_format",
    ]
    side_effects = [
        "writes audio file to output_path",
        "loads a local MLX TTS model into memory for the duration of the call",
    ]
    user_visible_verification = [
        "Listen to generated audio for intelligibility, tone, and voice match",
    ]

    def _backend_name(self) -> str:
        explicit = (os.environ.get("OM_TTS_PROVIDER") or "").strip().lower()
        return explicit or "omnivoice"

    def get_status(self) -> ToolStatus:
        if self._backend_name() != "omnivoice":
            return ToolStatus.UNAVAILABLE
        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            return ToolStatus.UNAVAILABLE
        for module_name in ("mlx", "mlx_audio", "huggingface_hub", "numpy"):
            if importlib.util.find_spec(module_name) is None:
                return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        try:
            return float(os.environ.get("OM_TTS_COST_PER_CHAR", "0")) * len(inputs.get("text", ""))
        except ValueError:
            return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        backend = self._backend_name()
        if backend != "omnivoice":
            return ToolResult(
                success=False,
                error=(
                    f"Unsupported local TTS backend {backend!r}. "
                    "Set OM_TTS_PROVIDER=omnivoice."
                ),
            )
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="Local TTS backend not available. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate_omnivoice(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Local TTS failed: {exc}")

        result.duration_seconds = round(time.time() - start, 2)
        result.cost_usd = self.estimate_cost(inputs)
        return result

    def _generate_omnivoice(self, inputs: dict[str, Any]) -> ToolResult:
        from huggingface_hub import snapshot_download
        import mlx.core as mx
        import numpy as np
        from mlx_audio.codec.models.higgs_audio.higgs_audio import HiggsAudioTokenizer
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt
        from mlx_audio.tts.utils import load_model

        from tools.analysis.audio_probe import probe_duration

        text = str(inputs["text"]).strip()
        if not text:
            raise ValueError("text 不能为空。")

        ref_audio = self._resolve_ref_audio(inputs)
        ref_text = self._resolve_ref_text(inputs)
        if ref_text and ref_audio is None:
            raise ValueError("传入 ref_text 时必须同时提供 ref_audio。")

        requested_format, output_path = self._resolve_output_target(inputs)
        if requested_format != "wav" and shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. OmniVoice 原生只输出 wav；"
                f"若要输出 {requested_format}，需要先安装 ffmpeg。"
            )
        temp_wav_path = output_path
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if requested_format != "wav":
            temp_dir = tempfile.TemporaryDirectory(prefix="local-tts-")
            temp_wav_path = Path(temp_dir.name) / "omnivoice-output.wav"

        model = None
        result = None
        ref_tokens = None

        try:
            self._apply_runtime_limits(mx, inputs)
            self._cleanup_incomplete_tokenizer_downloads()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Could not load audio tokenizer: .*",
                )
                model = load_model(self._resolve_model(inputs))

            if getattr(model, "audio_tokenizer", None) is None:
                tokenizer_dir = snapshot_download(
                    _TOKENIZER_SOURCE_ID,
                    allow_patterns=_TOKENIZER_FILES,
                )
                model.audio_tokenizer = HiggsAudioTokenizer.from_pretrained(tokenizer_dir)

            if ref_audio is not None:
                ref_tokens = create_voice_clone_prompt(
                    str(ref_audio),
                    tokenizer=model.audio_tokenizer,
                    ref_text=ref_text,
                    max_duration_s=self._resolve_optional_positive_float(
                        inputs.get("ref_audio_max_duration"),
                        "ref_audio_max_duration",
                        _DEFAULT_REF_AUDIO_MAX_DURATION,
                    ),
                )

            result = next(
                model.generate(
                    text=text,
                    duration_s=self._resolve_optional_duration(inputs),
                    language=self._resolve_language(inputs),
                    instruct=self._resolve_instruct(inputs),
                    ref_text=ref_text,
                    ref_tokens=ref_tokens,
                    num_steps=self._resolve_int(inputs, "num_steps", "OM_TTS_NUM_STEPS", _DEFAULT_NUM_STEPS),
                    guidance_scale=self._resolve_float(
                        inputs,
                        "guidance_scale",
                        "OM_TTS_GUIDANCE_SCALE",
                        _DEFAULT_GUIDANCE_SCALE,
                    ),
                )
            )

            audio = np.asarray(result.audio)
            if not np.any(audio):
                raise RuntimeError(
                    "OmniVoice 生成了静音音频。"
                    f"诊断: num_steps={self._resolve_int(inputs, 'num_steps', 'OM_TTS_NUM_STEPS', _DEFAULT_NUM_STEPS)}, "
                    f"guidance_scale={self._resolve_float(inputs, 'guidance_scale', 'OM_TTS_GUIDANCE_SCALE', _DEFAULT_GUIDANCE_SCALE)}, "
                    f"audio_tokenizer_loaded={getattr(model, 'audio_tokenizer', None) is not None}, "
                    f"peak_memory_gb={getattr(result, 'peak_memory_usage', 'n/a')}, "
                    f"real_time_factor={getattr(result, 'real_time_factor', 'n/a')}. "
                    "可能原因: audio tokenizer 未正确加载、num_steps 过低、或 guidance_scale 异常。"
                )

            self._save_wave(temp_wav_path, audio, result.sample_rate)
            if requested_format != "wav":
                self._transcode_audio(temp_wav_path, output_path, requested_format)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)

            audio_duration = probe_duration(output_path)
            instruct = self._resolve_instruct(inputs)

            return ToolResult(
                success=True,
                data={
                    "provider": self.provider,
                    "backend": "omnivoice",
                    "model": self._resolve_model(inputs),
                    "language": self._resolve_language(inputs),
                    "instruct": instruct,
                    "voice_hint": inputs.get("voice"),
                    "voice_mode": self._voice_mode(instruct, ref_audio),
                    "ref_audio": str(ref_audio) if ref_audio else None,
                    "ref_text": ref_text,
                    "response_format": requested_format,
                    "format": requested_format,
                    "text_length": len(text),
                    "audio_duration_seconds": round(audio_duration, 2) if audio_duration else None,
                    "duration_probe_warning": None if audio_duration else "ffprobe unavailable; audio_duration_seconds is None",
                    "peak_memory_gb": round(float(getattr(result, "peak_memory_usage", 0.0)), 2) or None,
                    "real_time_factor": round(float(getattr(result, "real_time_factor", 0.0)), 2) or None,
                    "output": str(output_path),
                },
                artifacts=[str(output_path)],
                model=self._resolve_model(inputs),
            )
        finally:
            del ref_tokens
            del result
            del model
            if temp_dir is not None:
                temp_dir.cleanup()
            gc.collect()
            self._release_mlx_resources(mx)

    @staticmethod
    def _resolve_model(inputs: dict[str, Any]) -> str:
        return inputs.get("model") or inputs.get("model_id") or os.environ.get("OM_TTS_MODEL", _DEFAULT_MODEL)

    @staticmethod
    def _resolve_instruct(inputs: dict[str, Any]) -> str | None:
        instruct = inputs.get("instruct")
        if instruct is None:
            instruct = inputs.get("instructions")
        if instruct is None:
            instruct = os.environ.get("OM_TTS_INSTRUCT")
        if instruct is None:
            return None
        instruct = str(instruct).strip()
        return instruct or None

    @staticmethod
    def _resolve_language(inputs: dict[str, Any]) -> str:
        raw = (
            inputs.get("language")
            or inputs.get("lang_code")
            or inputs.get("language_code")
            or os.environ.get("OM_TTS_LANGUAGE")
            or _DEFAULT_LANGUAGE
        )
        normalized = str(raw).strip()
        lowered = normalized.lower()
        return _LANGUAGE_ALIASES.get(lowered, lowered or _DEFAULT_LANGUAGE)

    @staticmethod
    def _resolve_optional_duration(inputs: dict[str, Any]) -> float | None:
        value = inputs.get("duration_seconds", inputs.get("duration"))
        if value in (None, ""):
            return None
        duration = float(value)
        if duration <= 0:
            raise ValueError("duration_seconds 必须大于 0。")
        return duration

    @staticmethod
    def _resolve_int(inputs: dict[str, Any], key: str, env_name: str, default: int) -> int:
        raw = inputs.get(key, os.environ.get(env_name, default))
        value = int(raw)
        if value < 1:
            raise ValueError(f"{key} 必须大于等于 1。")
        return value

    @staticmethod
    def _resolve_float(inputs: dict[str, Any], key: str, env_name: str, default: float) -> float:
        raw = inputs.get(key, os.environ.get(env_name, default))
        value = float(raw)
        if value <= 0:
            raise ValueError(f"{key} 必须大于 0。")
        return value

    @staticmethod
    def _resolve_optional_positive_float(raw: Any, key: str, default: float) -> float:
        if raw in (None, ""):
            return default
        value = float(raw)
        if value <= 0:
            raise ValueError(f"{key} 必须大于 0。")
        return value

    @staticmethod
    def _resolve_ref_audio(inputs: dict[str, Any]) -> Path | None:
        raw = inputs.get("ref_audio") or os.environ.get("OM_TTS_REF_AUDIO")
        if not raw:
            return None
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {path}")
        return path

    @staticmethod
    def _resolve_ref_text(inputs: dict[str, Any]) -> str | None:
        raw = inputs.get("ref_text") or os.environ.get("OM_TTS_REF_TEXT")
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def _resolve_output_target(self, inputs: dict[str, Any]) -> tuple[str, Path]:
        requested_format = (
            inputs.get("response_format")
            or inputs.get("format")
            or inputs.get("output_format")
            or "wav"
        )
        requested_format = str(requested_format).strip().lower()

        output_path = Path(inputs.get("output_path", f"{self.name}.{requested_format}"))
        suffix = output_path.suffix.lstrip(".").lower()
        if requested_format == "wav" and suffix in _SUPPORTED_OUTPUT_FORMATS:
            requested_format = suffix
        if requested_format not in _SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported response_format {requested_format!r}. "
                f"Supported: {sorted(_SUPPORTED_OUTPUT_FORMATS)}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        return requested_format, output_path

    @staticmethod
    def _apply_runtime_limits(mx: Any, inputs: dict[str, Any]) -> None:
        mx.clear_cache()
        mx.reset_peak_memory()

        memory_limit = inputs.get("memory_limit_gb", os.environ.get("OM_TTS_MEMORY_LIMIT_GB"))
        cache_limit = inputs.get("cache_limit_gb", os.environ.get("OM_TTS_CACHE_LIMIT_GB"))
        metal = getattr(mx, "metal", None)
        if metal is None:
            return
        if memory_limit not in (None, ""):
            metal.set_memory_limit(int(float(memory_limit) * 1024**3))
        if cache_limit not in (None, ""):
            metal.set_cache_limit(int(float(cache_limit) * 1024**3))

    @staticmethod
    def _cleanup_incomplete_tokenizer_downloads() -> None:
        blobs_dir = _TOKENIZER_REPO_CACHE / "blobs"
        if not blobs_dir.exists():
            return
        for incomplete_file in blobs_dir.glob("*.incomplete"):
            incomplete_file.unlink(missing_ok=True)

    @staticmethod
    def _release_mlx_resources(mx: Any) -> None:
        try:
            mx.clear_cache()
            mx.reset_peak_memory()
        except Exception:
            pass

    @staticmethod
    def _save_wave(output_path: Path, audio: Any, sample_rate: int) -> None:
        import numpy as np

        # mlx-audio returns (samples,) for mono or (samples, channels) for multi-channel.
        if audio.ndim == 1:
            channels = 1
        elif audio.ndim == 2:
            channels = audio.shape[1]
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")

        audio = np.clip(audio, -1.0, 1.0)
        audio_pcm = (audio * 32767).astype(np.int16)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_pcm.tobytes())

    @staticmethod
    def _transcode_audio(input_path: Path, output_path: Path, output_format: str) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found on PATH. OmniVoice 只会原生输出 wav；"
                "若要输出 mp3/flac/aac/opus，需要先安装 ffmpeg。"
            )

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
        ]
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

    @staticmethod
    def _voice_mode(instruct: str | None, ref_audio: Path | None) -> str:
        if ref_audio is not None:
            return "voice_cloning"
        if instruct:
            return "voice_design"
        return "auto"
