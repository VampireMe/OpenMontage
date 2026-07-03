#!/usr/bin/env python3
"""QA Test 01: Local OmniVoice TTS integration.

Runs the local_tts provider against the real OmniVoice backend when available.
This is an executable QA script, not a pytest unit test.

What it validates:
1. LocalTTS is discoverable and AVAILABLE on the current machine
2. A real generation call succeeds and writes a WAV file
3. A second generation succeeds immediately after the first one
   (smoke coverage for resource release between calls)
4. Output audio is non-empty and non-silent
5. If ffmpeg is present, the first WAV is also transcoded to MP3 for downstream QA reuse
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import wave
from array import array
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from lib.env_loader import load_env
except ModuleNotFoundError as exc:
    if exc.name == "dotenv":
        def load_env() -> None:
            print("  [info] python-dotenv not installed; skipping .env autoload")
    else:
        raise

load_env()

from tools.analysis.audio_probe import AudioProbe
from tools.audio.local_tts import LocalTTS


OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


def inspect_wav(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        if wav_file.getsampwidth() != 2:
            raise RuntimeError(f"Unsupported sample width for QA RMS check: {wav_file.getsampwidth()}")
        samples = array("h")
        samples.frombytes(frames)
        if not samples:
            rms = 0.0
        else:
            mean_square = sum(sample * sample for sample in samples) / len(samples)
            rms = math.sqrt(mean_square)
        duration = wav_file.getnframes() / float(wav_file.getframerate())
        return {
            "channels": float(wav_file.getnchannels()),
            "sample_rate": float(wav_file.getframerate()),
            "duration_seconds": duration,
            "rms": float(rms),
        }


def transcode_to_mp3(input_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil_which("ffmpeg")
    if not ffmpeg:
        print("  [info] ffmpeg not found; skipping mp3 export for downstream QA reuse")
        return False

    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(input_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"  [warn] ffmpeg transcode failed: {proc.stderr.strip()}")
        return False
    return output_path.exists()


def shutil_which(binary: str) -> str | None:
    from shutil import which

    return which(binary)


def run_generation(tool: LocalTTS, *, text: str, output_path: Path, instruct: str) -> dict:
    result = tool.execute(
        {
            "text": text,
            "instructions": instruct,
            "num_steps": int(os.environ.get("OM_TTS_QA_NUM_STEPS", "12")),
            "output_path": str(output_path),
            "response_format": "wav",
        }
    )
    if not result.success:
        raise RuntimeError(result.error or "unknown local_tts failure")
    return result.data


def main() -> int:
    print("--- QA Test 01: Local OmniVoice TTS ---")

    tool = LocalTTS()
    status = tool.get_status()
    print(f"Tool status: {status}")
    print(f"Backend: {tool._backend_name()}")

    if status.value != "available":
        print("\n[blocked] local_tts is not available on this machine.")
        print(tool.install_instructions)
        return 2

    prompt_a = "你好，这是本地 OmniVoice 集成测试。"
    prompt_b = "第二次调用用于确认资源释放后仍可继续生成。"
    instruct = os.environ.get("OM_TTS_INSTRUCT", "female, young adult")

    wav_a = OUT / "tts_short.wav"
    wav_b = OUT / "tts_repeat.wav"
    mp3_a = OUT / "tts_short.mp3"

    print("\n--- Run 1: primary generation ---")
    data_a = run_generation(tool, text=prompt_a, output_path=wav_a, instruct=instruct)
    print(json.dumps(data_a, ensure_ascii=False, indent=2))

    print("\n--- Run 2: repeat generation (resource release smoke) ---")
    data_b = run_generation(tool, text=prompt_b, output_path=wav_b, instruct=instruct)
    print(json.dumps(data_b, ensure_ascii=False, indent=2))

    probe = AudioProbe()
    probe_a = probe.execute({"input_path": str(wav_a)})
    probe_b = probe.execute({"input_path": str(wav_b)})
    wav_info_a = inspect_wav(wav_a)
    wav_info_b = inspect_wav(wav_b)

    check("Run 1 wrote WAV output", wav_a.exists(), str(wav_a))
    check("Run 2 wrote WAV output", wav_b.exists(), str(wav_b))
    check("Run 1 probe succeeded", probe_a.success, probe_a.error or "")
    check("Run 2 probe succeeded", probe_b.success, probe_b.error or "")
    check("Run 1 duration > 0.5s", wav_info_a["duration_seconds"] > 0.5, f"{wav_info_a['duration_seconds']:.2f}s")
    check("Run 2 duration > 0.5s", wav_info_b["duration_seconds"] > 0.5, f"{wav_info_b['duration_seconds']:.2f}s")
    check("Run 1 audio is non-silent", wav_info_a["rms"] > 0, f"rms={wav_info_a['rms']:.0f}")
    check("Run 2 audio is non-silent", wav_info_b["rms"] > 0, f"rms={wav_info_b['rms']:.0f}")
    check("Provider metadata is local_python", data_a.get("provider") == "local_python", str(data_a.get("provider")))
    check("Backend metadata is omnivoice", data_a.get("backend") == "omnivoice", str(data_a.get("backend")))
    check("Voice mode is voice_design", data_a.get("voice_mode") == "voice_design", str(data_a.get("voice_mode")))
    check(
        "Second output differs from first",
        wav_a.read_bytes() != wav_b.read_bytes(),
        "expected different waveform for different text",
    )

    print("\n--- Optional downstream fixture export ---")
    exported = transcode_to_mp3(wav_a, mp3_a)
    check("Exported tts_short.mp3 for downstream QA", exported or not shutil_which("ffmpeg"), str(mp3_a))

    print("\n--- Output summary ---")
    print(
        json.dumps(
            {
                "wav_a": {"path": str(wav_a), **wav_info_a},
                "wav_b": {"path": str(wav_b), **wav_info_b},
                "mp3_a": str(mp3_a) if mp3_a.exists() else None,
                "pass": PASS,
                "fail": FAIL,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== LOCAL TTS QA COMPLETE ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
