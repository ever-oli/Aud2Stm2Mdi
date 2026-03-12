from __future__ import annotations

from pathlib import Path

from lyrics_transcription import transcribe_lyrics_to_path
from music_analysis import analyze_audio_file


def disabled_analysis() -> dict[str, object]:
    return {
        "enabled": False,
        "status": "disabled",
        "source_audio": {"status": "disabled"},
        "target_audio": {"status": "disabled"},
        "stems": [],
    }


def run_optional_analysis(audio_path: str | Path, *, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"status": "disabled"}
    try:
        return analyze_audio_file(str(audio_path))
    except Exception as exc:
        return {
            "status": "error",
            "audio_path": str(Path(audio_path).expanduser().resolve()),
            "issues": [str(exc)],
        }


def disabled_lyrics() -> dict[str, object]:
    return {
        "enabled": False,
        "status": "disabled",
    }


def run_optional_lyrics(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    enabled: bool,
    model_size: str = "small",
    language: str | None = None,
    sections: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if not enabled:
        return {"status": "disabled"}
    try:
        return transcribe_lyrics_to_path(
            str(audio_path),
            str(output_path),
            model_size=model_size,
            language=language,
            sections=sections,
        )
    except Exception as exc:
        return {
            "status": "error",
            "output_path": str(Path(output_path).expanduser().resolve()),
            "issues": [str(exc)],
        }
