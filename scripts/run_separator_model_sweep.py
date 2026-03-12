#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from separator_backends import create_separator_processor
from separator_registry import (
    get_default_stem_for_model,
    get_separator_model_spec,
    list_separator_model_specs,
    resolve_separator_model_names,
)
from context_pipeline import disabled_analysis, disabled_lyrics, run_optional_analysis, run_optional_lyrics
from run_manifest import build_run_manifest, write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one audio file through all registered separator backends."
    )
    parser.add_argument("audio_file", nargs="?", help="Path to the input audio file.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Separator model keys to run. Use 'all' or omit to sweep every registered model.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where model-specific outputs and the JSON summary will be written.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported separator models and exit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run optional tempo/key/section analysis on the source audio and every produced stem.",
    )
    parser.add_argument(
        "--transcribe-lyrics",
        action="store_true",
        help="Run optional lyrics transcription on the generated vocal stem when available.",
    )
    parser.add_argument(
        "--lyrics-model",
        default="small",
        help="faster-whisper model size to use when --transcribe-lyrics is enabled.",
    )
    parser.add_argument(
        "--lyrics-language",
        default=None,
        help="Optional language hint for faster-whisper when --transcribe-lyrics is enabled.",
    )
    return parser


def print_models() -> None:
    for spec in list_separator_model_specs():
        print(f"{spec.key}: {spec.display_name} ({spec.backend})")


def run_model(
    audio_file: Path,
    output_root: Path,
    model_name: str,
    *,
    analyze: bool,
    transcribe_lyrics: bool,
    lyrics_model: str,
    lyrics_language: str | None,
) -> dict:
    processor = create_separator_processor(model_name)
    spec = get_separator_model_spec(model_name)
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    stem_paths = processor.separate_to_dir(str(audio_file), str(model_output_dir))
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    stem_entries = [
        {
            "stem": stem_name,
            "path": str(stem_path),
        }
        for stem_name, stem_path in sorted(stem_paths.items())
    ]

    analysis_payload = disabled_analysis()
    if analyze:
        source_analysis = run_optional_analysis(audio_file, enabled=True)
        stem_analyses = []
        for stem_name, stem_path in sorted(stem_paths.items()):
            stem_analyses.append(
                {
                    "stem": stem_name,
                    "analysis": run_optional_analysis(stem_path, enabled=True),
                }
            )
        default_stem = get_default_stem_for_model(model_name)
        default_analysis = next(
            (item["analysis"] for item in stem_analyses if item["stem"] == default_stem),
            {"status": "disabled"},
        )
        analysis_payload = {
            "enabled": True,
            "status": "ok",
            "source_audio": source_analysis,
            "target_audio": default_analysis,
            "stems": stem_analyses,
        }
        if any(
            item.get("status") in {"error", "partial"}
            for item in [source_analysis, default_analysis]
            if isinstance(item, dict)
        ) or any(
            stem_analysis["analysis"].get("status") in {"error", "partial"}
            for stem_analysis in stem_analyses
        ):
            analysis_payload["status"] = "partial"

    lyrics_payload = disabled_lyrics()
    lyrics_entries = []
    if transcribe_lyrics and "vocals" in stem_paths:
        sections = []
        if analyze:
            target_analysis = next(
                (item["analysis"] for item in analysis_payload.get("stems", []) if item["stem"] == "vocals"),
                {},
            )
            sections = target_analysis.get("sections") or []
        lyrics_output_dir = model_output_dir / "lyrics"
        lyrics_output_dir.mkdir(parents=True, exist_ok=True)
        lyrics_output_path = lyrics_output_dir / "vocals_lyrics.json"
        lyrics_result = run_optional_lyrics(
            stem_paths["vocals"],
            lyrics_output_path,
            enabled=True,
            model_size=lyrics_model,
            language=lyrics_language,
            sections=sections,
        )
        lyrics_payload = {
            "enabled": True,
            "stem": "vocals",
            **lyrics_result,
        }
        if lyrics_result.get("status") == "ok":
            lyrics_entries.append({"kind": "lyrics", "stem": "vocals", "path": str(lyrics_output_path)})

    manifest_status = "ok"
    if analysis_payload.get("status") in {"error", "partial"} or lyrics_payload.get("status") in {"error", "partial"}:
        manifest_status = "partial"
    manifest_path = model_output_dir / "run_manifest.json"
    manifest = build_run_manifest(
        run_type="separator_sweep",
        source_audio_path=audio_file,
        status=manifest_status,
        config={
            "separator_model": model_name,
            "separator_display_name": spec.display_name,
            "analyze_audio_metadata": analyze,
            "transcribe_lyrics": transcribe_lyrics,
            "lyrics_model": lyrics_model if transcribe_lyrics else None,
            "lyrics_language": lyrics_language if transcribe_lyrics else None,
        },
        separator={
            "status": "ok",
            "model": model_name,
            "display_name": spec.display_name,
            "backend": spec.backend,
            "elapsed_seconds": elapsed_seconds,
            "sources": list(processor.sources),
            "output_dir": str(model_output_dir),
        },
        analysis=analysis_payload,
        lyrics=lyrics_payload,
        references={
            "source_audio": str(audio_file),
            "stem_directory": str(model_output_dir),
            "selected_stem": get_default_stem_for_model(model_name),
            "stems": stem_entries,
            "lyrics": lyrics_entries,
            "manifests": [{"kind": "run_manifest", "path": str(manifest_path)}],
        },
    )
    manifest_output = write_run_manifest(manifest_path, manifest)

    return {
        "model": model_name,
        "display_name": spec.display_name,
        "backend": spec.backend,
        "status": "ok",
        "elapsed_seconds": elapsed_seconds,
        "sources": list(processor.sources),
        "output_dir": str(model_output_dir),
        "stems": [
            {
                "stem": stem_name,
                "path": str(stem_path),
            }
            for stem_name, stem_path in sorted(stem_paths.items())
        ],
        "manifest_path": manifest_output,
        "analysis_status": analysis_payload.get("status"),
        "lyrics_status": lyrics_payload.get("status"),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print_models()
        return 0

    if not args.audio_file:
        parser.error("audio_file is required unless --list-models is used.")

    audio_file = Path(args.audio_file).expanduser().resolve()
    if not audio_file.is_file():
        parser.error(f"Audio file does not exist: {audio_file}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "separator_sweeps" / audio_file.stem).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_models = resolve_separator_model_names(args.models)
    summary = {
        "audio_file": str(audio_file),
        "output_root": str(output_root),
        "analyze": args.analyze,
        "transcribe_lyrics": args.transcribe_lyrics,
        "models": [],
    }

    for model_name in selected_models:
        print(f"[sweep] running {model_name}")
        try:
            result = run_model(
                audio_file,
                output_root,
                model_name,
                analyze=args.analyze,
                transcribe_lyrics=args.transcribe_lyrics,
                lyrics_model=args.lyrics_model,
                lyrics_language=args.lyrics_language,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            result = {
                "model": model_name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc) or repr(exc),
            }
        summary["models"].append(result)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[sweep] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
