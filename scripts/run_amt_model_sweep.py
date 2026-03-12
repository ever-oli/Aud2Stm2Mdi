#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amt_backends import create_amt_processor
from amt_registry import (
    get_amt_model_spec,
    list_amt_model_specs,
    resolve_amt_model_names,
)
from context_pipeline import disabled_analysis, disabled_lyrics, run_optional_analysis, run_optional_lyrics
from run_manifest import build_run_manifest, write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one audio file through all registered AMT backends."
    )
    parser.add_argument("audio_file", nargs="?", help="Path to the input audio file.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="AMT model keys to run. Use 'all' or omit to sweep every registered model.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where model-specific MIDIs and the JSON summary will be written.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print supported AMT models and exit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run optional tempo/key/section analysis and include it in each manifest.",
    )
    parser.add_argument(
        "--transcribe-lyrics",
        action="store_true",
        help="Run optional lyrics transcription on the input audio and include it in each manifest.",
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
    for spec in list_amt_model_specs():
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
    processor = create_amt_processor(model_name)
    spec = get_amt_model_spec(model_name)
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = model_output_dir / f"{audio_file.stem}_{model_name}.mid"

    started_at = time.perf_counter()
    processor.convert_to_midi(
        str(audio_file),
        str(midi_path),
        onset_threshold=0.5,
        frame_threshold=0.4,
        minimum_note_length=150.0,
        multiple_pitch_bends=False,
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    analysis_payload = disabled_analysis()
    if analyze:
        source_analysis = run_optional_analysis(audio_file, enabled=True)
        analysis_payload = {
            "enabled": True,
            "status": "ok" if source_analysis.get("status") == "ok" else "partial",
            "source_audio": source_analysis,
            "target_audio": source_analysis,
            "stems": [],
        }

    lyrics_payload = disabled_lyrics()
    lyrics_json_path = model_output_dir / f"{audio_file.stem}_{model_name}_lyrics.json"
    if transcribe_lyrics:
        sections = []
        if analyze:
            sections = analysis_payload.get("source_audio", {}).get("sections") or []
        lyrics_result = run_optional_lyrics(
            audio_file,
            lyrics_json_path,
            enabled=True,
            model_size=lyrics_model,
            language=lyrics_language,
            sections=sections,
        )
        lyrics_payload = {
            "enabled": True,
            **lyrics_result,
        }

    manifest_status = "ok"
    if analysis_payload.get("status") in {"error", "partial"} or lyrics_payload.get("status") in {"error", "partial"}:
        manifest_status = "partial"
    manifest_path = model_output_dir / "run_manifest.json"
    manifest = build_run_manifest(
        run_type="amt_sweep",
        source_audio_path=audio_file,
        status=manifest_status,
        config={
            "amt_model": model_name,
            "amt_display_name": spec.display_name,
            "analyze_audio_metadata": analyze,
            "transcribe_lyrics": transcribe_lyrics,
            "lyrics_model": lyrics_model if transcribe_lyrics else None,
            "lyrics_language": lyrics_language if transcribe_lyrics else None,
        },
        transcription={
            "status": "ok",
            "model": model_name,
            "display_name": spec.display_name,
            "backend": spec.backend,
            "elapsed_seconds": elapsed_seconds,
            "midi_path": str(midi_path),
        },
        analysis=analysis_payload,
        lyrics=lyrics_payload,
        references={
            "source_audio": str(audio_file),
            "midi": [{"kind": "midi", "path": str(midi_path)}],
            "lyrics": ([{"kind": "lyrics", "path": str(lyrics_json_path)}] if lyrics_payload.get("status") == "ok" else []),
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
        "output_dir": str(model_output_dir),
        "midi_path": str(midi_path),
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
        else (Path.cwd() / "amt_sweeps" / audio_file.stem).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    selected_models = resolve_amt_model_names(args.models)
    summary = {
        "audio_file": str(audio_file),
        "output_root": str(output_root),
        "analyze": args.analyze,
        "transcribe_lyrics": args.transcribe_lyrics,
        "models": [],
    }

    for model_name in selected_models:
        print(f"[amt-sweep] running {model_name}")
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
    print(f"[amt-sweep] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
