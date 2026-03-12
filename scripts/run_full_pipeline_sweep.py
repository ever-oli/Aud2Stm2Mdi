#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNTIME_ROOT = PROJECT_ROOT / ".tmp" / "full_pipeline_runtime"
RUNTIME_CACHE_DIR = RUNTIME_ROOT / "cache"
RUNTIME_MPLCONFIG_DIR = RUNTIME_ROOT / "mplconfig"
RUNTIME_TMP_DIR = RUNTIME_ROOT / "tmp"
RUNTIME_HF_HOME = RUNTIME_CACHE_DIR / "huggingface"
RUNTIME_TORCH_HOME = PROJECT_ROOT / ".tmp" / "separator_runtime" / "cache" / "torch"

for runtime_dir in (
    RUNTIME_ROOT,
    RUNTIME_CACHE_DIR,
    RUNTIME_MPLCONFIG_DIR,
    RUNTIME_TMP_DIR,
    RUNTIME_HF_HOME,
    RUNTIME_TORCH_HOME,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)

os.environ["XDG_CACHE_HOME"] = str(RUNTIME_CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(RUNTIME_MPLCONFIG_DIR)
os.environ["TMPDIR"] = str(RUNTIME_TMP_DIR)
os.environ["HF_HOME"] = str(RUNTIME_HF_HOME)
os.environ["TORCH_HOME"] = str(RUNTIME_TORCH_HOME)
tempfile.tempdir = str(RUNTIME_TMP_DIR)

from amt_backends import create_amt_processor
from amt_registry import resolve_amt_model_names
from context_pipeline import disabled_analysis, disabled_lyrics, run_optional_analysis, run_optional_lyrics
from run_manifest import build_run_manifest, write_run_manifest
from separator_backends import create_separator_processor
from separator_registry import (
    get_default_stem_for_model,
    get_separator_model_spec,
    resolve_separator_model_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full separation -> MIDI pipeline across all registered separator "
            "and transcription backends."
        )
    )
    parser.add_argument("audio_file", nargs="?", help="Path to the input audio file.")
    parser.add_argument(
        "--separator-models",
        nargs="*",
        default=None,
        help="Separator model keys to run. Use 'all' or omit to sweep every registered model.",
    )
    parser.add_argument(
        "--amt-models",
        nargs="*",
        default=None,
        help="AMT model keys to run. Use 'all' or omit to sweep every registered model.",
    )
    parser.add_argument(
        "--stems",
        choices=["default", "all"],
        default="all",
        help=(
            "Whether to transcribe only each separator model's default stem or every "
            "available stem. Default: all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where separated stems, MIDIs, and the JSON summary will be written.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run optional tempo/key/section analysis for the source mix and transcribed stems.",
    )
    parser.add_argument(
        "--transcribe-lyrics",
        action="store_true",
        help="Run optional lyrics transcription during the full pipeline sweep.",
    )
    parser.add_argument(
        "--lyrics-stems",
        choices=["vocals", "all"],
        default="vocals",
        help="Which stems should receive lyrics transcription when --transcribe-lyrics is enabled.",
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


def _get_amt_processor(cache: dict[str, object], model_name: str):
    processor = cache.get(model_name)
    if processor is None:
        processor = create_amt_processor(model_name)
        cache[model_name] = processor
    return processor


def _run_amt_on_stem(
    amt_processors: dict[str, object],
    amt_model_name: str,
    stem_audio_path: Path,
    midi_output_path: Path,
) -> dict:
    processor = _get_amt_processor(amt_processors, amt_model_name)
    midi_output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    processor.convert_to_midi(
        str(stem_audio_path),
        str(midi_output_path),
        onset_threshold=0.5,
        frame_threshold=0.4,
        minimum_note_length=150.0,
        multiple_pitch_bends=False,
    )
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    return {
        "amt_model": amt_model_name,
        "status": "ok",
        "elapsed_seconds": elapsed_seconds,
        "midi_path": str(midi_output_path),
    }


def _get_cached_analysis(cache: dict[str, dict], audio_path: Path, *, enabled: bool) -> dict[str, object]:
    key = str(audio_path.resolve())
    if key not in cache:
        cache[key] = run_optional_analysis(audio_path, enabled=enabled)
    return cache[key]


def _should_transcribe_lyrics(stem_name: str, *, enabled: bool, lyrics_stems: str) -> bool:
    if not enabled:
        return False
    if lyrics_stems == "all":
        return True
    return stem_name == "vocals"


def _get_cached_lyrics(
    cache: dict[tuple[str, str, str | None], dict],
    audio_path: Path,
    output_path: Path,
    *,
    enabled: bool,
    model_size: str,
    language: str | None,
    sections: list[dict[str, object]] | None,
) -> dict[str, object]:
    key = (str(audio_path.resolve()), model_size, language)
    if key not in cache:
        cache[key] = run_optional_lyrics(
            audio_path,
            output_path,
            enabled=enabled,
            model_size=model_size,
            language=language,
            sections=sections,
        )
    return cache[key]


def run_separator_pipeline(
    audio_file: Path,
    output_root: Path,
    separator_model_name: str,
    amt_model_names: list[str],
    stems_mode: str,
    amt_processors: dict[str, object],
    analysis_cache: dict[str, dict],
    lyrics_cache: dict[tuple[str, str, str | None], dict],
    *,
    analyze: bool,
    transcribe_lyrics: bool,
    lyrics_stems: str,
    lyrics_model: str,
    lyrics_language: str | None,
) -> dict:
    separator_spec = get_separator_model_spec(separator_model_name)
    separator_output_dir = output_root / separator_model_name
    stems_output_dir = separator_output_dir / "stems"
    stems_output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    separator_processor = create_separator_processor(separator_model_name)
    stem_paths = separator_processor.separate_to_dir(str(audio_file), str(stems_output_dir))
    separation_elapsed_seconds = round(time.perf_counter() - started_at, 3)

    if stems_mode == "default":
        target_stems = [get_default_stem_for_model(separator_model_name)]
    else:
        target_stems = list(separator_processor.sources)

    pipelines: list[dict] = []
    stem_entries = [
        {
            "stem": stem_name,
            "path": str(stem_path),
        }
        for stem_name, stem_path in sorted(stem_paths.items())
    ]
    for stem_name in target_stems:
        stem_audio_path = Path(stem_paths[stem_name])
        for amt_model_name in amt_model_names:
            target_analysis = _get_cached_analysis(analysis_cache, stem_audio_path, enabled=analyze)
            source_analysis = _get_cached_analysis(analysis_cache, audio_file, enabled=analyze)
            midi_output_path = (
                separator_output_dir
                / "midi"
                / stem_name
                / f"{audio_file.stem}_{separator_model_name}_{stem_name}_{amt_model_name}.mid"
            )
            try:
                pipeline_result = _run_amt_on_stem(
                    amt_processors,
                    amt_model_name,
                    stem_audio_path,
                    midi_output_path,
                )
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                pipeline_result = {
                    "amt_model": amt_model_name,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc) or repr(exc),
                }

            lyrics_payload = disabled_lyrics()
            lyrics_entries = []
            if _should_transcribe_lyrics(
                stem_name,
                enabled=transcribe_lyrics,
                lyrics_stems=lyrics_stems,
            ):
                lyrics_output_path = (
                    separator_output_dir / "lyrics" / stem_name / f"{audio_file.stem}_{separator_model_name}_{stem_name}.json"
                )
                lyrics_result = _get_cached_lyrics(
                    lyrics_cache,
                    stem_audio_path,
                    lyrics_output_path,
                    enabled=True,
                    model_size=lyrics_model,
                    language=lyrics_language,
                    sections=(target_analysis.get("sections") or []) if analyze else [],
                )
                lyrics_payload = {
                    "enabled": True,
                    "stem": stem_name,
                    **lyrics_result,
                }
                if lyrics_result.get("status") == "ok":
                    lyrics_entries.append({"kind": "lyrics", "stem": stem_name, "path": str(lyrics_output_path)})

            analysis_payload = disabled_analysis()
            if analyze:
                analysis_payload = {
                    "enabled": True,
                    "status": "ok",
                    "source_audio": source_analysis,
                    "target_audio": target_analysis,
                    "stems": [
                        {
                            "stem": stem_name,
                            "analysis": target_analysis,
                        }
                    ],
                }
                if source_analysis.get("status") in {"error", "partial"} or target_analysis.get("status") in {"error", "partial"}:
                    analysis_payload["status"] = "partial"

            manifest_status = "ok"
            if analysis_payload.get("status") in {"error", "partial"} or lyrics_payload.get("status") in {"error", "partial"}:
                manifest_status = "partial"
            if pipeline_result.get("status") == "error":
                manifest_status = "error"

            manifest_path = (
                separator_output_dir / "manifests" / f"{audio_file.stem}_{separator_model_name}_{stem_name}_{amt_model_name}_manifest.json"
            )
            manifest = build_run_manifest(
                run_type="full_pipeline_sweep",
                source_audio_path=audio_file,
                status=manifest_status,
                config={
                    "separator_model": separator_model_name,
                    "separator_display_name": separator_spec.display_name,
                    "stem": stem_name,
                    "amt_model": amt_model_name,
                    "stems_mode": stems_mode,
                    "analyze_audio_metadata": analyze,
                    "transcribe_lyrics": transcribe_lyrics,
                    "lyrics_stems": lyrics_stems if transcribe_lyrics else None,
                    "lyrics_model": lyrics_model if transcribe_lyrics else None,
                    "lyrics_language": lyrics_language if transcribe_lyrics else None,
                },
                separator={
                    "status": "ok",
                    "model": separator_model_name,
                    "display_name": separator_spec.display_name,
                    "backend": separator_spec.backend,
                    "elapsed_seconds": separation_elapsed_seconds,
                    "sources": list(separator_processor.sources),
                    "output_dir": str(stems_output_dir),
                },
                transcription={
                    **pipeline_result,
                    "model": amt_model_name,
                } if pipeline_result.get("status") == "ok" else pipeline_result,
                analysis=analysis_payload,
                lyrics=lyrics_payload,
                references={
                    "source_audio": str(audio_file),
                    "stem_directory": str(stems_output_dir),
                    "selected_stem": stem_name,
                    "stems": stem_entries,
                    "midi": (
                        [{"kind": "midi", "stem": stem_name, "path": pipeline_result["midi_path"]}]
                        if pipeline_result.get("status") == "ok"
                        else []
                    ),
                    "lyrics": lyrics_entries,
                    "manifests": [{"kind": "run_manifest", "path": str(manifest_path)}],
                },
            )
            manifest_output = write_run_manifest(manifest_path, manifest)

            pipelines.append(
                {
                    "stem": stem_name,
                    "stem_audio_path": str(stem_audio_path),
                    "manifest_path": manifest_output,
                    "analysis_status": analysis_payload.get("status"),
                    "lyrics_status": lyrics_payload.get("status"),
                    **pipeline_result,
                }
            )

    return {
        "separator_model": separator_model_name,
        "display_name": separator_spec.display_name,
        "backend": separator_spec.backend,
        "sources": list(separator_processor.sources),
        "separation": {
            "status": "ok",
            "elapsed_seconds": separation_elapsed_seconds,
            "output_dir": str(stems_output_dir),
            "stems": [
                {
                    "stem": stem_name,
                    "path": str(stem_path),
                }
                for stem_name, stem_path in sorted(stem_paths.items())
            ],
        },
        "pipelines": pipelines,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.audio_file:
        parser.error("audio_file is required.")

    audio_file = Path(args.audio_file).expanduser().resolve()
    if not audio_file.is_file():
        parser.error(f"Audio file does not exist: {audio_file}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "pipeline_sweeps" / audio_file.stem).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    separator_model_names = resolve_separator_model_names(args.separator_models)
    amt_model_names = resolve_amt_model_names(args.amt_models)
    amt_processors: dict[str, object] = {}

    summary = {
        "audio_file": str(audio_file),
        "output_root": str(output_root),
        "separator_models": separator_model_names,
        "amt_models": amt_model_names,
        "stems_mode": args.stems,
        "analyze": args.analyze,
        "transcribe_lyrics": args.transcribe_lyrics,
        "lyrics_stems": args.lyrics_stems,
        "results": [],
    }
    analysis_cache: dict[str, dict] = {}
    lyrics_cache: dict[tuple[str, str, str | None], dict] = {}

    for separator_model_name in separator_model_names:
        print(f"[pipeline-sweep] separating with {separator_model_name}")
        try:
            result = run_separator_pipeline(
                audio_file,
                output_root,
                separator_model_name,
                amt_model_names,
                args.stems,
                amt_processors,
                analysis_cache,
                lyrics_cache,
                analyze=args.analyze,
                transcribe_lyrics=args.transcribe_lyrics,
                lyrics_stems=args.lyrics_stems,
                lyrics_model=args.lyrics_model,
                lyrics_language=args.lyrics_language,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            result = {
                "separator_model": separator_model_name,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc) or repr(exc),
            }
        summary["results"].append(result)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[pipeline-sweep] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
