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


def run_separator_pipeline(
    audio_file: Path,
    output_root: Path,
    separator_model_name: str,
    amt_model_names: list[str],
    stems_mode: str,
    amt_processors: dict[str, object],
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
    for stem_name in target_stems:
        stem_audio_path = Path(stem_paths[stem_name])
        for amt_model_name in amt_model_names:
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

            pipelines.append(
                {
                    "stem": stem_name,
                    "stem_audio_path": str(stem_audio_path),
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
        "results": [],
    }

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
