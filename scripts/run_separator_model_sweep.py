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
    get_separator_model_spec,
    list_separator_model_specs,
    resolve_separator_model_names,
)


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
    return parser


def print_models() -> None:
    for spec in list_separator_model_specs():
        print(f"{spec.key}: {spec.display_name} ({spec.backend})")


def run_model(audio_file: Path, output_root: Path, model_name: str) -> dict:
    processor = create_separator_processor(model_name)
    spec = get_separator_model_spec(model_name)
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    stem_paths = processor.separate_to_dir(str(audio_file), str(model_output_dir))
    elapsed_seconds = round(time.perf_counter() - started_at, 3)

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
        "models": [],
    }

    for model_name in selected_models:
        print(f"[sweep] running {model_name}")
        try:
            result = run_model(audio_file, output_root, model_name)
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
