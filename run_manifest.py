from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from music_analysis import build_retrieval_hints


SCHEMA_VERSION = "0.1.0"


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _normalize_artifact_list(items: list[dict[str, object]] | None) -> list[dict[str, object]]:
    normalized = []
    for item in items or []:
        normalized_item = dict(item)
        if "path" in normalized_item:
            normalized_item["path"] = _normalize_path(normalized_item["path"])
        normalized.append(normalized_item)
    return normalized


def build_run_manifest(
    *,
    run_type: str,
    source_audio_path: str | Path,
    status: str,
    config: dict[str, object],
    separator: dict[str, object] | None = None,
    transcription: dict[str, object] | None = None,
    analysis: dict[str, object] | None = None,
    lyrics: dict[str, object] | None = None,
    references: dict[str, object] | None = None,
    run_id: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, object]:
    run_id = run_id or uuid.uuid4().hex
    source_audio_path = _normalize_path(source_audio_path)

    references = dict(references or {})
    references["source_audio"] = _normalize_path(references.get("source_audio") or source_audio_path)
    references["stem_directory"] = _normalize_path(references.get("stem_directory"))
    references["stems"] = _normalize_artifact_list(references.get("stems"))
    references["midi"] = _normalize_artifact_list(references.get("midi"))
    references["lyrics"] = _normalize_artifact_list(references.get("lyrics"))
    references["manifests"] = _normalize_artifact_list(references.get("manifests"))

    separator = dict(separator or {"status": "not_requested"})
    transcription = dict(transcription or {"status": "not_requested"})
    analysis = dict(analysis or {"enabled": False, "status": "disabled"})
    lyrics = dict(lyrics or {"enabled": False, "status": "disabled"})

    target_ref = None
    selected_stem = references.get("selected_stem")
    if selected_stem and references["stems"]:
        target_ref = next(
            (stem for stem in references["stems"] if stem.get("stem") == selected_stem),
            None,
        )
    if target_ref is None and references["stems"]:
        target_ref = references["stems"][0]
    if target_ref is None and references["midi"]:
        target_ref = references["midi"][0]

    retrieval_item_id = f"{run_id}:source"
    retrieval_label = Path(source_audio_path).name if source_audio_path else run_id
    retrieval_hints = build_retrieval_hints(
        item_id=retrieval_item_id,
        analysis=(analysis.get("source_audio") if isinstance(analysis, dict) else None),
        source_audio_path=references["source_audio"],
        lyrics_excerpt=(lyrics.get("normalized_excerpt") if isinstance(lyrics, dict) else None),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_epoch": int(time.time()),
        "run_id": run_id,
        "run_type": run_type,
        "status": status,
        "input": {
            "audio_file": source_audio_path,
            "audio_name": Path(source_audio_path).name if source_audio_path else None,
        },
        "config": config,
        "separator": separator,
        "transcription": transcription,
        "analysis": analysis,
        "lyrics": lyrics,
        "references": references,
        "retrieval": {
            "label": retrieval_label,
            "item_id": retrieval_item_id,
            "target_reference": target_ref,
            "hints": retrieval_hints,
        },
        "notes": list(notes or []),
    }
    return manifest


def write_run_manifest(output_path: str | Path, manifest: dict[str, object]) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(output.resolve())


def load_run_manifest(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
