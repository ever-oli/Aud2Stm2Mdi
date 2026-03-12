from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


INDEX_METADATA_SUFFIX = ".json"
INDEX_EMBEDDINGS_SUFFIX = ".npz"


class ClapIndexer:
    def __init__(self, enable_fusion: bool = False, checkpoint_path: str | None = None):
        try:
            import laion_clap
        except ImportError as exc:
            raise RuntimeError(
                "LAION-CLAP retrieval dependencies are incomplete. Install "
                "`requirements-context-backends.txt` to enable CLAP indexing."
            ) from exc

        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        if checkpoint_path:
            self.model.load_ckpt(ckpt=checkpoint_path)
        else:
            self.model.load_ckpt()

    def embed_audio_files(self, file_paths: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.get_audio_embedding_from_filelist(x=list(file_paths), use_tensor=False)
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.get_text_embedding(list(texts), use_tensor=False))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def save_index(prefix: str, metadata: list[dict[str, object]], embeddings: np.ndarray) -> tuple[str, str]:
    metadata_path = prefix + INDEX_METADATA_SUFFIX
    embeddings_path = prefix + INDEX_EMBEDDINGS_SUFFIX
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    np.savez_compressed(embeddings_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    return metadata_path, embeddings_path


def load_index(prefix: str) -> tuple[list[dict[str, object]], np.ndarray]:
    metadata_path = prefix + INDEX_METADATA_SUFFIX
    embeddings_path = prefix + INDEX_EMBEDDINGS_SUFFIX
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    embeddings = np.load(embeddings_path)["embeddings"]
    return metadata, embeddings


def index_exists(prefix: str) -> bool:
    return os.path.isfile(prefix + INDEX_METADATA_SUFFIX) and os.path.isfile(prefix + INDEX_EMBEDDINGS_SUFFIX)


def search_by_embedding(
    query_embedding: np.ndarray,
    metadata: list[dict[str, object]],
    embeddings: np.ndarray,
    *,
    limit: int = 10,
    exclude_item_id: str | None = None,
) -> list[dict[str, object]]:
    if len(metadata) == 0:
        return []
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    embeddings = _normalize_rows(embeddings)
    query_embedding = _normalize_rows(query_embedding)[0]
    scores = embeddings @ query_embedding
    ranking = np.argsort(scores)[::-1]
    results = []
    for idx in ranking:
        item = metadata[idx]
        if exclude_item_id and item.get("item_id") == exclude_item_id:
            continue
        result = dict(item)
        result["score"] = float(scores[idx])
        results.append(result)
        if len(results) >= limit:
            break
    return results


def _artifact_label(manifest: dict[str, object], kind: str, stem_role: str | None) -> str:
    base_name = manifest.get("input", {}).get("audio_name") or manifest.get("run_id") or "audio"
    separator = manifest.get("separator", {})
    separator_name = separator.get("model") or separator.get("separator_model") or separator.get("display_name")
    if kind == "track":
        return str(base_name)
    if stem_role and separator_name:
        return f"{base_name} [{stem_role}] ({separator_name})"
    if stem_role:
        return f"{base_name} [{stem_role}]"
    return str(base_name)


def collect_audio_assets_from_manifest(manifest_path: str | Path) -> list[dict[str, object]]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = manifest.get("references", {})
    analysis_block = manifest.get("analysis", {})
    lyrics_block = manifest.get("lyrics", {})
    assets = []

    source_audio_path = references.get("source_audio")
    if source_audio_path and os.path.isfile(source_audio_path):
        source_analysis = analysis_block.get("source_audio") or {}
        assets.append(
            {
                "item_id": f"{manifest['run_id']}:source",
                "kind": "track",
                "label": _artifact_label(manifest, "track", None),
                "path": str(Path(source_audio_path).resolve()),
                "tempo": source_analysis.get("tempo"),
                "key": source_analysis.get("key"),
                "section_labels": source_analysis.get("structure_labels") or [],
                "mood_candidates": source_analysis.get("mood_tags") or [],
                "lyric_excerpt": lyrics_block.get("normalized_excerpt"),
                "manifest_path": str(manifest_path),
                "run_type": manifest.get("run_type"),
            }
        )

    stem_analysis_lookup = {}
    for item in analysis_block.get("stems", []) or []:
        stem_name = item.get("stem")
        if stem_name:
            stem_analysis_lookup[stem_name] = item.get("analysis") or {}

    for stem_ref in references.get("stems", []) or []:
        stem_path = stem_ref.get("path")
        stem_name = stem_ref.get("stem")
        if not stem_path or not os.path.isfile(stem_path):
            continue
        analysis = stem_analysis_lookup.get(stem_name)
        if analysis is None and references.get("selected_stem") == stem_name:
            analysis = analysis_block.get("target_audio") or {}
        if analysis is None:
            analysis = {}
        assets.append(
            {
                "item_id": f"{manifest['run_id']}:stem:{stem_name}",
                "kind": "stem",
                "stem_role": stem_name,
                "label": _artifact_label(manifest, "stem", stem_name),
                "path": str(Path(stem_path).resolve()),
                "tempo": analysis.get("tempo"),
                "key": analysis.get("key"),
                "section_labels": analysis.get("structure_labels") or [],
                "mood_candidates": analysis.get("mood_tags") or [],
                "lyric_excerpt": lyrics_block.get("normalized_excerpt")
                if (
                    stem_name == references.get("selected_stem")
                    or stem_name == lyrics_block.get("stem")
                )
                else None,
                "manifest_path": str(manifest_path),
                "run_type": manifest.get("run_type"),
            }
        )
    return assets


def iter_manifest_paths(root: str | Path) -> list[str]:
    root_path = Path(root).expanduser().resolve()
    if root_path.is_file():
        return [str(root_path)]
    return sorted(
        str(path)
        for path in root_path.rglob("*.json")
        if path.name.endswith("manifest.json") or path.name == "run_manifest.json"
    )


def collect_audio_assets_from_root(root: str | Path) -> list[dict[str, object]]:
    manifest_paths = iter_manifest_paths(root)
    deduped: dict[str, dict[str, object]] = {}
    for manifest_path in manifest_paths:
        for item in collect_audio_assets_from_manifest(manifest_path):
            deduped.setdefault(item["path"], item)
    return list(deduped.values())


def build_clap_index_from_root(
    root: str | Path,
    output_prefix: str,
    *,
    indexer: ClapIndexer | None = None,
) -> dict[str, object]:
    metadata = collect_audio_assets_from_root(root)
    if len(metadata) == 0:
        raise RuntimeError("No audio assets were discovered from manifests.")
    indexer = indexer or ClapIndexer()
    audio_paths = [str(item["path"]) for item in metadata]
    embeddings = indexer.embed_audio_files(audio_paths)
    metadata_path, embeddings_path = save_index(output_prefix, metadata, embeddings)
    return {
        "metadata_path": metadata_path,
        "embeddings_path": embeddings_path,
        "asset_count": len(metadata),
    }


def format_result(result: dict[str, object], rank: int | None = None) -> str:
    prefix = f"{rank}. " if rank is not None else ""
    label = result.get("label", result.get("item_id", "unknown"))
    kind = result.get("kind", "asset")
    role = result.get("stem_role") or kind
    score = result.get("score")
    score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "n/a"
    feature_bits = []
    if result.get("tempo") is not None:
        feature_bits.append(f"BPM {result['tempo']}")
    if result.get("key"):
        feature_bits.append(f"Key {result['key']}")
    if result.get("section_labels"):
        feature_bits.append("Sections " + ",".join(str(label) for label in result["section_labels"][:4]))
    if result.get("mood_candidates"):
        feature_bits.append("Mood " + ",".join(result["mood_candidates"][:3]))
    summary = " | ".join(feature_bits)
    line = f"{prefix}{label} [{kind}/{role}] score={score_text}"
    if summary:
        line += f" | {summary}"
    return line


def format_results(results: list[dict[str, object]], header: str | None = None) -> str:
    lines = []
    if header:
        lines.append(header)
    for idx, result in enumerate(results, start=1):
        lines.append(format_result(result, rank=idx))
        lines.append(f"   item_id: {result.get('item_id')}")
        lines.append(f"   path: {result.get('path')}")
        if result.get("lyric_excerpt"):
            lines.append(f"   lyrics: {result.get('lyric_excerpt')}")
        if result.get("manifest_path"):
            lines.append(f"   manifest: {result.get('manifest_path')}")
    return "\n".join(lines)
