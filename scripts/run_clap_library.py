#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clap_retrieval import (  # noqa: E402
    ClapIndexer,
    build_clap_index_from_root,
    format_results,
    index_exists,
    load_index,
    search_by_embedding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or query a CLAP retrieval index from run manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a CLAP index from a manifest root.")
    build_parser.add_argument("manifest_root", help="Root directory containing run manifests.")
    build_parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for the CLAP metadata/embedding files.",
    )

    similar_parser = subparsers.add_parser("similar", help="Find nearest neighbors for an indexed item.")
    similar_parser.add_argument("--index-prefix", required=True, help="Existing CLAP index prefix.")
    similar_parser.add_argument("--item-id", required=True, help="Indexed item id to query against.")
    similar_parser.add_argument("--limit", type=int, default=10, help="Maximum number of search results.")

    text_parser = subparsers.add_parser("text", help="Query the CLAP index with free text.")
    text_parser.add_argument("--index-prefix", required=True, help="Existing CLAP index prefix.")
    text_parser.add_argument("--query", required=True, help="Text query to embed and search.")
    text_parser.add_argument("--limit", type=int, default=10, help="Maximum number of search results.")
    return parser


def _search_similar(index_prefix: str, item_id: str, limit: int) -> str:
    if not index_exists(index_prefix):
        raise RuntimeError(f"CLAP index does not exist: {index_prefix}")
    metadata, embeddings = load_index(index_prefix)
    item_lookup = {item["item_id"]: idx for idx, item in enumerate(metadata)}
    if item_id not in item_lookup:
        raise RuntimeError(f"CLAP item not found in index: {item_id}")
    idx = item_lookup[item_id]
    results = search_by_embedding(
        embeddings[idx],
        metadata,
        embeddings,
        limit=limit,
        exclude_item_id=item_id,
    )
    return format_results(results, header=f"CLAP similar to {item_id}")


def _search_text(index_prefix: str, query: str, limit: int) -> str:
    if not index_exists(index_prefix):
        raise RuntimeError(f"CLAP index does not exist: {index_prefix}")
    metadata, embeddings = load_index(index_prefix)
    indexer = ClapIndexer()
    query_embedding = indexer.embed_texts([query])[0]
    results = search_by_embedding(query_embedding, metadata, embeddings, limit=limit)
    return format_results(results, header=f'CLAP text search: "{query}"')


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        result = build_clap_index_from_root(args.manifest_root, args.output_prefix)
        print("CLAP index saved:", result["metadata_path"])
        print("CLAP embeddings saved:", result["embeddings_path"])
        print("CLAP indexed assets:", result["asset_count"])
        return 0

    if args.command == "similar":
        print(_search_similar(args.index_prefix, args.item_id, args.limit))
        return 0

    if args.command == "text":
        print(_search_text(args.index_prefix, args.query, args.limit))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
