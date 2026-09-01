#!/usr/bin/env python3
"""
export_chroma_to_json.py
========================
Query a ChromaDB instance, retrieve every record from every collection, and
write the results to JSON for inspection, interpretation, and downstream use.

Designed for reproducibility:
  * Deterministic ordering (collections and records are sorted by id).
  * Full-precision embeddings by default (no silent rounding).
  * Provenance is embedded in each output file: the chromadb version, the
    source, and a UTC timestamp, so an export can always be traced.

Two connection modes
--------------------
1. Local on-disk database (default):
       python export_chroma_to_json.py --source path --path ./chroma_db

2. Running Chroma server (e.g. your `chroma run` / docker service):
       python export_chroma_to_json.py --source http --host localhost --port 8000

Output
------
By default two files are written to the output directory:
  * chroma_export_full.json          -> every field, including embeddings
  * chroma_export_no_embeddings.json -> same records without embedding vectors
Use --per-collection to additionally write one full file per collection.

Requirements
------------
  pip install chromadb
Note: to read a local --path database, your installed chromadb version must be
compatible with the version that WROTE that database. If you hit a
`KeyError: '_type'`, the on-disk database is newer than your chromadb; either
upgrade chromadb or read via --source http from a server that can open it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys


def get_client(args):
    """Return a Chroma client for either a local path or a running server."""
    import chromadb

    if args.source == "path":
        if not os.path.isdir(args.path):
            sys.exit(f"ERROR: --path '{args.path}' is not a directory.")
        return chromadb.PersistentClient(path=args.path)

    # args.source == "http"
    return chromadb.HttpClient(
        host=args.host,
        port=args.port,
        tenant=args.tenant,
        database=args.database,
    )


def to_float_list(embedding, ndigits):
    """Convert an embedding (list / numpy array) to a plain list of floats.

    Rounding is applied only if ndigits is not None; the default keeps full
    precision so the export faithfully reproduces what is stored in Chroma.
    """
    if embedding is None:
        return None
    if ndigits is None:
        return [float(x) for x in embedding]
    return [round(float(x), ndigits) for x in embedding]


def export_collection(collection, ndigits):
    """Pull all records from one collection into a serialisable dict."""
    # include=... controls which fields Chroma returns. ids are always returned.
    data = collection.get(include=["documents", "embeddings", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    embeddings = data["embeddings"]
    metadatas = data["metadatas"]

    records = []
    for i, _id in enumerate(ids):
        records.append(
            {
                "id": _id,
                "document": documents[i] if documents is not None else None,
                "metadata": metadatas[i] if metadatas is not None else None,
                "embedding": to_float_list(
                    embeddings[i] if embeddings is not None else None, ndigits
                ),
            }
        )

    # Deterministic ordering so re-running the script produces identical files.
    records.sort(key=lambda r: str(r["id"]))

    embedding_dim = (
        len(records[0]["embedding"])
        if records and records[0]["embedding"] is not None
        else None
    )

    return {
        "collection_metadata": collection.metadata,
        "count": len(records),
        "embedding_dim": embedding_dim,
        "records": records,
    }


def strip_embeddings(collection_payload):
    """Return a copy of a collection payload with embedding vectors removed."""
    light_records = [
        {k: v for k, v in rec.items() if k != "embedding"}
        for rec in collection_payload["records"]
    ]
    return {
        "collection_metadata": collection_payload["collection_metadata"],
        "count": collection_payload["count"],
        "records": light_records,
    }


def build_provenance(args):
    """Metadata describing how/when this export was produced (for reproducibility)."""
    import chromadb

    source = (
        {"type": "path", "path": os.path.abspath(args.path)}
        if args.source == "path"
        else {
            "type": "http",
            "host": args.host,
            "port": args.port,
            "tenant": args.tenant,
            "database": args.database,
        }
    )
    return {
        "exported_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "chromadb_version": getattr(chromadb, "__version__", "unknown"),
        "source": source,
        "embedding_rounding_ndigits": args.round,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export all ChromaDB collections and records to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["path", "http"],
        default="path",
        help="Read from a local on-disk database ('path') or a running server ('http').",
    )
    # Local path options
    parser.add_argument(
        "--path",
        default="./chroma_db",
        help="Path to the on-disk Chroma directory (used when --source path).",
    )
    # HTTP server options
    parser.add_argument("--host", default="localhost", help="Server host (--source http).")
    parser.add_argument("--port", type=int, default=8000, help="Server port (--source http).")
    parser.add_argument("--tenant", default="default_tenant", help="Tenant (--source http).")
    parser.add_argument("--database", default="default_database", help="Database (--source http).")
    # Output options
    parser.add_argument(
        "--out-dir",
        default="./chroma_json_export",
        help="Directory to write JSON files into (created if missing).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round embedding floats to this many decimals. Default: full precision.",
    )
    parser.add_argument(
        "--per-collection",
        action="store_true",
        help="Also write one full JSON file per collection.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact single-line files.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    indent = args.indent if args.indent > 0 else None

    client = get_client(args)
    collections = sorted(client.list_collections(), key=lambda c: c.name)
    if not collections:
        sys.exit("No collections found. Check the source / path / server settings.")

    provenance = build_provenance(args)

    full = {"_provenance": provenance}
    light = {"_provenance": provenance}

    print(f"chromadb {provenance['chromadb_version']} | {len(collections)} collection(s)")
    for col in collections:
        payload = export_collection(col, args.round)
        full[col.name] = payload
        light[col.name] = strip_embeddings(payload)
        print(f"  - {col.name}: {payload['count']} records, dim {payload['embedding_dim']}")

        if args.per_collection:
            p = os.path.join(args.out_dir, f"{col.name}.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=indent)

    full_path = os.path.join(args.out_dir, "chroma_export_full.json")
    light_path = os.path.join(args.out_dir, "chroma_export_no_embeddings.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=indent)
    with open(light_path, "w", encoding="utf-8") as f:
        json.dump(light, f, ensure_ascii=False, indent=indent)

    total = sum(v["count"] for k, v in full.items() if k != "_provenance")
    print(f"\nWrote {total} records to:")
    print(f"  {full_path}")
    print(f"  {light_path}")
    if args.per_collection:
        print(f"  + one file per collection in {args.out_dir}/")


if __name__ == "__main__":
    main()
