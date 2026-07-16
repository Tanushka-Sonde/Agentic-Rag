"""
scripts/ingest.py
──────────────────
Batch ingestion pipeline.

Pipeline A: Parse → Chunk → Deduplicate → Embed (OpenAI) → Index (Pinecone + Postgres)
Pipeline B: ColPali page-level visual indexing (optional)

DEDUPLICATION:
  chunk_id is a hash of (source_file, kind, page, index).
  Already-indexed chunks are skipped — safe to re-run on the same folder.

Usage:
    python scripts/ingest.py --path data/
    python scripts/ingest.py --path data/ --manifest scripts/manifest.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.indexer import DocumentIndexer
from ingestion.metadata_extractor import extract_metadata_from_document
from config.settings import get_settings

settings = get_settings()

try:
    from ingestion.colpali_embedder import ColPaliEmbedder
    from ingestion.colpali_indexer import ColPaliIndexer
    COLPALI_ENABLED = settings.colpali_enabled
    if not COLPALI_ENABLED:
        print("ℹ️  ColPali disabled by configuration.")
except ImportError:
    COLPALI_ENABLED = False
    print("⚠️  ColPali not available. Running chunk pipeline only.")

_parser = DocumentParser()

FILE_CONCURRENCY = 4   # max documents processed in parallel


# ── Deduplication ─────────────────────────────────────────────────────────────

async def filter_new_chunks(chunks: list, indexer: DocumentIndexer) -> tuple[list, int]:
    """Return only chunks not yet present in Postgres."""
    if not chunks:
        return [], 0

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession

    candidate_ids = [c.chunk_id for c in chunks]
    placeholders  = ", ".join(f":id_{i}" for i in range(len(candidate_ids)))
    params        = {f"id_{i}": cid for i, cid in enumerate(candidate_ids)}

    async with AsyncSession(indexer.engine) as session:
        result = await session.execute(
            text(f"SELECT chunk_id FROM chunks WHERE chunk_id IN ({placeholders})"),
            params,
        )
        existing_ids = {row[0] for row in result.fetchall()}
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    skipped    = len(chunks) - len(new_chunks)
    return new_chunks, skipped


# ── Pipeline A ────────────────────────────────────────────────────────────────

async def ingest_file_chunks(file_path: Path, metadata: dict, indexer: DocumentIndexer) -> dict:
    parsed = _parser.parse(file_path)
    chunker = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        project_metadata=metadata,
    )
    all_chunks = chunker.chunk_document(parsed)
    new_chunks, skipped = await filter_new_chunks(all_chunks, indexer)

    if not new_chunks:
        return {"indexed": 0, "skipped": skipped}

    embedder = Embedder()
    embedded = await embedder.embed_chunks(new_chunks)
    await indexer.index_chunks(embedded)
    return {"indexed": len(new_chunks), "skipped": skipped}


# ── Pipeline B ────────────────────────────────────────────────────────────────

def ingest_file_colpali(
    file_path: Path,
    metadata: dict,
    colpali_embedder: "ColPaliEmbedder",
) -> list:
    pages        = colpali_embedder.extract_pages(file_path, metadata)
    page_vectors = colpali_embedder.embed_pages(pages)
    return page_vectors


# ── Per-file orchestration ────────────────────────────────────────────────────

async def ingest_file(
    file_path: Path,
    indexer: DocumentIndexer,
    colpali_embedder: "ColPaliEmbedder | None",
    colpali_indexer:  "ColPaliIndexer | None",
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        print(f"\n📄 Ingesting: {file_path.name}")

        # Step 1: Metadata extraction
        try:
            parsed_preview = _parser.parse(file_path)
            first_text = ""
            for tc in parsed_preview.text_chunks:
                first_text += tc.text[:500]
                if len(first_text) > 3000:
                    break
            metadata = await extract_metadata_from_document(first_text, file_path)
            print(f"   Metadata: {metadata}")
        except Exception as e:
            print(f"   ⚠️  Metadata extraction failed: {e}. Using defaults.")
            metadata = {
                "engagement_id": "ME-AUTO-2024-001",
                "client":        "Unknown",
                "country":       "GCC",
                "practice":      "General",
                "year":          2024,
            }

        results = {"file": file_path.name, "chunks": 0, "skipped": 0, "pages": 0, "errors": []}

        # Pipeline A
        try:
            chunk_result = await ingest_file_chunks(file_path, metadata, indexer)
            results["chunks"]  = chunk_result["indexed"]
            results["skipped"] = chunk_result["skipped"]
            skip_msg = f" ({chunk_result['skipped']} duplicates skipped)" if chunk_result["skipped"] else ""
            print(f"   ✅ Chunks: {chunk_result['indexed']} indexed{skip_msg}")
        except Exception as e:
            results["errors"].append(f"Chunk pipeline: {e}")
            print(f"   ❌ Chunk pipeline failed: {e}")

        # Pipeline B
        if COLPALI_ENABLED and colpali_embedder and colpali_indexer:
            try:
                loop = asyncio.get_event_loop()
                page_vectors = await loop.run_in_executor(
                    None, ingest_file_colpali, file_path, metadata, colpali_embedder,
                )
                colpali_indexer._upsert_pinecone(page_vectors)
                await colpali_indexer._upsert_postgres(page_vectors)
                results["pages"] = len(page_vectors)
                print(f"   ✅ ColPali: {len(page_vectors)} pages indexed")
            except Exception as e:
                results["errors"].append(f"ColPali pipeline: {e}")
                print(f"   ❌ ColPali pipeline failed: {e}")
        else:
            print("   ⏭️  ColPali skipped")

        return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser(description="EY RAG ingestion pipeline")
    ap.add_argument("--path",     default="data",  help="Folder containing documents")
    ap.add_argument("--manifest", default=None,    help="Optional JSON manifest file")
    args = ap.parse_args()

    data_dir = Path(args.path)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    supported = {".pdf", ".pptx", ".docx", ".xlsx"}
    files = [f for f in data_dir.iterdir() if f.suffix.lower() in supported]

    if args.manifest:
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest_names = {Path(e["file_path"]).name for e in manifest}
            files = [f for f in files if f.name in manifest_names]
            print(f"📋 Manifest loaded — restricting to {len(files)} file(s)")
        else:
            print(f"⚠️  Manifest not found: {manifest_path}. Processing all files.")

    if not files:
        print(f"❌ No supported files found in {data_dir}")
        return

    print(f"📚 Found {len(files)} document(s)  |  ColPali: {COLPALI_ENABLED}")

    indexer = DocumentIndexer()
    await indexer.init_db()

    colpali_embedder = None
    colpali_indexer  = None

    if COLPALI_ENABLED:
        try:
            colpali_embedder = ColPaliEmbedder()
            colpali_indexer  = ColPaliIndexer()
            await colpali_indexer.init_db()
            # Load the model ONCE up front so the parallel files don't each
            # trigger a duplicate ~5GB load (the earlier race condition).
            print("⏳ Loading ColPali model once (CPU — this is slow)…")
            colpali_embedder._get_model()
            print("✅ ColPali indexer ready")
        except Exception as e:
            print(f"⚠️  ColPali init failed: {e}. Chunk pipeline only.")
            colpali_embedder = None
            colpali_indexer  = None

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # ColPali shares one non-thread-safe model on CPU, so serialize files when
    # it's enabled. Otherwise process several files in parallel for speed.
    effective_concurrency = 1 if (COLPALI_ENABLED and colpali_embedder) else FILE_CONCURRENCY
    print(f"⚙️  File concurrency: {effective_concurrency}")
    sem = asyncio.Semaphore(effective_concurrency)
    tasks = [
        ingest_file(f, indexer, colpali_embedder, colpali_indexer, sem)
        for f in files
    ]
    all_results = await asyncio.gather(*tasks)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("INGESTION SUMMARY")
    print("=" * 55)
    for r in all_results:
        status = "✅" if not r["errors"] else "⚠️"
        print(f"  {status} {r['file']}: {r['chunks']} new | {r['skipped']} skipped | {r['pages']} ColPali pages")
        for err in r["errors"]:
            print(f"      ❌ {err}")

    total_chunks  = sum(r["chunks"]  for r in all_results)
    total_skipped = sum(r["skipped"] for r in all_results)
    total_pages   = sum(r["pages"]   for r in all_results)
    print(f"\n  Total: {total_chunks} new chunks | {total_skipped} skipped | {total_pages} ColPali pages")
    print("=" * 55)
    print("\n✅ Data is indexed in Pinecone + Postgres.")
    print("   Others can query via the API — no re-embedding needed.")


if __name__ == "__main__":
    asyncio.run(main())
