"""
scripts/ingest.py   [CHANGE 1 + CHANGE 5 — modified]
──────────────────
Changes vs original:
  Change 1: ColPali now indexes per-element crops (image/table/chart)
            instead of full-page screenshots. Text-only pages produce
            zero ColPali entries — no wasted embedding.
  Change 5: File-level SHA-256 dedup via DocRegistry (shared Postgres table).
            If an identical file was already ingested on any machine, it is
            skipped with a clear log message. Modified files (same name,
            new content) are re-ingested in full.

Everything else (chunk pipeline, metadata extraction, parallel control,
manifest support) is unchanged.
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
# Change 5
from ingestion.doc_registry import DocRegistry
# Change 1
from ingestion.element_classifier import extract_visual_elements
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

_parser       = DocumentParser()
FILE_CONCURRENCY = 4


# ── Chunk-level dedup (unchanged from original) ───────────────────────────────

async def filter_new_chunks(chunks: list, indexer: DocumentIndexer) -> tuple[list, int]:
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
    return new_chunks, len(chunks) - len(new_chunks)


# ── Pipeline A: chunk embedding (unchanged) ───────────────────────────────────

async def ingest_file_chunks(
    file_path: Path, metadata: dict, indexer: DocumentIndexer
) -> dict:
    parsed    = _parser.parse(file_path)
    chunker   = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        project_metadata=metadata,
    )
    all_chunks            = chunker.chunk_document(parsed)
    new_chunks, skipped   = await filter_new_chunks(all_chunks, indexer)
    if not new_chunks:
        return {"indexed": 0, "skipped": skipped}
    embedded = await Embedder().embed_chunks(new_chunks)
    await indexer.index_chunks(embedded)
    return {"indexed": len(new_chunks), "skipped": skipped}


# ── Pipeline B: ColPali element-level (Change 1) ──────────────────────────────

def ingest_file_colpali_elements(
    file_path: Path,
    metadata: dict,
    colpali_embedder: "ColPaliEmbedder",
) -> list:
    """
    Change 1: Extract only visual elements (image/table/chart crops),
    then embed each crop with ColPali. Text-only pages are skipped.
    """
    elements     = extract_visual_elements(file_path, metadata)
    if not elements:
        return []

    # Embed each crop through ColPali (sync, CPU/GPU)
    results = []
    for elem in elements:
        try:
            from PIL import Image
            import base64, io
            img_bytes = base64.b64decode(elem.image_b64)
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # embed_pages expects a list of (page, vec) — we build one element at a time
            # to reuse the existing embed_pages path cleanly
            import torch
            colpali_embedder._get_model()
            batch  = colpali_embedder._processor.process_images([pil_img]).to(
                colpali_embedder._device
            )
            with torch.no_grad():
                emb = colpali_embedder._model(**batch)
            vec = emb[0].float().mean(dim=0).tolist()
            results.append((elem, vec))
        except Exception as e:
            print(f"   ⚠️  ColPali embed failed for element {elem.element_idx} "
                  f"(page {elem.page_idx}): {e}")

    return results


# ── Per-file orchestration ────────────────────────────────────────────────────

async def ingest_file(
    file_path: Path,
    indexer: DocumentIndexer,
    registry: DocRegistry,           # Change 5
    colpali_embedder,
    colpali_indexer,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        print(f"\n📄 {file_path.name}")

        # ── Change 5: document-level dedup check ──────────────────────────
        status = await registry.check(file_path)

        if status.already_ingested:
            print(f"   ⏭️  SKIPPED — identical content already ingested "
                  f"(hash: {status.existing_hash[:12]}…)")
            return {
                "file": file_path.name, "chunks": 0, "skipped": 0,
                "pages": 0, "errors": [], "deduped": True,
            }

        if status.same_name_new_hash:
            print(f"   🔄 Same filename, new content detected — "
                  f"re-ingesting (old hash: {status.existing_hash[:12]}…)")
            await registry.remove(status.existing_hash)

        # ── Metadata extraction (unchanged) ───────────────────────────────
        try:
            parsed_preview = _parser.parse(file_path)
            first_text = "".join(tc.text[:500] for tc in parsed_preview.text_chunks)[:3000]
            metadata   = await extract_metadata_from_document(first_text, file_path)
            print(f"   Metadata: {metadata}")
        except Exception as e:
            print(f"   ⚠️  Metadata extraction failed: {e}. Using defaults.")
            metadata = {
                "engagement_id": "ME-AUTO-2024-001",
                "client": "Unknown", "country": "GCC",
                "practice": "General", "year": 2024,
            }

        results = {
            "file": file_path.name, "chunks": 0, "skipped": 0,
            "pages": 0, "errors": [], "deduped": False,
        }

        # ── Pipeline A: chunk embedding (unchanged) ───────────────────────
        try:
            cr = await ingest_file_chunks(file_path, metadata, indexer)
            results["chunks"]  = cr["indexed"]
            results["skipped"] = cr["skipped"]
            skip_msg = f" ({cr['skipped']} duplicates skipped)" if cr["skipped"] else ""
            print(f"   ✅ Chunks: {cr['indexed']} indexed{skip_msg}")
        except Exception as e:
            results["errors"].append(f"Chunk pipeline: {e}")
            print(f"   ❌ Chunk pipeline failed: {e}")

        # ── Pipeline B: ColPali element-level (Change 1) ──────────────────
        if COLPALI_ENABLED and colpali_embedder and colpali_indexer:
            try:
                loop = asyncio.get_event_loop()
                element_vectors = await loop.run_in_executor(
                    None, ingest_file_colpali_elements,
                    file_path, metadata, colpali_embedder,
                )
                if element_vectors:
                    await colpali_indexer.index_elements(element_vectors)
                    results["pages"] = len(element_vectors)
                    print(f"   ✅ ColPali: {len(element_vectors)} visual elements indexed "
                          f"(image/table/chart crops only)")
                else:
                    print(f"   ⏭️  ColPali: no visual elements found in this document")
            except Exception as e:
                results["errors"].append(f"ColPali pipeline: {e}")
                print(f"   ❌ ColPali pipeline failed: {e}")
        else:
            print("   ⏭️  ColPali skipped")

        # ── Change 5: record in registry after successful ingest ──────────
        if not results["errors"]:
            await registry.record(file_path, metadata)
            print(f"   📋 Registered in doc_registry")

        return results


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser(description="EY RAG ingestion pipeline")
    ap.add_argument("--path",     default="data", help="Folder containing documents")
    ap.add_argument("--manifest", default=None,   help="Optional JSON manifest file")
    args = ap.parse_args()

    data_dir = Path(args.path)
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    supported = {".pdf", ".pptx", ".docx", ".xlsx"}
    files     = [f for f in data_dir.iterdir() if f.suffix.lower() in supported]

    if args.manifest:
        mp = Path(args.manifest)
        if mp.exists():
            manifest_names = {Path(e["file_path"]).name for e in json.loads(mp.read_text())}
            files = [f for f in files if f.name in manifest_names]
            print(f"📋 Manifest loaded — {len(files)} file(s)")
        else:
            print(f"⚠️  Manifest not found: {mp}. Processing all files.")

    if not files:
        print(f"❌ No supported files in {data_dir}")
        return

    print(f"📚 {len(files)} document(s)  |  ColPali: {COLPALI_ENABLED}")

    indexer  = DocumentIndexer()
    await indexer.init_db()

    # Change 5: initialise registry
    registry = DocRegistry(indexer.engine)
    await registry.init()

    colpali_embedder = None
    colpali_indexer  = None

    if COLPALI_ENABLED:
        try:
            colpali_embedder = ColPaliEmbedder()
            colpali_indexer  = ColPaliIndexer()
            await colpali_indexer.init_db()
            print("⏳ Loading ColPali model once…")
            colpali_embedder._get_model()
            print("✅ ColPali ready")
        except Exception as e:
            print(f"⚠️  ColPali init failed: {e}. Chunk pipeline only.")
            colpali_embedder = None
            colpali_indexer  = None

    effective_concurrency = 1 if (COLPALI_ENABLED and colpali_embedder) else FILE_CONCURRENCY
    sem  = asyncio.Semaphore(effective_concurrency)
    tasks = [
        ingest_file(f, indexer, registry, colpali_embedder, colpali_indexer, sem)
        for f in files
    ]
    all_results = await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for r in all_results:
        if r.get("deduped"):
            print(f"  ⏭️  {r['file']}: skipped (already ingested)")
        else:
            status = "✅" if not r["errors"] else "⚠️"
            print(f"  {status} {r['file']}: {r['chunks']} chunks | "
                  f"{r['skipped']} skipped | {r['pages']} ColPali elements")
            for err in r["errors"]:
                print(f"      ❌ {err}")

    print(f"\n  Total new chunks  : {sum(r['chunks']  for r in all_results)}")
    print(f"  Total deduped     : {sum(1 for r in all_results if r.get('deduped'))}")
    print(f"  Total ColPali elem: {sum(r['pages']   for r in all_results)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())