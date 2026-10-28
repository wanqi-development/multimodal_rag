"""Milvus Lite 的真实兼容性回归：BM25 + dense + hybrid search。"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_lite_supports_current_hybrid_schema_and_search(tmp_path: Path) -> None:
    """在独立子进程里跑，进程退出后 Windows 才会释放 Milvus Lite 的文件锁。"""
    db = tmp_path / "milvus.db"
    code = textwrap.dedent(
        """
        import asyncio
        import sys

        from rag.config.schema import VectorIndexConfig
        from rag.contracts import Chunk, Embeddings
        from rag.indexing.milvus.versioned import DualVersionMilvusIndex


        class Registry:
            def __init__(self) -> None:
                self.batches = {}

            async def active_batches(self):
                return dict(self.batches)


        async def main() -> None:
            cfg = VectorIndexConfig(
                backend="milvus-lite",
                uri=sys.argv[1],
                collection="probe",
                versions_enabled=True,
            )
            assert cfg.bm25.analyzer == "jieba"
            registry = Registry()
            index = DualVersionMilvusIndex(cfg, dim=3, registry=registry)
            await index.ensure_ready()
            await index.upsert_batch(
                [
                    Chunk(chunk_id="a", doc_id="d1", text="alpha beta"),
                    Chunk(chunk_id="b", doc_id="d2", text="gamma delta"),
                    Chunk(chunk_id="c", doc_id="d3", text="alpha gamma"),
                ],
                Embeddings(
                    dense=[
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.7, 0.7, 0.0],
                    ]
                ),
                batch_id="batch-1",
            )
            registry.batches = {"d1": "batch-1", "d2": "batch-1", "d3": "batch-1"}
            dense = [hit.chunk.chunk_id for hit in await index.search_dense([1, 0, 0], 2)]
            sparse = [hit.chunk.chunk_id for hit in await index.search_sparse("alpha", 2)]
            hybrid = [hit.chunk.chunk_id for hit in await index.search_hybrid(
                [1, 0, 0], "alpha", top_k=2
            )]
            assert dense == ["a", "c"], dense
            assert sparse == ["a", "c"], sparse
            assert hybrid == ["a", "c"], hybrid
            print("milvus-lite-hybrid-ok")


        asyncio.run(main())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(db)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "milvus-lite-hybrid-ok" in result.stdout
