"""Vector memory — persistent long-term storage powered by ChromaDB."""

from __future__ import annotations

import uuid
from typing import Any

from ..utils.logger import get_logger


class VectorMemory:
    """Persistent vector store for long-term agent memory.

    Stores text snippets as embeddings in a local ChromaDB collection.
    Useful for remembering facts, preferences, and learnings across sessions.
    """

    def __init__(self, persist_dir: str = "./data/vector_store") -> None:
        self._log = get_logger(__name__)
        self._persist_dir = persist_dir

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="agent_memory",
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
            self._log.info(f"Vector store ready at {persist_dir}")
        except ImportError:
            self._log.warning(
                "chromadb not installed — vector memory disabled. "
                "Install with: pip install chromadb"
            )
            self._client = None  # type: ignore[assignment]
            self._collection = None  # type: ignore[assignment]
            self._ready = False
        except Exception as exc:
            self._log.error(f"Failed to initialize ChromaDB: {exc}")
            self._ready = False
            self._client = None  # type: ignore[assignment]
            self._collection = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether the vector store is operational."""
        return self._ready

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        """Store a text snippet and return its ID."""
        if not self._ready:
            return ""
        doc_id = str(uuid.uuid4())
        try:
            self._collection.add(  # type: ignore[union-attr]
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata or {}],
            )
            self._log.debug(f"Stored vector memory: {doc_id}")
        except Exception as exc:
            self._log.error(f"Failed to add vector memory: {exc}")
        return doc_id

    def query(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search for similar memories."""
        if not self._ready:
            return []
        try:
            results = self._collection.query(  # type: ignore[union-attr]
                query_texts=[query],
                n_results=top_k,
            )
        except Exception as exc:
            self._log.error(f"Vector query failed: {exc}")
            return []

        items: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            items.append({
                "id": doc_id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": distances[i] if i < len(distances) else 0.0,
            })
        return items

    def delete(self, doc_id: str) -> bool:
        """Remove a memory by ID."""
        if not self._ready:
            return False
        try:
            self._collection.delete(ids=[doc_id])  # type: ignore[union-attr]
            return True
        except Exception as exc:
            self._log.error(f"Failed to delete vector memory: {exc}")
            return False

    def count(self) -> int:
        """Number of stored memories."""
        if not self._ready:
            return 0
        try:
            return self._collection.count()  # type: ignore[union-attr]
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all vector memories."""
        if not self._ready:
            return
        try:
            all_ids = self._collection.get()["ids"]  # type: ignore[union-attr]
            if all_ids:
                self._collection.delete(ids=all_ids)  # type: ignore[union-attr]
            self._log.info("Cleared all vector memories.")
        except Exception as exc:
            self._log.error(f"Failed to clear vector store: {exc}")
