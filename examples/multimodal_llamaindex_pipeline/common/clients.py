from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.schemas import EntityEvent, QueryFilters, SearchResult


@dataclass
class MetadataStore:
    dsn: str

    def __post_init__(self) -> None:
        self.path = self.dsn.removeprefix("sqlite:///")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  event_id TEXT PRIMARY KEY,
                  entity_type TEXT,
                  source_id TEXT,
                  camera_id TEXT,
                  ts_utc TEXT,
                  image_uri TEXT,
                  crop_uri TEXT,
                  attr_json TEXT
                )
                """
            )

    def upsert_events(self, events: Iterable[EntityEvent]) -> None:
        rows = [
            (
                e.event_id,
                e.entity_type.value,
                e.source_id,
                e.camera_id,
                e.ts_utc.isoformat(),
                e.image_uri,
                e.crop_uri,
                json.dumps(e.attr, ensure_ascii=False),
            )
            for e in events
        ]
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                """
                INSERT INTO events(event_id, entity_type, source_id, camera_id, ts_utc, image_uri, crop_uri, attr_json)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(event_id) DO UPDATE SET
                  entity_type=excluded.entity_type,
                  source_id=excluded.source_id,
                  camera_id=excluded.camera_id,
                  ts_utc=excluded.ts_utc,
                  image_uri=excluded.image_uri,
                  crop_uri=excluded.crop_uri,
                  attr_json=excluded.attr_json
                """,
                rows,
            )

    def filter_events(self, entity_type: str, filters: QueryFilters) -> list[str]:
        clauses = ["entity_type = ?"]
        params: list[str] = [entity_type]
        if filters.time_start:
            clauses.append("ts_utc >= ?")
            params.append(filters.time_start.isoformat())
        if filters.time_end:
            clauses.append("ts_utc <= ?")
            params.append(filters.time_end.isoformat())
        if filters.camera_ids:
            placeholders = ",".join("?" for _ in filters.camera_ids)
            clauses.append(f"camera_id IN ({placeholders})")
            params.extend(filters.camera_ids)

        where = " AND ".join(clauses)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(f"SELECT event_id FROM events WHERE {where}", params).fetchall()
        return [row[0] for row in rows]

    def load_results(self, event_ids: list[str]) -> list[SearchResult]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT event_id, ts_utc, camera_id, image_uri, crop_uri, attr_json FROM events WHERE event_id IN ({placeholders})",
                event_ids,
            ).fetchall()
        return [
            SearchResult(
                event_id=r[0],
                score=0.0,
                ts_utc=r[1],
                camera_id=r[2],
                image_uri=r[3],
                crop_uri=r[4],
                attr=json.loads(r[5]) if r[5] else {},
            )
            for r in rows
        ]


@dataclass
class VectorStore:
    root: str

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    def _path(self, collection: str) -> str:
        return os.path.join(self.root, f"{collection}.npz")

    def upsert_embeddings(self, collection: str, events: Iterable[EntityEvent]) -> None:
        path = self._path(collection)
        ids = []
        vectors = []
        if os.path.exists(path):
            existing = np.load(path, allow_pickle=True)
            ids = existing["ids"].tolist()
            vectors = existing["vectors"].tolist()

        for e in events:
            if not e.embedding:
                continue
            if e.event_id in ids:
                idx = ids.index(e.event_id)
                vectors[idx] = e.embedding
            else:
                ids.append(e.event_id)
                vectors.append(e.embedding)

        np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=np.array(vectors, dtype=np.float32))

    def search(self, collection: str, query_vector: list[float], topk: int) -> list[tuple[str, float]]:
        path = self._path(collection)
        if not os.path.exists(path):
            return []
        data = np.load(path, allow_pickle=True)
        ids = data["ids"].tolist()
        vectors = data["vectors"].astype(np.float32)
        if len(ids) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        scores = vectors @ q
        order = np.argsort(scores)[::-1][:topk]
        return [(ids[i], float(scores[i])) for i in order]


@dataclass
class ObjectStorage:
    media_root: str

    def __post_init__(self) -> None:
        os.makedirs(self.media_root, exist_ok=True)

    def save_crop(self, source_path: str, bbox: tuple[int, int, int, int] | None) -> str:
        _ = bbox
        return source_path
