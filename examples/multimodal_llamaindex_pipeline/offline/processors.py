from __future__ import annotations

import hashlib
from datetime import UTC

from common.schemas import EntityEvent, EntityType, OfflineRecord


def detect_entities(record: OfflineRecord) -> list[tuple[EntityType, tuple[int, int, int, int]]]:
    _ = record
    return [(EntityType.VEHICLE, (0, 0, 224, 224))]


def extract_attributes(entity_type: EntityType, image_path: str) -> dict:
    _ = image_path
    if entity_type == EntityType.PERSON:
        return {"upper_color": "unknown", "carrying_bag": False}
    if entity_type == EntityType.VEHICLE:
        return {"vehicle_type": "car", "body_color": "unknown"}
    return {"object_class": "unknown"}


def generate_embedding(seed: str, dim: int = 32) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return [digest[i % len(digest)] / 255.0 for i in range(dim)]


def build_event(record: OfflineRecord, entity_type: EntityType, bbox: tuple[int, int, int, int], crop_uri: str) -> EntityEvent:
    event_key = f"{record.source_id}-{entity_type.value}-{bbox}"
    event_id = hashlib.md5(event_key.encode("utf-8")).hexdigest()  # noqa: S324
    attr = extract_attributes(entity_type, record.image_path)

    ts = record.ts_utc if record.ts_utc.tzinfo else record.ts_utc.replace(tzinfo=UTC)
    emb_seed = f"{entity_type.value}|{record.image_path}|{attr}"

    return EntityEvent(
        event_id=event_id,
        entity_type=entity_type,
        source_id=record.source_id,
        camera_id=record.camera_id,
        ts_utc=ts,
        image_uri=record.image_path,
        crop_uri=crop_uri,
        bbox=bbox,
        attr=attr,
        embedding=generate_embedding(emb_seed),
    )
