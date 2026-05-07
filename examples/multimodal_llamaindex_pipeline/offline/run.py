from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from common.clients import MetadataStore, ObjectStorage, VectorStore
from common.schemas import OfflineRecord
from offline.processors import build_event, detect_entities


def load_manifest(path: str) -> list[OfflineRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(OfflineRecord.model_validate(payload))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    metadata = MetadataStore(cfg["storage"]["metadata_db"])
    vectors = VectorStore(cfg["vector"]["root"])
    objects = ObjectStorage(cfg["storage"]["media_root"])

    records = load_manifest(args.manifest)
    events = []
    for record in records:
        for entity_type, bbox in detect_entities(record):
            crop_uri = objects.save_crop(record.image_path, bbox)
            events.append(build_event(record, entity_type, bbox, crop_uri))

    metadata.upsert_events(events)
    for key, collection in cfg["vector"]["collections"].items():
        typed_events = [e for e in events if e.entity_type.value == key]
        vectors.upsert_embeddings(collection, typed_events)

    print(f"Indexed {len(events)} events from {len(records)} records")


if __name__ == "__main__":
    main()
