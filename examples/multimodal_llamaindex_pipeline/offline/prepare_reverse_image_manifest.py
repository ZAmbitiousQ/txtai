from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to extracted reverse_image_search images directory")
    parser.add_argument("--output", required=True, help="Output jsonl manifest path")
    args = parser.parse_args()

    image_dir = Path(args.images)
    files = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    now = datetime.now(UTC).isoformat()
    with open(args.output, "w", encoding="utf-8") as out:
        for i, file in enumerate(files):
            row = {
                "source_id": f"reverse-image-{i}",
                "image_path": str(file.resolve()),
                "camera_id": "reverse_image_search",
                "ts_utc": now,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(files)} records to {args.output}")


if __name__ == "__main__":
    main()
