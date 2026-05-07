from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
import yaml
from fastapi import FastAPI

from common.clients import MetadataStore, VectorStore
from common.schemas import SearchRequest, SearchResult
from online.retrieval import RetrievalService

app = FastAPI(title="Multimodal Text-to-Image Search")
service: RetrievalService | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest) -> list[SearchResult]:
    if service is None:
        return []
    return service.search(req)


def build_service(config_path: str) -> RetrievalService:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return RetrievalService(
        metadata=MetadataStore(cfg["storage"]["metadata_db"]),
        vectors=VectorStore(cfg["vector"]["root"]),
        collections=cfg["vector"]["collections"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global service
    service = build_service(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
