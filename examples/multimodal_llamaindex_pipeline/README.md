# Multimodal LlamaIndex Pipeline Scaffold

This project provides an end-to-end framework for **offline ingestion/indexing** and **online retrieval** for text-to-image tasks (person/object/vehicle).

## Project layout

- `config/settings.yaml`: runtime configuration
- `common/`: shared schemas and storage clients
- `offline/`: batch pipeline (manifest -> event -> index)
- `online/`: FastAPI online retrieval service
- `Dockerfile` / `docker-compose.yml`: container deployment

## 1) Prepare and test with reverse_image_search dataset

```bash
cd examples/multimodal_llamaindex_pipeline
mkdir -p datasets data
wget https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip -O datasets/reverse_image_search.zip
unzip -o datasets/reverse_image_search.zip -d datasets/
python -m offline.prepare_reverse_image_manifest \
  --images datasets/reverse_image_search \
  --output data/reverse_image_manifest.jsonl
python -m offline.run --config config/settings.yaml --manifest data/reverse_image_manifest.jsonl
```

## 2) Run online retrieval (local)

```bash
python -m online.api --config config/settings.yaml --host 0.0.0.0 --port 8000
```

Test query in another shell:

```bash
curl -X POST "http://127.0.0.1:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"red car", "entity_type":"vehicle", "topk":10}'
```

## 3) Docker deployment

### Build image

```bash
docker build -t multimodal-search:latest .
```

### Run container

```bash
docker run --rm -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  multimodal-search:latest
```

### Use docker compose

```bash
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

## Notes for production

- Replace `VectorStore` local numpy backend with Milvus/Qdrant.
- Replace SQLite metadata with PostgreSQL/ElasticSearch hybrid filters.
- Replace placeholder embeddings and query parsing with your online model stack and LlamaIndex structured parser.
