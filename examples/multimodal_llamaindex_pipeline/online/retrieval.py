from __future__ import annotations

from dataclasses import dataclass

from common.clients import MetadataStore, VectorStore
from common.schemas import QueryFilters, SearchRequest, SearchResult


@dataclass
class RetrievalService:
    metadata: MetadataStore
    vectors: VectorStore
    collections: dict[str, str]

    def parse_query(self, req: SearchRequest) -> tuple[str, QueryFilters]:
        # TODO: replace with LlamaIndex structured output parser
        semantic_query = req.query
        return semantic_query, QueryFilters()

    def embed_query(self, semantic_query: str) -> list[float]:
        by = semantic_query.encode("utf-8")
        return [float(by[i % len(by)] / 255.0) if by else 0.0 for i in range(32)]

    def search(self, req: SearchRequest) -> list[SearchResult]:
        semantic_query, filters = self.parse_query(req)
        query_vec = self.embed_query(semantic_query)
        collection = self.collections[req.entity_type.value]

        vec_hits = self.vectors.search(collection, query_vec, req.topk)
        filtered = set(self.metadata.filter_events(req.entity_type.value, filters))
        ranked = [(eid, s) for eid, s in vec_hits if not filtered or eid in filtered]

        results = self.metadata.load_results([eid for eid, _ in ranked])
        score_map = dict(ranked)
        for r in results:
            r.score = score_map.get(r.event_id, 0.0)
            r.evidence = {"semantic_query": semantic_query, "source": "vector+filter"}

        return sorted(results, key=lambda x: x.score, reverse=True)
