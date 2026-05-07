from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    OBJECT = "object"


class OfflineRecord(BaseModel):
    source_id: str
    image_path: str
    camera_id: str
    ts_utc: datetime
    longitude: float | None = None
    latitude: float | None = None


class EntityEvent(BaseModel):
    event_id: str
    entity_type: EntityType
    source_id: str
    camera_id: str
    ts_utc: datetime
    image_uri: str
    crop_uri: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    attr: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None


class QueryFilters(BaseModel):
    time_start: datetime | None = None
    time_end: datetime | None = None
    camera_ids: list[str] = Field(default_factory=list)
    district_codes: list[str] = Field(default_factory=list)
    attrs: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    entity_type: EntityType
    topk: int = 200


class SearchResult(BaseModel):
    event_id: str
    score: float
    image_uri: str
    crop_uri: str | None = None
    ts_utc: datetime
    camera_id: str
    attr: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
