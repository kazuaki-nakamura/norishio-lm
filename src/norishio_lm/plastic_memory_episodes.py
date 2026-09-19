"""Deterministic authored episodes for the plastic-memory toy experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class MemoryExample:
    example_id: str
    key: tuple[float, ...]
    value: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.example_id or not self.key or not self.value:
            raise ValueError("memory examples require an id, key, and value")
        if any(type(item) not in {int, float} for item in (*self.key, *self.value)):
            raise TypeError("memory example vectors must be numeric")


@dataclass(frozen=True)
class MemoryEpisode:
    episode_id: str
    support: tuple[MemoryExample, ...]
    queries: tuple[MemoryExample, ...]

    def __post_init__(self) -> None:
        if not self.episode_id or len(self.support) < 1 or len(self.queries) < 1:
            raise ValueError("episodes require an id, support, and queries")
        identifiers = [item.example_id for item in (*self.support, *self.queries)]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("support and query IDs must be disjoint")
        key_dims = {len(item.key) for item in (*self.support, *self.queries)}
        value_dims = {len(item.value) for item in (*self.support, *self.queries)}
        if len(key_dims) != 1 or len(value_dims) != 1:
            raise ValueError("episode vector dimensions must be consistent")

    @property
    def key_dim(self) -> int:
        return len(self.support[0].key)

    @property
    def value_dim(self) -> int:
        return len(self.support[0].value)


def build_memory_episodes(seed: int = 7) -> tuple[MemoryEpisode, MemoryEpisode]:
    """Return two disjoint authored episodes in a seeded support order."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    raw = (
        (
            "episode-a",
            (
                MemoryExample("a-support-0", (1.0, 0.0, 0.0, 0.0), (1.0, 0.0)),
                MemoryExample("a-support-1", (0.0, 1.0, 0.0, 0.0), (0.0, 1.0)),
            ),
            (
                MemoryExample("a-query-0", (1.0, 0.05, 0.0, 0.0), (1.0, 0.0)),
                MemoryExample("a-query-1", (0.05, 1.0, 0.0, 0.0), (0.0, 1.0)),
            ),
        ),
        (
            "episode-b",
            (
                MemoryExample("b-support-0", (0.0, 0.0, 1.0, 0.0), (1.0, 0.0)),
                MemoryExample("b-support-1", (0.0, 0.0, 0.0, 1.0), (0.0, 1.0)),
            ),
            (
                MemoryExample("b-query-0", (0.0, 0.0, 1.0, 0.05), (1.0, 0.0)),
                MemoryExample("b-query-1", (0.0, 0.0, 0.05, 1.0), (0.0, 1.0)),
            ),
        ),
    )
    rng = random.Random(seed)
    episodes: list[MemoryEpisode] = []
    for episode_id, support, queries in raw:
        support_order = list(support)
        rng.shuffle(support_order)
        episodes.append(MemoryEpisode(episode_id, tuple(support_order), queries))
    return tuple(episodes)  # type: ignore[return-value]


def episode_digest(episodes: tuple[MemoryEpisode, ...]) -> str:
    payload = [asdict(episode) for episode in episodes]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["MemoryExample", "MemoryEpisode", "build_memory_episodes", "episode_digest"]
