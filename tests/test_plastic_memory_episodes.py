from norishio_lm.plastic_memory_episodes import (
    MemoryEpisode,
    MemoryExample,
    build_memory_episodes,
    episode_digest,
)


def test_episode_fixture_is_deterministic_and_disjoint() -> None:
    first = build_memory_episodes(7)
    second = build_memory_episodes(7)
    assert first == second
    assert episode_digest(first) == episode_digest(second)
    assert len(episode_digest(first)) == 64
    assert {item.example_id for episode in first for item in episode.support}.isdisjoint(
        {item.example_id for episode in first for item in episode.queries}
    )
    assert all(episode.key_dim == 4 and episode.value_dim == 2 for episode in first)


def test_seed_changes_only_support_order() -> None:
    first = build_memory_episodes(7)
    second = build_memory_episodes(17)
    assert [[item.value for item in episode.queries] for episode in first] == [
        [item.value for item in episode.queries] for episode in second
    ]
    assert episode_digest(first) != episode_digest(second)


def test_schema_rejects_duplicate_ids_and_mismatched_dimensions() -> None:
    one = MemoryExample("same", (1.0, 0.0), (1.0,))
    duplicate = MemoryExample("same", (0.0, 1.0), (0.0,))
    try:
        MemoryEpisode("bad", (one,), (duplicate,))
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("duplicate IDs were accepted")

    wider = MemoryExample("query", (1.0, 0.0, 0.0), (1.0,))
    try:
        MemoryEpisode("bad", (one,), (wider,))
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("mismatched dimensions were accepted")
