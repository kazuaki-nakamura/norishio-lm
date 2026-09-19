from copy import deepcopy

import pytest

from norishio_lm import benchmark_v3_data as benchmark
from norishio_lm.benchmark_v3_audit import audit_frozen_bundle


def test_frozen_bundle_passes_post_freeze_audit() -> None:
    audit_frozen_bundle(benchmark.build())


def test_cross_row_source_target_overlap_is_rejected() -> None:
    bundle = benchmark.build()
    changed = deepcopy(bundle)
    changed["train"][1]["inputs"]["text"] = changed["train"][0]["targets"]["text"]
    with pytest.raises(ValueError, match="source and target text sets overlap"):
        audit_frozen_bundle(changed)


def test_duplicate_target_is_rejected() -> None:
    bundle = benchmark.build()
    changed = deepcopy(bundle)
    changed["train"][1]["targets"]["text"] = changed["train"][0]["targets"]["text"]
    with pytest.raises(ValueError, match="duplicate target text"):
        audit_frozen_bundle(changed)
