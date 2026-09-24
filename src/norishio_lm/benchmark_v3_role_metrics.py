"""Raw-only factor-head metrics and frozen decision rules for Issue #48."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .benchmark_v3_head_metrics import aggregate_head_metrics
from .benchmark_v3_model import FACTOR_ORDER

ARMS = ("ROLE_ALIASED", "ROLE_DISJOINT")
SEEDS = (7, 17, 29)
# The shared raw metric reducer canonicalizes ``seen_pair/unseen_triple`` to
# this longer key; saved raw retains the Issue #48 support label verbatim.
STRATA = ("unseen_pair", "seen_pair/unseen_higher_order")
FACTORS = tuple(FACTOR_ORDER)


def _score(metrics: Mapping[str, Any], arm: str, seed: int, split: str,
           factor: str, stratum: str = "all") -> float:
    tree = metrics["by_arm_seed"][arm][str(seed)]
    field = (tree["train_resubstitution"] if split == "train" else
             tree["confirmation"][stratum])["fields"][factor]
    value = field["class_balanced_accuracy"]
    if not isinstance(value, (int, float)) or field["unavailable_count"]:
        raise ValueError(f"missing {arm} seed{seed} {split}/{stratum}/{factor} score")
    return float(value)


def summarize_role_heads(
    raw_rows: Sequence[Mapping[str, Any]],
    expected_classes: Mapping[str, Sequence[str]],
    aliased_train_ceilings: Mapping[str, float],
    *,
    strong_min: float = 0.80,
    ceiling_margin: float = 0.05,
    generalization_gap: float = 0.15,
    event_operator_noncollapse_min: float = 0.50,
    event_operator_distinct_classes_min: int = 2,
) -> dict[str, Any]:
    """Aggregate complete saved rows; never load a model or recompute logits.

    The caller must authenticate every raw row and attempt against the frozen
    fixture before invoking this pure reducer. The thresholds are frozen in
    the experiment descriptor; callers must pass those exact values.
    """
    if tuple(expected_classes) != FACTORS or set(aliased_train_ceilings) != set(FACTORS):
        raise ValueError("all canonical factors and aliased ceilings are required")
    if (len(raw_rows) != 6 * 480 or not 0 < strong_min <= 1
            or not 0 <= ceiling_margin <= 1 or not 0 <= generalization_gap <= 1
            or not 0 <= event_operator_noncollapse_min <= 1
            or event_operator_distinct_classes_min < 1):
        raise ValueError("incomplete rows or invalid frozen interpretation thresholds")
    metrics = aggregate_head_metrics(raw_rows, expected_classes, factors=FACTORS)
    if metrics["arms"] != list(ARMS) or metrics["seeds"] != list(SEEDS):
        raise ValueError("raw arm/seed coverage is incomplete")
    for arm in ARMS:
        for seed in SEEDS:
            group = metrics["by_arm_seed"][arm][str(seed)]
            if group["train_resubstitution"]["scheduled_count"] != 384:
                raise ValueError("train row count differs from freeze")
            if group["confirmation"]["all"]["scheduled_count"] != 96:
                raise ValueError("confirmation row count differs from freeze")
            if any(group["confirmation"][s]["scheduled_count"] != 48 for s in STRATA):
                raise ValueError("confirmation stratum count differs from freeze")

    decisions: dict[str, Any] = {}
    for factor in FACTORS:
        a = [_score(metrics, ARMS[0], seed, "train", factor) for seed in SEEDS]
        b = [_score(metrics, ARMS[1], seed, "train", factor) for seed in SEEDS]
        ceiling = aliased_train_ceilings[factor]
        if not isinstance(ceiling, (int, float)) or not 0 <= ceiling <= 1:
            raise ValueError("invalid aliased train ceiling")
        strong = all(value >= strong_min for value in b)
        paired_positive = all(right > left for left, right in zip(a, b, strict=True))
        b_mean = sum(b) / len(b)
        rescue = strong and paired_positive and b_mean >= ceiling + ceiling_margin
        confirmation: dict[str, Any] = {}
        for stratum in ("all", *STRATA):
            a_scores = [_score(metrics, ARMS[0], seed, "confirmation", factor, stratum) for seed in SEEDS]
            b_scores = [_score(metrics, ARMS[1], seed, "confirmation", factor, stratum) for seed in SEEDS]
            b_confirmation_mean = sum(b_scores) / len(b_scores)
            confirmation[stratum] = {
                "aliased_balanced_by_seed": dict(zip(map(str, SEEDS), a_scores, strict=True)),
                "disjoint_balanced_by_seed": dict(zip(map(str, SEEDS), b_scores, strict=True)),
                "disjoint_minus_aliased_by_seed": dict(zip(map(str, SEEDS),
                                                            [right - left for left, right in zip(a_scores, b_scores, strict=True)], strict=True)),
                "disjoint_mean": b_confirmation_mean,
                "train_minus_confirmation_mean": b_mean - b_confirmation_mean,
                "generalization_gap_if_rescued": bool(rescue and stratum != "all"
                                                       and b_mean - b_confirmation_mean >= generalization_gap),
                "all_disjoint_seeds_at_least_strong_min": all(value >= strong_min for value in b_scores),
            }
        decisions[factor] = {
            "aliased_train_ceiling": ceiling,
            "aliased_train_balanced_by_seed": dict(zip(map(str, SEEDS), a, strict=True)),
            "disjoint_train_balanced_by_seed": dict(zip(map(str, SEEDS), b, strict=True)),
            "disjoint_minus_aliased_train_by_seed": dict(zip(map(str, SEEDS),
                                                              [right - left for left, right in zip(a, b, strict=True)], strict=True)),
            "disjoint_train_mean": b_mean,
            "strong_train": strong,
            "paired_train_positive_all_seeds": paired_positive,
            "above_aliased_ceiling_with_margin": b_mean >= ceiling + ceiling_margin,
            "rescue": rescue,
            "confirmation": confirmation,
        }
    rescued = [factor for factor in FACTORS if decisions[factor]["rescue"]]
    if all(factor in rescued for factor in ("participant", "time")):
        branch = "participant_time_role_readout_rescued"
    elif len(rescued) == 1:
        branch = "one_factor_rescued_narrow_follow_up"
    elif not all(decisions[factor]["strong_train"] for factor in ("participant", "time")):
        branch = "residual_optimization_capacity_conditioning_unresolved"
    else:
        branch = "mixed_or_inconclusive"
    event_operator_noncollapsed: dict[str, bool] = {}
    for factor in ("event", "operator"):
        noncollapsed = True
        for seed in SEEDS:
            for stratum in STRATA:
                score = _score(metrics, "ROLE_DISJOINT", seed, "confirmation", factor, stratum)
                predictions = {
                    row["predicted_frame"][factor] for row in raw_rows
                    if row["arm"] == "ROLE_DISJOINT" and row["seed"] == seed
                    and row["split"] == "confirmation"
                    and ("seen_pair/unseen_higher_order" if row["support_group"] ==
                         "seen_pair/unseen_triple" else row["support_group"]) == stratum
                }
                if (score < event_operator_noncollapse_min
                        or len(predictions) < event_operator_distinct_classes_min):
                    noncollapsed = False
        event_operator_noncollapsed[factor] = noncollapsed
    participant_time_strong_all_strata = all(
        decisions[f]["strong_train"] and
        all(decisions[f]["confirmation"][s]["all_disjoint_seeds_at_least_strong_min"]
            for s in STRATA)
        for f in ("participant", "time")
    )
    return {
        "schema": "norishio.issue48.role-identifiability-metrics.v1",
        "metrics": metrics,
        "decision": {
            "branch": branch,
            "by_factor": decisions,
            "strong_train_min": strong_min,
            "above_aliased_ceiling_margin": ceiling_margin,
            "generalization_gap_min": generalization_gap,
            "event_operator_noncollapse_balanced_min": event_operator_noncollapse_min,
            "event_operator_distinct_predicted_classes_min": event_operator_distinct_classes_min,
            "event_operator_noncollapsed": event_operator_noncollapsed,
            "rescue_factors": rescued,
            "authored_readout_condition": participant_time_strong_all_strata and
                all(event_operator_noncollapsed.values()),
            "all_four_strong_train_and_confirmation": all(
                decisions[f]["strong_train"] and
                all(decisions[f]["confirmation"][s]["all_disjoint_seeds_at_least_strong_min"]
                    for s in STRATA)
                for f in FACTORS
            ),
            "score_boundary": "authored structural-factor heads; not modern semantics or generation",
        },
    }


__all__ = ["ARMS", "FACTORS", "SEEDS", "STRATA", "summarize_role_heads"]
