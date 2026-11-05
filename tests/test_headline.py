"""Locks the two headline plan numbers to their pinned definitions (src/plans.py HEADLINE)."""
import json
from pathlib import Path

import pytest

from src import rank
from src.clean import load_and_clean
from src.plans import HEADLINE, at_open_sensitivity, plan_summary

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected.json").read_text(encoding="utf-8"))
S = EXPECTED["plan_summary"]


@pytest.fixture(scope="module")
def panel():
    return load_and_clean()


@pytest.fixture(scope="module")
def result(panel):
    return rank.run(panel, write=False)


def test_chosen_window_is_last_week_on_or_before_open():
    assert HEADLINE["at_open_window"] == "last_week_on_or_before_open"
    assert HEADLINE["no_data_at_open_label"] == "no data at open"
    assert HEADLINE["tracking_labels"] == ["confirmed", "achieved", "early"]


def test_met_at_open_count_and_denominator(result):
    s = plan_summary(result["all_plans"])
    assert s["met_at_open"] == S["met_at_open"]
    assert s["at_open_denominator"] == S["at_open_denominator"]
    assert s["plans"] == S["plans"]
    assert s["mistargeted"] == S["mistargeted"], "mistargeted is the same count under the pinned window"


def test_no_data_at_open_plans(result):
    plans = result["all_plans"].set_index("plan_id")
    s = plan_summary(result["all_plans"])
    assert s["no_data_at_open"] == S["no_data_at_open"]
    for pid in S["no_data_at_open"]:
        assert plans.loc[pid, "met_at_open"] == "no data at open"
        assert "no panel row" in plans.loc[pid, "flags"]
        assert plans.loc[pid, "label"] != "mistargeted"


def test_tracking_as_reported_without_a_flag(result):
    s = plan_summary(result["all_plans"])
    assert s["plans_tracking_as_reported"] == S["plans_tracking_as_reported"]
    assert s["plans_tracking_as_reported_ids"] == S["plans_tracking_as_reported_ids"]
    plans = result["all_plans"]
    clean = plans[plans.label.isin(HEADLINE["tracking_labels"]) & plans["flags"].apply(lambda f: not f)]
    assert len(clean) == S["plans_tracking_as_reported"]


def test_direction_is_metric_aware(result):
    plans = result["all_plans"].set_index("plan_id")
    # Wait time: lower is better. AP_008 opened below its target (met); AP_003 opened above it (not met).
    for pid, want in EXPECTED["met_at_open_direction"].items():
        assert plans.loc[pid, "met_at_open"] == want, pid


def test_sensitivity_table(panel):
    t = at_open_sensitivity(panel).set_index("window")
    for window, (met, denom) in EXPECTED["sensitivity"].items():
        assert t.loc[window, ["met", "denominator"]].tolist() == [met, denom], window
    assert t.loc["last_week_on_or_before_open", "no_data"] == S["no_data_at_open"]


def test_baseline_mismatch_denominator(result):
    # One baseline rule, the noise band, counted over the plans with a panel row.
    s = plan_summary(result["all_plans"])
    assert s["baseline_mismatch_flags"] == S["baseline_mismatch_flags"]
    assert s["baseline_mismatch_denominator"] == S["baseline_mismatch_denominator"]


def test_feedback_wiring(panel, tmp_path):
    # not_useful raises the line 25 percent for 4 weeks; already_handled hides the item for 2 weeks.
    from src import feedback
    base = rank.run(panel, write=False)
    csv = tmp_path / "feedback.csv"
    csv.write_text("digest_date,item_id,verdict,note\n2026-04-27,item1,not_useful,\n2026-04-27,item5,already_handled,\n", encoding="utf-8")
    item_map = {f"item{it['rank']}": (it["location_id"], it["lead"]["metric"]) for it in base["items"]}
    th = feedback.thresholds(panel.today, item_map, rank.ITEM_THRESHOLD, path=csv, log_path=tmp_path / "log.json")
    first = item_map["item1"]
    assert th[first]["threshold"] == pytest.approx(3.75)
    assert th[item_map["item5"]]["suppressed_until"] == "2026-05-11"
    out = rank.run(panel, thresholds=th, write=False)
    leads = {it["center"]: it["lead"]["metric"] for it in out["items"]}
    assert "Clinic 11" in leads, "15.0 still clears a 3.75 line"
    # already_handled hides that exact center and metric pair, not the center: Clinic 07 may stay on another metric.
    assert leads.get("Clinic 07") != "staff_call_outs"
    v = out["scores"][(out["scores"].center == "Clinic 07") & (out["scores"].metric == "staff_call_outs")].iloc[0]
    assert v.suppressed.startswith("feedback: already handled") and not v.eligible
    mo = out["scores"][(out["scores"].center == "Clinic 11") & (out["scores"].metric == "record_completion_24h_pct")].iloc[0]
    assert mo.threshold == pytest.approx(3.75)
    log = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    assert len(log["decisions"]) == 2 and all(d["applied"] for d in log["decisions"])
