"""Milestone 5e evidence. The feedback file changes the line per center and metric, and nothing else.

(a) an empty feedback.csv leaves the five items, their order and the accepted digest unchanged
(b) not useful on three prior Mondays raises the line and moves the pair to the hidden list, reason naming the dates
(c) already handled hides the pair for the next two digests and it comes back on the third
"""
import json

import pandas as pd
import pytest

from src import facts as facts_mod, feedback, generate, rank, validate, write
from src.clean import ROOT, load_and_clean

OUT = ROOT / "outputs"
CLINIC_07 = ("CL_007", "staff_call_outs")


@pytest.fixture(scope="module")
def panel():
    return load_and_clean()


@pytest.fixture(scope="module")
def item_map():
    published = json.loads((OUT / "items.json").read_text(encoding="utf-8"))
    return {f"item{it['rank']}": (it["location_id"], it["lead"]["metric"]) for it in published["items"]}


def test_a_empty_feedback_leaves_items_and_accepted_digest_unchanged(panel, tmp_path):
    empty = tmp_path / "feedback.csv"
    empty.write_text(",".join(feedback.COLUMNS) + "\n", encoding="utf-8")
    out = rank.run(panel, write=False, feedback_path=empty)
    published = json.loads((OUT / "items.json").read_text(encoding="utf-8"))["items"]
    got = [(it["rank"], it["center"], it["lead"]["metric"]) for it in out["items"]]
    want = [(it["rank"], it["center"], it["lead"]["metric"]) for it in published]
    assert got == want and len(got) == 5
    assert (out["scores"].threshold == rank.ITEM_THRESHOLD).all()
    assert not out["scores"].suppressed.str.startswith("feedback").any()
    fx = facts_mod.build_facts(panel, out)
    slots = json.loads((OUT / "digest_accepted_slots.json").read_text(encoding="utf-8"))["slots"]
    assert validate.validate(write.render(fx, slots), fx, slots) == []


def test_b_not_useful_three_times_raises_the_line_and_hides_the_pair(panel, item_map, tmp_path):
    assert item_map["item5"] == CLINIC_07
    csv = tmp_path / "feedback.csv"
    csv.write_text("digest_date,item_id,verdict,note\n2026-04-13,item5,not_useful,\n2026-04-20,item5,not_useful,\n2026-04-27,item5,not_useful,\n", encoding="utf-8")
    th = feedback.thresholds(panel.today, item_map, rank.ITEM_THRESHOLD, path=csv, log_path=tmp_path / "log.json")
    assert th[CLINIC_07]["threshold"] == pytest.approx(rank.ITEM_THRESHOLD * 1.25 ** 3, abs=1e-3)
    out = rank.run(panel, thresholds=th, write=False)
    row = out["scores"][(out["scores"].location_id == CLINIC_07[0]) & (out["scores"].metric == CLINIC_07[1])].iloc[0]
    assert rank.ITEM_THRESHOLD <= row.score < row.threshold, "the pair clears the normal line and fails the raised one"
    assert not row.eligible and row.suppressed.startswith("feedback: line raised")
    assert "2026-04-20" in row.suppressed and "2026-04-27" in row.suppressed
    assert CLINIC_07 not in {(it["location_id"], it["lead"]["metric"]) for it in out["items"]}
    assert CLINIC_07 not in {(m["location_id"], m["metric"]) for m in out["other_movements"]}
    # The hidden list on the page is suppressed.json; write it to a scratch folder and read it back.
    paths = rank.write_outputs(panel, out["scores"], out, out["why_not"], out["plans"], out["outliers"], out_dir=tmp_path / "out")
    hidden = json.loads(paths["suppressed"].read_text(encoding="utf-8"))["suppressed"]
    match = [s for s in hidden if s["center"] == "Clinic 07" and s["metric"] == CLINIC_07[1]]
    assert len(match) == 1 and "2026-04-20" in match[0]["reason"] and "2026-04-27" in match[0]["reason"]


def test_c_already_handled_hides_two_digests_then_returns(panel, item_map, tmp_path):
    csv = tmp_path / "feedback.csv"
    csv.write_text("digest_date,item_id,verdict,note\n2026-05-04,item5,already_handled,\n", encoding="utf-8")
    log = tmp_path / "log.json"
    base = rank.ITEM_THRESHOLD
    # Week one and week two after the verdict: hidden. Week three: back.
    assert feedback.thresholds("2026-05-11", item_map, base, path=csv, log_path=log)[CLINIC_07]["suppressed_until"] == "2026-05-18"
    assert feedback.thresholds("2026-05-18", item_map, base, path=csv, log_path=log)[CLINIC_07]["suppressed_until"] == "2026-05-18"
    week3 = feedback.thresholds("2026-05-25", item_map, base, path=csv, log_path=log)[CLINIC_07]
    assert week3["suppressed_until"] is None and week3["threshold"] == base
    # The digest the verdict was given on is not changed by it.
    same_day = feedback.thresholds("2026-05-04", item_map, base, path=csv, log_path=log)[CLINIC_07]
    assert same_day["suppressed_until"] is None
    # The ranking honours the hide on both boundary dates: panel today is 2026-05-04, so use a verdict from 2026-04-20
    # (until 2026-05-04, hidden today) and one from 2026-04-13 (until 2026-04-27, back today).
    for given, hidden in [("2026-04-20", True), ("2026-04-13", False)]:
        csv.write_text(f"digest_date,item_id,verdict,note\n{given},item5,already_handled,\n", encoding="utf-8")
        th = feedback.thresholds(panel.today, item_map, base, path=csv, log_path=log)
        out = rank.run(panel, thresholds=th, write=False)
        row = out["scores"][(out["scores"].location_id == CLINIC_07[0]) & (out["scores"].metric == CLINIC_07[1])].iloc[0]
        assert row.suppressed.startswith("feedback: already handled") == hidden, given
        assert (CLINIC_07 in {(it["location_id"], it["lead"]["metric"]) for it in out["items"]}) == (not hidden), given
