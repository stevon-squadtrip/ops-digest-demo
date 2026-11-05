"""Ranking tests. Every expected value comes from tests/fixtures/expected.json, which is rebuilt from the synthetic
data. The tests are the spec for behaviour: which planted case is caught, where, and why."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src import rank
from src.clean import load_and_clean

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel():
    return load_and_clean()


@pytest.fixture(scope="module")
def result(panel):
    return rank.run(panel, write=False)


def row(scores, center, metric):
    hit = scores[(scores.center == center) & (scores.metric == metric)]
    assert len(hit) == 1, f"{center} {metric} missing from scores"
    return hit.iloc[0]


def series(panel, center, metric):
    lid = panel.locations.set_index("location_name").location_id[center]
    g = panel.my_clinic[panel.my_clinic.location_id == lid].sort_values("week_start")
    return g.set_index("week_start")[metric]


def item(result, center):
    return [it for it in result["items"] if it["center"] == center][0]


# Item order


def test_item_one_is_clinic_11_record_completion(result):
    first = result["items"][0]
    assert first["center"] == "Clinic 11"
    assert first["lead"]["metric"] == "record_completion_24h_pct"


def test_top_five_matches_findings(result):
    got = [(it["center"], it["lead"]["metric"]) for it in result["items"]]
    e = EXPECTED["top_five"]
    pair = lambda x: (x["center"], x["metric"])
    assert len(got) == rank.CAP
    assert len({c for c, _ in got}) == rank.CAP, "one item per center"
    assert got[0] == pair(e["first"])
    assert set(got[1:3]) == {pair(x) for x in e["tie_2_3"]}, "items 2 and 3 may come in either order"
    assert got[3] == pair(e["fourth"])
    assert got[4] == pair(e["fifth"])


def test_tie_break_spike_first(result):
    scores = result["scores"]
    a, b = EXPECTED["top_five"]["tie_2_3"]
    ra, rb = row(scores, a["center"], a["metric"]), row(scores, b["center"], b["metric"])
    if abs(ra.score - rb.score) < 1e-6:
        first = result["items"][1]
        winner = ra if (first["center"], first["lead"]["metric"]) == (a["center"], a["metric"]) else rb
        loser = rb if winner is ra else ra
        assert winner.spike_sd > (loser.spike_sd if pd.notna(loser.spike_sd) else -99)


# Planted case: a plan marked on track while its metric gets worse (Clinic 11)


def test_clinic_11_record_completion_values(panel):
    s = series(panel, "Clinic 11", "record_completion_24h_pct")
    for week, value in EXPECTED["clinic_11"]["record_completion"].items():
        assert s[pd.Timestamp(week)] == pytest.approx(value, abs=0.05)


def test_clinic_11_is_a_slow_drift(result):
    r = row(result["scores"], "Clinic 11", "record_completion_24h_pct")
    assert abs(r.spike_sd) < 1.0, "last week alone would not have raised it"
    assert r.drift_sd > 4.0
    assert r.run_weeks == rank.PERSIST_MAX_WEEKS
    assert "AP_001 contradicted" in r.plan_context


def test_clinic_11_throughput_rise_is_context_inside_item_one(result):
    it = result["items"][0]
    thr = [f for f in it["favorable_context"] if f["metric"] == "appts_per_doctor_hour"]
    assert len(thr) == 1
    assert thr[0]["latest"] == pytest.approx(EXPECTED["clinic_11"]["throughput_latest"], abs=0.005)
    ap009 = [p for p in it["plans"] if p["plan_id"] == "AP_009"][0]
    assert ap009["status"] == EXPECTED["clinic_11"]["ap009"]["status"]
    assert ap009["target_value"] == EXPECTED["clinic_11"]["ap009"]["target_value"]
    assert thr[0]["latest"] > ap009["target_value"]
    assert result["opportunity"]["center"] != "Clinic 11", "a ranked center never supplies the opportunity"


def test_clinic_11_all_doctors_slid(panel):
    pw = panel.my_providers
    lid = panel.locations.set_index("location_name").location_id["Clinic 11"]
    g = pw[pw.location_id == lid]
    weeks = sorted(g.week_start.unique())
    first8, last8 = weeks[:8], weeks[-8:]

    def hw(x):
        return (x.record_completion_24h_pct * x.scheduled_hours).sum() / x.scheduled_hours.sum()

    early = g[g.week_start.isin(first8)].groupby("provider_id").apply(hw, include_groups=False)
    late = g[g.week_start.isin(last8)].groupby("provider_id").apply(hw, include_groups=False)
    assert len(late) == EXPECTED["clinic_11"]["doctors_slid"]
    assert (late < early.reindex(late.index) - 5).all(), "every doctor fell by more than five points"


# Planted case: a staffing spike with doctor hours falling (Clinic 06)


def test_clinic_06_call_outs(panel, result):
    s = series(panel, "Clinic 06", "staff_call_outs")
    assert s.iloc[-4:].tolist() == EXPECTED["clinic_06"]["call_outs_last_4"]
    before_march = s[s.index < pd.Timestamp("2026-03-01")]
    assert before_march.median() <= EXPECTED["clinic_06"]["call_outs_before_march_median_max"]
    it = item(result, "Clinic 06")
    assert it["lead"]["metric"] == "staff_call_outs"
    assert it["lane"] == "ops"


def test_clinic_06_doctor_hours_context(result):
    it = item(result, "Clinic 06")
    hours = [c for c in it["unscored_context"] if c["metric"] == "doctor_hours_scheduled"]
    assert len(hours) == 1
    assert hours[0]["direction"] == "down"
    assert hours[0]["start"] == pytest.approx(EXPECTED["clinic_06"]["doctor_hours_12_weeks_ago"], abs=1.0)
    assert hours[0]["latest"] == pytest.approx(EXPECTED["clinic_06"]["doctor_hours_latest"], abs=1.0)


def test_clinic_06_recheck_reported_as_is(panel, result):
    s = series(panel, "Clinic 06", "recheck_compliance_pct")
    assert [round(v, 1) for v in s.iloc[-4:]] == EXPECTED["clinic_06"]["recheck_last_4"]
    r = row(result["scores"], "Clinic 06", "recheck_compliance_pct")
    assert (r.score < rank.SUPPORT_THRESHOLD) == EXPECTED["clinic_06"]["recheck_below_supporting_line"]


# Planted case: one doctor drives a clinic rate (Clinic 05, DVM_0024)


def test_clinic_05_dvm_0024(result):
    e = EXPECTED["clinic_05"]
    o = result["outliers"]
    hit = o[(o.provider_id == e["provider_id"]) & (o.metric == "recheck_compliance_pct")].iloc[0]
    assert hit.outlier
    assert hit.employment_type == e["employment_type"]
    assert hit.tenure_months == e["tenure_months"]
    assert hit.rate_simple_mean == pytest.approx(e["recheck_8w_simple_mean"], abs=0.05)
    assert hit.rate_hours_weighted == pytest.approx(e["recheck_8w_hours_weighted"], abs=0.05)
    assert hit.peers_min == pytest.approx(e["peers_hours_weighted_min"], abs=0.05)
    assert hit.peers_max == pytest.approx(e["peers_hours_weighted_max"], abs=0.05)
    rec = o[(o.provider_id == e["provider_id"]) & (o.metric == "record_completion_24h_pct")].iloc[0]
    assert rec.rate_simple_mean == pytest.approx(e["record_8w_simple_mean"], abs=0.05)
    assert rec.rate_hours_weighted == pytest.approx(e["record_8w_hours_weighted"], abs=0.05)
    assert not rec.outlier, "record completion is fine, only recheck is the coaching point"


def test_clinic_05_item_carries_provider_context(result):
    e = EXPECTED["clinic_05"]
    r = row(result["scores"], "Clinic 05", "recheck_compliance_pct")
    assert r.latest == pytest.approx(e["recheck_latest"], abs=0.05)
    assert r.latest < e["ap002"]["baseline_value"], "below the recorded baseline, which is what the owner sees"
    assert e["provider_id"] in r.provider_context
    assert r.context >= 2.0
    it = item(result, "Clinic 05")
    assert it["rank"] in (2, 3)
    assert it["lane"] == "medical"
    assert any(p["provider_id"] == e["provider_id"] for p in it["providers"])
    ap002 = [p for p in it["plans"] if p["plan_id"] == "AP_002"][0]
    assert ap002["baseline_panel"] != ap002["baseline_recorded"]
    assert "baseline mismatch" in ap002["flags"]


# Planted case: a perfect score that points at a logging gap (Clinic 09)


def test_clinic_09_integrity_not_praise(panel, result):
    e = EXPECTED["clinic_09"]
    s = series(panel, "Clinic 09", "callback_compliance_pct")
    assert (s.iloc[-10:] >= rank.INTEGRITY_LEVEL).sum() >= e["callback_weeks_at_100_of_last_10_min"]
    r = row(result["scores"], "Clinic 09", "callback_compliance_pct")
    assert r.integrity
    assert r["size"] == rank.SIZE_CAP
    recheck = series(panel, "Clinic 09", "recheck_compliance_pct").iloc[-8:].mean()
    record = series(panel, "Clinic 09", "record_completion_24h_pct").iloc[-8:].mean()
    assert recheck == pytest.approx(e["recheck_8w_approx"], abs=2.5)
    assert record == pytest.approx(e["record_8w_approx"], abs=2.5)
    opp = result["opportunity"]
    assert not (opp["center"] == "Clinic 09" and opp["metric"] == "callback_compliance_pct")


# Planted case: too few responses to trust (Clinic 07), the definition change, the ramp rule


def test_clinic_07_csat_is_a_watch_item(result):
    e = EXPECTED["suppressed"]["clinic_07_csat"]
    r = row(result["scores"], e["center"], e["metric"])
    assert not r.eligible
    assert r.suppressed.startswith(e["reason_prefix"])
    assert r.latest == pytest.approx(e["latest"], abs=0.005)
    assert r.denominator == e["responses"]


def test_membership_suppressed_everywhere(result):
    e = EXPECTED["suppressed"]["membership_all_centers"]
    rows = result["scores"][result["scores"].metric == e["metric"]]
    assert len(rows) == 11
    assert (~rows.eligible).all()
    assert rows.suppressed.str.startswith(e["reason_prefix"]).all()


def test_ramp_suppression_only_volume_and_commercial(result):
    e = EXPECTED["suppressed"]["ramp"]
    scores = result["scores"]
    for center in e["centers"]:
        for metric in e["metrics"]:
            r = row(scores, center, metric)
            assert r.suppressed.startswith(e["reason_prefix"]), f"{center} {metric}"
        for metric in rank.QUALITY_METRICS | {"staff_call_outs", "open_dvm_requisitions", "avg_wait_time_min"}:
            r = row(scores, center, metric)
            assert not r.suppressed.startswith("ramp"), f"{center} {metric} must never be age suppressed"
    ramp_rows = scores[scores.suppressed.str.startswith("ramp")]
    assert set(ramp_rows.metric) <= rank.RAMP_METRICS
    assert set(ramp_rows.center) == set(e["centers"])


def test_required_items_are_not_suppressed(result):
    for e in EXPECTED["not_suppressed"]:
        r = row(result["scores"], e["center"], e["metric"])
        assert r.eligible, f"{e['center']} {e['metric']} is suppressed: {r.suppressed}"
        assert r.score >= rank.ITEM_THRESHOLD


def test_every_considered_pair_is_shown_or_logged(panel, result, tmp_path):
    scores = result["scores"]
    assert len(scores) == 11 * len(rank.METRICS)
    paths = rank.write_outputs(panel, scores, result, result["why_not"], result["plans"], result["outliers"], out_dir=tmp_path)
    logged = json.loads(paths["suppressed"].read_text(encoding="utf-8"))["suppressed"]
    logged_pairs = {(s["center"], s["metric"]) for s in logged}
    shown = {(it["center"], it["lead"]["metric"]) for it in result["items"]}
    for it in result["items"]:
        shown |= {(it["center"], s["metric"]) for s in it["supporting"] + it["favorable_context"]}
    if result["opportunity"]:
        shown.add((result["opportunity"]["center"], result["opportunity"]["metric"]))
    shown |= {(m["center"], m["metric"]) for m in result["other_movements"]}
    everything = {(r.center, r.metric) for r in scores.itertuples()}
    assert shown | logged_pairs == everything
    assert not (shown & logged_pairs)
    assert all(s["reason"] for s in logged)


# Plan labels


def test_plan_labels_match_findings(result):
    plans = result["plans"].set_index("plan_id")
    for plan_id, e in EXPECTED["plan_labels"].items():
        got = plans.loc[plan_id]
        assert got["label"] == e["label"], f"{plan_id}: expected {e['label']}, got {got['label']} ({got['reason']})"
        assert list(got["flags"]) == e["flags"], f"{plan_id} flags {got['flags']}"
    for plan_id, not_flags in EXPECTED["plan_labels_not_flags"].items():
        for flag in not_flags:
            assert flag not in plans.loc[plan_id, "flags"], f"{plan_id} must not carry flag {flag}"
    assert plans.loc["AP_003", "status"] == "at_risk"
    assert plans.loc["AP_024", "days_since_update"] < 28


def test_ap001_is_the_plan_marked_on_track_while_the_metric_worsens(result):
    plans = result["plans"].set_index("plan_id")
    r = plans.loc["AP_001"]
    assert r["status"] == "on_track" and r["label"] == "contradicted"
    assert r["actual"] < r["baseline_panel"] - r["noise_band"]
    assert "Update the status to match the data" in r["action"]


def test_ap009_linked_metric_flag(result):
    # A plan cannot be blamed for an unlinked metric; a paired one that slid after it opened carries the flag.
    from src.plans import LINKED_FLAG, LABEL_ORDER
    plans = result["plans"].set_index("plan_id")
    r = plans.loc["AP_009"]
    assert r["label"] == "mistargeted"
    assert LINKED_FLAG in r["flags"] and "baseline mismatch" in r["flags"]
    assert any("Record completion" in s for s in r["linked_metrics"])
    assert "Check the linked metric" in r["action"]
    assert "complete with side effects" not in LABEL_ORDER


def test_linked_pair_list(result):
    from src.plans import LINKED_FLAG, LINKED_METRICS, LINKED_MIN_RUN_WEEKS
    assert frozenset({"appts_per_doctor_hour", "record_completion_24h_pct"}) in LINKED_METRICS
    assert frozenset({"client_csat", "avg_wait_time_min"}) in LINKED_METRICS
    assert frozenset({"recheck_compliance_pct", "staff_call_outs"}) not in LINKED_METRICS
    plans = result["plans"].set_index("plan_id")
    assert LINKED_FLAG not in plans.loc["AP_024", "flags"]
    assert LINKED_FLAG in plans.loc["AP_009", "flags"]
    assert LINKED_FLAG in plans.loc["AP_007", "flags"], "client satisfaction is paired with throughput"
    for pid in ["AP_004", "AP_019"]:
        assert LINKED_FLAG not in plans.loc[pid, "flags"]
    scores = result["scores"]
    for pid, r in plans.iterrows():
        for line in r["linked_metrics"]:
            metric_label = line.split(" slid from")[0]
            hit = scores[(scores.location_id == r["location_id"]) & (scores.label == metric_label)].iloc[0]
            assert hit.run_weeks_uncapped >= LINKED_MIN_RUN_WEEKS
            assert hit.score >= rank.SUPPORT_THRESHOLD
            assert frozenset({r["target_metric"], hit.metric}) in LINKED_METRICS


def test_linked_minimum_run(result):
    from src.plans import LINKED_MIN_RUN_WEEKS
    assert LINKED_MIN_RUN_WEEKS == 3
    plans = result["plans"].set_index("plan_id")
    scores = result["scores"]
    short = scores[(scores.run_weeks_uncapped < LINKED_MIN_RUN_WEEKS) & (scores.score >= rank.SUPPORT_THRESHOLD)]
    assert len(short) > 0, "the test needs at least one short run that scored high"
    for s in short.itertuples():
        for pid, r in plans[plans.location_id == s.location_id].iterrows():
            assert not any(line.startswith(s.label) for line in r["linked_metrics"]), f"{pid} carries a {s.run_weeks_uncapped} week run"


def test_other_movements_filter(result):
    # Only movements at or above the supporting line, at most 4. A 0.01 move never shows.
    om = result["other_movements"]
    assert len(om) <= rank.OTHER_MOVEMENTS_CAP
    for m in om:
        assert max(m["score"], m["favorable_score"]) >= rank.SUPPORT_THRESHOLD
    scores = result["scores"]
    shown = {(m["center"], m["metric"]) for m in om}
    below = scores[scores.eligible & (scores[["score", "favorable_score"]].max(axis=1) < rank.SUPPORT_THRESHOLD)]
    assert not any((r.center, r.metric) in shown for r in below.itertuples())


def test_new_center_spike_only_rule(result):
    # At a center under RAMP_WEEKS, an adverse signal with a run under 2 weeks is watched, not ranked.
    scores = result["scores"]
    new = scores[scores.weeks_available < rank.RAMP_WEEKS]
    watched = new[new.suppressed.str.startswith("watching")]
    for r in watched.itertuples():
        assert r.run_weeks_uncapped < rank.NEW_CENTER_MIN_RUN
    v = row(scores, "Clinic 07", "staff_call_outs")
    # Clinic 07 call outs ran 2 weeks (3 then 5), the minimum run, so the rule as written does not move it.
    assert v.run_weeks_uncapped >= rank.NEW_CENTER_MIN_RUN
    assert (v.run_weeks_uncapped < rank.NEW_CENTER_MIN_RUN) == v.suppressed.startswith("watching")


def test_ap004_label_is_not_robust(result):
    plans = result["plans"].set_index("plan_id")
    r = plans.loc["AP_004"]
    assert r["label"] == "mistargeted"
    assert r["robust"] == False
    assert r["actual"] > r["target_value"], "the action is close it whichever baseline you believe"
    assert r["panel_base_1wk"] >= r["target_value"] and r["panel_base_4wk"] < r["target_value"]


def test_robustness_columns(result):
    """Single week rules carry a trailing 4 week check in their own output."""
    plans = result["plans"].set_index("plan_id")
    for col in ["panel_base_1wk", "panel_base_4wk", "robust", "robust_note"]:
        assert col in plans.columns
    for pid in ["AP_007", "AP_009", "AP_024"]:
        assert plans.loc[pid, "robust"] is True or plans.loc[pid, "robust"] == True, plans.loc[pid, "robust_note"]
    assert plans.loc["AP_019", "robust"] is None or pd.isna(plans.loc["AP_019", "robust"])
    scores = result["scores"]
    for col in ["denominator_4wk", "watch_robust", "spike_4wk_sd", "ceiling_weeks_last4"]:
        assert col in scores.columns
    v = row(scores, "Clinic 07", "client_csat")
    assert "over the trailing 4 weeks" in v.suppressed
    assert v.denominator_4wk >= v.denominator


def test_plan_summary_counts(result):
    s = result["plan_summary"]
    e = EXPECTED["plan_summary"]
    assert s["plans"] == e["plans"]
    assert s["mistargeted"] == e["mistargeted"]
    assert s["baseline_mismatch_flags"] == e["baseline_mismatch_flags"]
    plans = result["all_plans"]
    assert s["baseline_mismatch_flags"] == int(plans["flags"].apply(lambda f: "baseline mismatch" in f).sum())
    assert sum(s["per_label"].values()) == e["plans"]
    assert s["per_flag"]["linked metric slipped"] >= 1
    clean = plans[plans.label.isin(["confirmed", "achieved", "early"]) & plans["flags"].apply(lambda f: not f)]
    assert s["plans_tracking_as_reported"] == len(clean) == e["plans_tracking_as_reported"]
    if result["paths"]:
        csv = pd.read_csv(result["paths"]["plan_check"], comment="#")
        assert len(csv) == e["plans"]


def test_every_plan_has_one_label_and_a_reason(result):
    plans = result["all_plans"]
    assert len(plans) == EXPECTED["plan_summary"]["plans"]
    from src.plans import LABEL_ORDER
    assert set(plans.label) <= set(LABEL_ORDER)
    assert plans.reason.str.len().gt(0).all()
    assert plans.action.str.len().gt(0).all()
    assert plans.noise_band.notna().all()


def test_move_inside_noise_band_never_fires_contradicted(panel, result):
    """A move against the prior 4 weeks that sits inside the band is a note, never a trigger."""
    from src.plans import plan_check
    e = EXPECTED["noise_band_regression"]
    plans = result["plans"].set_index("plan_id")
    r = plans.loc[e["plan_id"]]
    assert abs(r["actual"] - r["prior"]) == pytest.approx(e["move_vs_prior_points"], abs=0.02)
    assert abs(r["actual"] - r["prior"]) < r["noise_band"]
    assert r["label"] != "contradicted"
    assert "worse by" in r["note"] or "better by" in r["note"], "the prior-window move is a note, never a trigger"
    # Same plan with a target the opening week had not met, so mistargeted cannot fire first.
    panel2 = type(panel)(**{**panel.__dict__})
    panel2.plans = panel.plans.copy()
    panel2.plans.loc[panel2.plans.plan_id == e["plan_id"], "target_value"] = 95.0
    again = plan_check(panel2, scores=result["scores"], in_span_only=True).set_index("plan_id").loc[e["plan_id"]]
    assert again["label"] != "mistargeted"
    assert again["label"] != "contradicted", again["reason"]


def test_plans_reported_even_when_metric_is_ramp_suppressed(result):
    plans = result["plans"].set_index("plan_id")
    for plan_id in EXPECTED["plans_reported_despite_ramp"]:
        assert plan_id in plans.index
        assert plans.loc[plan_id, "ramp_suppressed_in_ranking"]
        assert plans.loc[plan_id, "label"]
    assert len(plans) == 10


# Other movements and the opportunity


def test_other_movements(result):
    e = EXPECTED["other_movements"]
    scores = result["scores"]
    others = {(m["center"], m["metric"]) for m in result["other_movements"]}
    c03 = row(scores, "Clinic 03", "appts_no_show")
    assert c03.latest == e["clinic_03_no_shows"]["latest"]
    assert round(c03.avg8) == e["clinic_03_no_shows"]["avg8"]
    assert ("Clinic 03", "appts_no_show") in others
    c08 = row(scores, "Clinic 08", "appts_no_show")
    assert c08.latest == e["clinic_08_no_shows"]["latest"]
    assert round(c08.avg8) == e["clinic_08_no_shows"]["avg8"]
    # A spike at a 0.7 weight metric needs a score at or above the supporting line to show.
    assert (("Clinic 08", "appts_no_show") in others) == (c08.score >= rank.SUPPORT_THRESHOLD)
    cb = row(scores, "Clinic 02", "callback_compliance_pct")
    assert cb.latest == pytest.approx(e["clinic_02_callbacks"]["latest"], abs=0.05)
    assert cb.direction == "favorable"
    wait = row(scores, "Clinic 08", "avg_wait_time_min")
    assert wait.direction == "favorable" and wait.latest < wait.avg8


def test_opportunity_rule(result):
    opp = result["opportunity"]
    assert opp is not None
    ranked = {it["center"] for it in result["items"]}
    assert opp["center"] not in ranked
    assert opp["direction"] == "favorable"
    assert opp["metric"] in rank.OPPORTUNITY_METRICS
    assert opp["rule"] == "held and needed"
    assert opp["held_weeks_of_6"] >= rank.OPP_HELD_WEEKS
    assert opp["needed_by"]
    assert opp["level_sd"] <= 0, "copy only from a center at or above the peer median"
    e = EXPECTED["opportunity_expected"]
    assert (opp["center"], opp["metric"]) == (e["center"], e["metric"])
    assert e["needed_by"] in {n["center"] for n in opp["needed_by"]}
    for c in result["opportunity_candidates"]:
        assert c["metric"] in rank.OPPORTUNITY_METRICS


def test_clinic_09_callbacks_is_never_an_opportunity_source(result):
    e = EXPECTED["opportunity_never_source"]
    opp = result["opportunity"]
    assert (opp["center"], opp["metric"]) != (e["center"], e["metric"])
    assert not any((c["center"], c["metric"]) == (e["center"], e["metric"]) for c in result["opportunity_candidates"])
    r = row(result["scores"], e["center"], e["metric"])
    assert r.integrity, "the integrity rule is what keeps it out, not the ranking cap"


def test_clinic_06_appointments_context(result):
    e = EXPECTED["clinic_06_appts_context"]
    it = item(result, "Clinic 06")
    hit = [c for c in it["unscored_context"] if c["metric"] == e["metric"]]
    assert len(hit) == 1
    assert hit[0]["direction"] == e["direction"]
    assert abs(hit[0]["raw_change_pct"]) >= rank.CONTEXT_RAW_MOVE_PCT or abs(hit[0]["fitted_change_pct"]) >= rank.CONTEXT_MOVE_PCT


def test_why_not_covers_all_eleven(result):
    why = result["why_not"]
    assert len(why) == 11
    assert sum(w["in_top_five"] for w in why) == rank.CAP
    assert all(w["why"].endswith(".") for w in why)


def test_lane_tag_never_changes_score(panel):
    from src.metrics import METRICS
    original = {m: s["lane"] for m, s in METRICS.items()}
    for m in METRICS:
        METRICS[m]["lane"] = "ops"
    try:
        flipped = rank.run(panel, write=False)["scores"]
    finally:
        for m, lane in original.items():
            METRICS[m]["lane"] = lane
    base = rank.run(panel, write=False)["scores"]
    pd.testing.assert_series_equal(flipped.score, base.score)
