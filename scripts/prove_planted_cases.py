"""Rebuild tests/fixtures/expected.json from the synthetic data and print the planted case proof.

Run from the repo root: python scripts/prove_planted_cases.py
Pass --no-fixture to print the proof without rewriting the fixture.

The fixture is the behaviour spec the tests read. Every value in it comes from the pipeline run on data/raw, so
after any change to scripts/make_synthetic_data.py the order is: regenerate the data, run this script, run pytest.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rank  # noqa: E402
from src import facts as facts_mod  # noqa: E402
from src.clean import ROOT, load_and_clean  # noqa: E402
from src.plans import at_open_sensitivity  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "expected.json"
ITEM1_WEEKS = ["2026-02-09", "2026-02-23", "2026-03-16", "2026-04-06", "2026-04-27"]
DISPLAY_KEYS = ["c06_hours_change_pct", "c06_appts_change_pct", "c06_open_reqs_last", "c06_callouts_before_march_median",
                "c07_csat_4wk_weighted", "c11_throughput_last", "ap_001_target", "c06_callouts_last4", "ap_009_flags",
                "c11_duration_change", "item1_headline", "item2_headline", "item3_headline", "item4_headline", "item5_headline",
                "one_thing_center", "one_thing_item_rank"]


def f(x, n=2):
    return round(float(x), n)


def build(panel, res, F) -> dict:
    sc, pl, ap, s = res["scores"], res["plans"].set_index("plan_id"), res["all_plans"].set_index("plan_id"), res["plan_summary"]

    def ser(c, m):
        lid = panel.locations.set_index("location_name").location_id[c]
        return panel.my_clinic[panel.my_clinic.location_id == lid].sort_values("week_start").set_index("week_start")[m]

    def row(c, m):
        return sc[(sc.center == c) & (sc.metric == m)].iloc[0]

    o = res["outliers"]
    d = o[(o.provider_id == "DVM_0024") & (o.metric == "recheck_compliance_pct")].iloc[0]
    dr = o[(o.provider_id == "DVM_0024") & (o.metric == "record_completion_24h_pct")].iloc[0]
    rc, co = ser("Clinic 11", "record_completion_24h_pct"), ser("Clinic 06", "staff_call_outs")
    hrs, rk = ser("Clinic 06", "doctor_hours_scheduled"), ser("Clinic 06", "recheck_compliance_pct")
    v = row("Clinic 07", "client_csat")
    sens = at_open_sensitivity(panel)
    return {
        "_source": "Built from the synthetic data by scripts/prove_planted_cases.py. The tests are the spec for behaviour, not for exact values. Rebuild after any change to scripts/make_synthetic_data.py.",
        "digest_date": "2026-05-04", "latest_week": "2026-04-27",
        "top_five": {"_rule": "one item per center; items 2 and 3 may appear in either order",
                     "first": {"center": "Clinic 11", "metric": "record_completion_24h_pct"},
                     "tie_2_3": [{"center": "Clinic 06", "metric": "staff_call_outs"}, {"center": "Clinic 05", "metric": "recheck_compliance_pct"}],
                     "fourth": {"center": "Clinic 09", "metric": "callback_compliance_pct", "integrity": True},
                     "fifth": {"center": "Clinic 07", "metric": "staff_call_outs"}},
        "clinic_11": {"record_completion": {w: f(rc[pd.Timestamp(w)], 1) for w in ITEM1_WEEKS},
                      "throughput_latest": f(row("Clinic 11", "appts_per_doctor_hour").latest, 3),
                      "ap009": {"status": "complete", "target_value": 2.52}, "ap001": {"status": "on_track", "label": "contradicted"},
                      "doctors_slid": int(F["c11_doctors_fell"]["value"])},
        "clinic_06": {"call_outs_last_4": [int(x) for x in co.iloc[-4:]], "call_outs_before_march_median_max": 2,
                      "doctor_hours_12_weeks_ago": f(hrs.iloc[-12], 1), "doctor_hours_latest": f(hrs.iloc[-1], 1),
                      "recheck_last_4": [f(x, 1) for x in rk.iloc[-4:]],
                      "recheck_below_supporting_line": bool(row("Clinic 06", "recheck_compliance_pct").score < rank.SUPPORT_THRESHOLD)},
        "clinic_05": {"provider_id": "DVM_0024", "employment_type": "full_time", "tenure_months": 89,
                      "recheck_8w_simple_mean": f(d.rate_simple_mean, 1), "recheck_8w_hours_weighted": f(d.rate_hours_weighted, 1),
                      "record_8w_simple_mean": f(dr.rate_simple_mean, 1), "record_8w_hours_weighted": f(dr.rate_hours_weighted, 1),
                      "peers_hours_weighted_min": f(d.peers_min, 1), "peers_hours_weighted_max": f(d.peers_max, 1),
                      "recheck_latest": f(row("Clinic 05", "recheck_compliance_pct").latest, 1), "ap002": {"baseline_value": 74.0, "label": "stalled"}},
        "clinic_09": {"callback_weeks_at_100_of_last_10_min": 6, "recheck_8w_approx": f(ser("Clinic 09", "recheck_compliance_pct").iloc[-8:].mean(), 1),
                      "record_8w_approx": f(ser("Clinic 09", "record_completion_24h_pct").iloc[-8:].mean(), 1), "label": "integrity"},
        "suppressed": {"clinic_07_csat": {"center": "Clinic 07", "metric": "client_csat", "latest": f(v.latest), "responses": int(v.denominator), "reason_prefix": "watch"},
                       "membership_all_centers": {"metric": "membership_conversion_pct", "reason_prefix": "definition change"},
                       "ramp": {"centers": ["Clinic 07", "Clinic 10"], "metrics": ["appts_per_doctor_hour", "revenue_per_appt", "appts_no_show"], "reason_prefix": "ramp"}},
        "not_suppressed": [{"center": "Clinic 05", "metric": "recheck_compliance_pct"}, {"center": "Clinic 09", "metric": "callback_compliance_pct"},
                           {"center": "Clinic 06", "metric": "staff_call_outs"}],
        "plan_labels": {pid: {"label": pl.loc[pid, "label"], "flags": list(pl.loc[pid, "flags"])} for pid in pl.index},
        "plan_labels_not_flags": {"AP_024": ["stale", "linked metric slipped"], "AP_004": ["linked metric slipped"], "AP_019": ["linked metric slipped"]},
        "plan_summary": {k: s[k] for k in ["plans", "mistargeted", "met_at_open", "at_open_denominator", "no_data_at_open", "baseline_mismatch_flags",
                                           "baseline_mismatch_denominator", "plans_tracking_as_reported", "plans_tracking_as_reported_ids"]},
        "sensitivity": {r.window: [int(r.met), int(r.denominator)] for r in sens.itertuples()},
        "met_at_open_direction": {"AP_008": ap.loc["AP_008", "met_at_open"], "AP_003": ap.loc["AP_003", "met_at_open"]},
        "noise_band_regression": {"plan_id": "AP_024", "move_vs_prior_points": f(abs(pl.loc["AP_024", "actual"] - pl.loc["AP_024", "prior"])),
                                  "note": "a move inside the band never fires contradicted"},
        "plans_reported_despite_ramp": ["AP_027", "AP_019"],
        "other_movements": {"clinic_03_no_shows": {"center": "Clinic 03", "metric": "appts_no_show", "latest": int(row("Clinic 03", "appts_no_show").latest), "avg8": int(round(row("Clinic 03", "appts_no_show").avg8))},
                            "clinic_08_no_shows": {"center": "Clinic 08", "metric": "appts_no_show", "latest": int(row("Clinic 08", "appts_no_show").latest), "avg8": int(round(row("Clinic 08", "appts_no_show").avg8))},
                            "clinic_02_callbacks": {"center": "Clinic 02", "metric": "callback_compliance_pct", "latest": f(row("Clinic 02", "callback_compliance_pct").latest, 1)},
                            "clinic_08_wait_improved": {"center": "Clinic 08", "metric": "avg_wait_time_min"}},
        "opportunity_expected": {"center": "Clinic 01", "metric": "recheck_compliance_pct", "needed_by": "Clinic 05"},
        "opportunity_never_source": {"center": "Clinic 09", "metric": "callback_compliance_pct"},
        "clinic_06_appts_context": {"metric": "appts_completed", "direction": "down"},
        "display": {k: F[k]["display"] for k in DISPLAY_KEYS},
        "data_notes": {"locations": int(len(panel.locations)), "plans": int(len(panel.plans)), "duplicates": 9, "negative_waits": 11,
                       "tier_mismatch_centers": sorted(panel.tier_mismatches.location_name),
                       "partial_history": {"Clinic 10": 18, "Clinic 07": 22, "Clinic 05": 30, "Clinic 09": 36, "Clinic 11": 52},
                       "fte_oddity_centers": ["Clinic 09", "Clinic 11"], "north_centers": [f"Clinic {i:02d}" for i in range(1, 12)]},
    }


def proof(panel, res, F) -> list:
    sc, pl, s = res["scores"], res["plans"].set_index("plan_id"), res["plan_summary"]
    items = {it["center"]: it for it in res["items"]}

    def row(c, m):
        return sc[(sc.center == c) & (sc.metric == m)].iloc[0]

    o = res["outliers"]
    d = o[(o.provider_id == "DVM_0024") & (o.metric == "recheck_compliance_pct")].iloc[0]
    ap001, ap009, ap007 = pl.loc["AP_001"], pl.loc["AP_009"], pl.loc["AP_007"]
    cb = row("Clinic 09", "callback_compliance_pct")
    cs = row("Clinic 07", "client_csat")
    mem = sc[sc.metric == "membership_conversion_pct"]
    co = row("Clinic 06", "staff_call_outs")
    opp = res["opportunity"]
    notes = {n["note"]: n["count"] for n in panel.notes}
    return [
        f"1. Plan marked on track while its metric worsens: AP_001 at Clinic 11 is {ap001.label} (status {ap001.status}, actual {ap001.actual} vs panel baseline {ap001.baseline_panel}). Item {items['Clinic 11']['rank']}.",
        f"2. Related metric slips while the tracked one looks fine: AP_009 ({ap009.status}) carries '{', '.join(ap009['flags'])}'; {ap009.linked_metrics[0] if ap009.linked_metrics else 'no linked metric'}. AP_007 at Clinic 02: {ap007.linked_metrics[0] if ap007.linked_metrics else 'none'}.",
        f"3. Suspicious perfect score: Clinic 09 callbacks integrity={bool(cb.integrity)}, {int(cb.weeks_at_ceiling)} of the last 10 weeks at or above {rank.INTEGRITY_LEVEL}. Item {items['Clinic 09']['rank']}, '{F['item' + str(items['Clinic 09']['rank']) + '_headline']['display']}'.",
        f"4. Too few responses to trust: Clinic 07 client satisfaction {cs.latest} on {int(cs.denominator)} responses, eligible={bool(cs.eligible)}, reason '{cs.suppressed.split('.')[0]}'.",
        f"5. Definition change mid period: membership conversion suppressed at {int((~mem.eligible).sum())} of {len(mem)} clinics, reason '{mem.suppressed.iloc[0]}'.",
        f"6. One doctor drives a clinic rate: DVM_0024 at Clinic 05 outlier={bool(d.outlier)}, recheck {d.rate_hours_weighted} vs peers {d.peers_hours_weighted}; record completion is not an outlier. Item {items['Clinic 05']['rank']}.",
        f"7. Staffing spike with hours falling: Clinic 06 call outs spike {co.spike_sd:.1f} sd, run {int(co.run_weeks_uncapped)} weeks, doctor hours {F['c06_hours_change_pct']['display']}. Item {items['Clinic 06']['rank']} and the one thing today ({F['one_thing_center']['display']}).",
        f"8. Opportunity rule: {opp['center']} {opp['metric']} held {opp['held_weeks_of_6']} of 6 weeks, needed by {sorted({n['center'] for n in opp['needed_by']})}.",
        f"9. Data faults caught: {notes['duplicate_rows_removed']} duplicates, {notes['negative_wait_set_missing']} negative waits, {notes['maturity_tier_mismatch']} tier mismatches, {notes['state_code_mismatch']} state code mismatches, {notes['doctor_fte_target_oddity']} FTE oddities, {notes['employment_type_lowercased']} employment_type rows lowercased.",
        f"10. Plan headline: {s['met_at_open']} of the {s['at_open_denominator']} plans with data at open had already met their target, and {s['plans_tracking_as_reported']} of {s['plans']} track as reported without a flag.",
    ]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    panel = load_and_clean()
    res = rank.run(panel, write=False)
    F = facts_mod.build_facts(panel, res)["facts"]
    if "--no-fixture" not in argv:
        FIXTURE.write_text(json.dumps(build(panel, res, F), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {FIXTURE.relative_to(ROOT)}")
    print("\n".join(proof(panel, res, F)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
