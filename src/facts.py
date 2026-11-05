"""Build outputs/digest_facts.json: everything the digest can say, computed and formatted in code.

Every fact carries a value, a display string, a kind, and a source "file | column | scope | window".
The digest text never holds a number that is not a display value here. Prose (Milestone 4 part two)
fills named slots and may only reference the facts listed in that slot's allowed_facts.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import ROOT, Panel
from src.metrics import METRICS, MIN_CSAT_RESPONSES
from src import plans as plans_mod
from src import rank

OUT_DIR = ROOT / "outputs"
CW, PW, LOC, AP = "data/raw/clinic_weekly.csv", "data/raw/provider_weekly.csv", "data/raw/locations.csv", "data/raw/action_plans.csv"
README = "data/README.md"
RANKED, PLANS, ITEMS, SUMMARY, NOTES, SUPP = ("outputs/ranked.csv", "outputs/plan_check.csv", "outputs/items.json",
                                              "outputs/plan_check_summary.json", "outputs/data_notes.json",
                                              "outputs/suppressed.json")
FEEDBACK_FILE = "feedback/feedback.csv"
FEEDBACK_COLUMNS = ["digest_date", "item_id", "verdict", "note"]
FEEDBACK_VERDICTS = ["useful", "not_useful", "already_handled"]
DURATION_WINDOW_WEEKS = 8

# Selection rule for One thing today. Printed in the README.
ONE_THING_RULE = (
    "The highest ranked item in the ops lane whose lead signal is a spike of 3 sd or more with a run of "
    "at least 2 weeks. A staffing spike needs a same day call because this week's schedule is set on Monday. "
    "If no item qualifies, item 1."
)
# Slot word caps.
SLOT_CAPS = {"one_thing_today_why": 20, "what_happened": 40, "why_it_matters": 20, "what_to_do": 15,
             "plan_check": 20, "opportunity": 25, "suppressed": 15}
# 1,100 words is about 5 minutes at 220 words a minute; the top block (header, at a glance, one thing today)
# is under 1 minute. Tables are excluded from both counts.
BODY_CAP = 1100
TOP_CAP = 150
# "Also moving here" lines show only when the term that drove the score is outside the noise band (1 sd).
ALSO_MOVING_MIN_SD = 1.0
# Never offered to the prose model. Audit only.
AUDIT_ONLY = re.compile(r"(_progress_pct|_window_elapsed_pct|_weeks_past_due|_score|spike_sd|drift_sd|level_sd)$")

METRIC_KIND = {
    "recheck_compliance_pct": "pct", "record_completion_24h_pct": "pct", "callback_compliance_pct": "pct",
    "membership_conversion_pct": "pct", "client_csat": "csat", "appts_per_doctor_hour": "rate2",
    "avg_wait_time_min": "min1", "revenue_per_appt": "money2", "appts_no_show": "count",
    "staff_call_outs": "count", "open_dvm_requisitions": "count", "appts_completed": "count",
    "doctor_hours_scheduled": "hours1", "csat_responses": "count",
}
def check_back(when, weekday: bool = True) -> str:
    """'Monday May 11' or 'May 11' from a date, for the check back lines code writes."""
    ts = pd.Timestamp(when)
    return f"{ts:%A} {ts:%B} {ts.day}" if weekday else f"{ts:%B} {ts.day}"


METRIC_SHORT = {
    "recheck_compliance_pct": "recheck compliance", "record_completion_24h_pct": "record completion",
    "callback_compliance_pct": "callbacks", "client_csat": "client satisfaction", "appts_per_doctor_hour": "throughput",
    "avg_wait_time_min": "wait time", "revenue_per_appt": "revenue per appointment", "appts_no_show": "no shows",
    "staff_call_outs": "call outs", "open_dvm_requisitions": "open doctor requisitions",
    "membership_conversion_pct": "membership conversion",
}


def display(value, kind: str) -> str:
    """Code formats every number. Prose only ever sees these strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "none"
    if kind == "bool":
        return "yes" if value else "no"
    if kind == "list":
        return ", ".join(display(v, "auto") for v in value)
    if kind in ("text", "date"):
        return str(value)
    if kind == "auto":
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
        return str(value)
    v = float(value)
    if kind == "count":
        return str(int(round(v)))
    if kind in ("pct", "min1", "hours1", "points1"):
        return f"{v:.1f}"
    if kind in ("csat", "rate2", "money2", "sd2"):
        return f"{v:.2f}"
    if kind == "pct_change":
        n = int(round(abs(v)))
        return "flat" if n == 0 else f"{'down' if v < 0 else 'up'} {n} percent"
    if kind == "pct0":
        n = int(round(abs(v)))
        return f"minus {n}" if v < 0 and n else str(n)
    if kind == "minutes1":
        return f"{v:.1f} minutes"
    if kind == "min_change":
        return "unchanged" if abs(v) < 0.05 else f"{abs(v):.1f} minutes {'shorter' if v < 0 else 'longer'}"
    return str(value)


def shown_diff(a, b, kind: str = "pct") -> float:
    """a minus b, taken after each side is rounded the way its display shows it, so a displayed change always
    equals displayed after minus displayed before. Used for every fact that is a difference of two shown values."""
    d = 2 if kind in ("csat", "rate2", "money2", "sd2") else 1
    return round(round(float(a), d) - round(float(b), d), d)


class Facts:
    def __init__(self, leader: str, center_ids: dict = None):
        self.facts = {}
        self.leader = leader
        # Center name to location_id, from locations.csv. Every source string names a center as "Name (id)".
        self.center_ids = dict(sorted((center_ids or {}).items(), key=lambda kv: -len(kv[0])))

    def tag_centers(self, source: str) -> str:
        """Append the location_id after each center name in a source string, once, so a reader can find the rows
        in the CSVs, which store centers by id."""
        for name, lid in self.center_ids.items():
            pattern = re.compile(r"\b" + re.escape(name) + r"\b(?! \(" + re.escape(lid) + r"\))")
            source = pattern.sub(name + " (" + lid + ")", source)
        return source

    def add(self, fid: str, value, source: str, kind: str = "auto", unit: str = "", note: str = "") -> str:
        if isinstance(value, (np.floating,)):
            value = None if pd.isna(value) else float(value)
        elif isinstance(value, float) and pd.isna(value):
            value = None
        elif isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.bool_,)):
            value = bool(value)
        elif isinstance(value, pd.Timestamp):
            value = value.strftime("%Y-%m-%d")
            kind = "date"
        elif isinstance(value, list):
            value = [int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v for v in value]
            kind = "list"
        if isinstance(value, float):
            value = round(value, 3)
        if kind == "owner":
            disp = "You" if value == self.leader else str(value)
            kind = "text"
        else:
            disp = display(value, kind)
        entry = {"value": value, "display": disp, "kind": kind, "source": self.tag_centers(source)}
        if unit:
            entry["unit"] = unit
        if note:
            entry["note"] = note
        assert fid not in self.facts, f"duplicate fact id {fid}"
        self.facts[fid] = entry
        return fid

    def __getitem__(self, fid):
        return self.facts[fid]["value"]

    def d(self, fid):
        return self.facts[fid]["display"]


def _w(week) -> str:
    return pd.Timestamp(week).strftime("%Y-%m-%d")


def _hw(g: pd.DataFrame, col: str) -> float:
    h = g.scheduled_hours[g[col].notna()]
    return float((g[col] * g.scheduled_hours).sum() / h.sum()) if h.sum() else float("nan")


def allowed(fids) -> list:
    """Filter a candidate list down to facts prose may use. Keeps order, drops repeats and audit-only ids."""
    out = []
    for f in fids:
        if f not in out and not AUDIT_ONLY.search(f):
            out.append(f)
    return out


SPIKE_HEADLINE_SD = 3.0


def signal_type(row) -> str:
    """Which term leads the headline. Integrity first. A spike of 3 sd or more is a current event and
    beats a standing gap. Otherwise the largest of spike, drift and level."""
    if row.integrity:
        return "integrity"
    if pd.notna(row.spike_sd) and row.spike_sd >= SPIKE_HEADLINE_SD:
        return "spike"
    parts = {"spike": row.spike_sd if pd.notna(row.spike_sd) else -9, "drift": row.drift_sd if pd.notna(row.drift_sd) else -9,
             "level": row.level_sd if pd.notna(row.level_sd) else -9}
    return max(parts, key=parts.get)


def build_facts(panel: Panel, result: dict) -> dict:
    names = panel.locations.set_index("location_id").location_name
    ids = {v: k for k, v in names.items()}
    F = Facts(panel.leader, ids)
    cw = panel.my_clinic.sort_values(["location_id", "week_start"])
    scores = result["scores"]
    plans = result["plans"].set_index("plan_id")
    items = result["items"]
    latest = panel.latest_week
    pw = panel.my_providers

    def srow(center, metric):
        return scores[(scores.center == center) & (scores.metric == metric)].iloc[0]

    def series(center, metric):
        return cw[cw.location_id == ids[center]].set_index("week_start")[metric]

    def peer_median(metric, center):
        return scores[(scores.metric == metric) & (scores.center != center)].avg8.median()

    # Header, dates, counts
    F.add("leader", panel.leader, f"{LOC} | rmp_name | span", "text")
    F.add("rop_name", panel.my_locations.rop_name.iloc[0], f"{LOC} | rop_name | all centers in span", "text")
    F.add("digest_date", panel.today, "src/run.py | --date | today")
    F.add("latest_week", latest, f"{CW} | week_start | max")
    F.add("n_centers", len(panel.my_ids), f"{LOC} | rmp_name == leader | count", "count")
    F.add("n_other_centers", len(panel.my_ids) - 1, f"{LOC} | rmp_name == leader | count minus one, the peer set for any center", "count")
    F.add("next_digest_date", panel.today + pd.Timedelta(days=7), "src/run.py | --date plus 7 days")
    F.add("two_week_check_date", panel.today + pd.Timedelta(days=14), "src/run.py | --date plus 14 days")
    F.add("one_thing_check_back", check_back(panel.today + pd.Timedelta(days=7)), "src/run.py | --date plus 7 days | weekday month day", "text")
    F.add("n_items", len(items), f"{ITEMS} | items | count", "count")
    for k, (v, src, note) in {
        "window_8wk": (8, "src/rank.py | trailing 8 weeks in score_panel (avg8)", "trailing 8 week average"),
        "window_4wk": (4, "src/plans.py | ACTUAL_WEEKS", "trailing 4 weeks"),
        "window_10wk": (rank.PERSIST_MAX_WEEKS, "src/rank.py | PERSIST_MAX_WEEKS", "integrity and persistence window"),
        "window_6wk": (rank.OPP_WINDOW, "src/rank.py | OPP_WINDOW", "opportunity held window"),
        "window_12wk": (rank.CONTEXT_WINDOW, "src/rank.py | CONTEXT_WINDOW", "context trend window"),
        "line_item": (rank.ITEM_THRESHOLD, "src/rank.py | ITEM_THRESHOLD", "item line"),
        "line_support": (rank.SUPPORT_THRESHOLD, "src/rank.py | SUPPORT_THRESHOLD", "supporting line"),
        "integrity_level": (rank.INTEGRITY_LEVEL, "src/rank.py | INTEGRITY_LEVEL", "integrity ceiling"),
        "integrity_weeks": (rank.INTEGRITY_WEEKS, "src/rank.py | INTEGRITY_WEEKS", "integrity weeks of 10"),
        "min_csat_responses": (MIN_CSAT_RESPONSES, "src/metrics.py | MIN_CSAT_RESPONSES", "watch floor, CSAT responses a week"),
        "ramp_weeks": (rank.RAMP_WEEKS, "src/rank.py | RAMP_WEEKS", "ramp suppression, weeks of history"),
        "new_center_min_run": (rank.NEW_CENTER_MIN_RUN, "src/rank.py | NEW_CENTER_MIN_RUN", "spike only rule, minimum run at a new center"),
        "min_run_linked": (plans_mod.LINKED_MIN_RUN_WEEKS, "src/plans.py | LINKED_MIN_RUN_WEEKS", "linked metric minimum run"),
        "stale_days": (plans_mod.STALE_DAYS, "src/plans.py | STALE_DAYS", "stale flag, days"),
    }.items():
        F.add(k, v, src, "auto", note=note)
    F.add("metric_def_recheck", "recheck scheduled before the client left", f"{README} | recheck_compliance_pct | definition", "text")
    F.add("record_window_hours", 24, f"{README} | record_completion_24h_pct | records closed within this many hours", "count")

    # Clinic 06 staffing facts (one thing today and item 3)
    mp = "Clinic 06"
    co = series(mp, "staff_call_outs")
    hours = series(mp, "doctor_hours_scheduled")
    appts = series(mp, "appts_completed")
    before_march = co[co.index < pd.Timestamp("2026-03-01")]
    r_co = srow(mp, "staff_call_outs")
    F.add("c06_callouts_last", co.iloc[-1], f"{CW} | staff_call_outs | {mp} | {_w(co.index[-1])}", "count")
    F.add("c06_callouts_prior", co.iloc[-2], f"{CW} | staff_call_outs | {mp} | {_w(co.index[-2])}", "count")
    F.add("c06_callouts_last4", co.iloc[-4:].tolist(), f"{CW} | staff_call_outs | {mp} | {_w(co.index[-4])} to {_w(co.index[-1])}")
    n_at_latest = 0
    for v in co.iloc[::-1]:
        if v >= co.iloc[-1]:
            n_at_latest += 1
        else:
            break
    F.add("c06_callouts_weeks_at_latest", n_at_latest, f"{CW} | staff_call_outs | {mp} | consecutive weeks at or above last week's value", "count")
    F.add("c06_callouts_before_march_median", before_march.median(), f"{CW} | staff_call_outs | {mp} | weeks before 2026-03-01, median", "count")
    F.add("c06_callouts_before_march_p75", before_march.quantile(0.75), f"{CW} | staff_call_outs | {mp} | weeks before 2026-03-01, 75th percentile", "count")
    F.add("c06_callouts_long_mean", r_co.long_mean, f"{RANKED} | long_mean | {mp} staff_call_outs", "min1")
    F.add("c06_callouts_avg8", r_co.avg8, f"{RANKED} | avg8 | {mp} staff_call_outs", "min1")
    F.add("c06_callouts_spike_sd", r_co.spike_sd, f"{RANKED} | spike_sd | {mp} staff_call_outs", "sd2")
    F.add("c06_callouts_score", r_co.score, f"{RANKED} | score | {mp} staff_call_outs", "sd2")
    F.add("c06_weeks_7_or_more", int((co.iloc[-4:] >= 7).sum()), f"{CW} | staff_call_outs | {mp} | last 4 weeks at or above 7", "count")
    F.add("c06_hours_start", hours.iloc[-12], f"{CW} | doctor_hours_scheduled | {mp} | {_w(hours.index[-12])}", "hours1")
    F.add("c06_hours_last", hours.iloc[-1], f"{CW} | doctor_hours_scheduled | {mp} | {_w(hours.index[-1])}", "hours1")
    F.add("c06_hours_change_pct", (hours.iloc[-1] - hours.iloc[-12]) / hours.iloc[-12] * 100, f"{CW} | doctor_hours_scheduled | {mp} | 12 week first to last", "pct_change")
    F.add("c06_appts_start", appts.iloc[-12], f"{CW} | appts_completed | {mp} | {_w(appts.index[-12])}", "count")
    F.add("c06_appts_last", appts.iloc[-1], f"{CW} | appts_completed | {mp} | {_w(appts.index[-1])}", "count")
    F.add("c06_appts_change_pct", (appts.iloc[-1] - appts.iloc[-12]) / appts.iloc[-12] * 100, f"{CW} | appts_completed | {mp} | 12 week first to last", "pct_change")
    r_rq = srow(mp, "open_dvm_requisitions")
    F.add("c06_open_reqs_last", r_rq.latest, f"{CW} | open_dvm_requisitions | {mp} | {_w(latest)}", "count")
    F.add("c06_open_reqs_avg8", r_rq.avg8, f"{RANKED} | avg8 | {mp} open_dvm_requisitions", "min1")
    r_wt = srow(mp, "avg_wait_time_min")
    F.add("c06_wait_last", r_wt.latest, f"{CW} | avg_wait_time_min | {mp} | {_w(latest)}", "min1")
    F.add("c06_wait_avg8", r_wt.avg8, f"{RANKED} | avg8 | {mp} avg_wait_time_min", "min1")

    # Plans in span
    for pid, p in plans.iterrows():
        pre = pid.lower()
        kind = METRIC_KIND.get(p.target_metric, "auto")
        F.add(f"{pre}_center", p.location_name, f"{AP} | location_id | {pid}", "text")
        F.add(f"{pre}_metric", p.metric_label, f"{AP} | target_metric | {pid}", "text")
        F.add(f"{pre}_status", p.status.replace("_", " "), f"{AP} | status | {pid}", "text")
        F.add(f"{pre}_owner", p.owner_name, f"{AP} | owner_name | {pid}", "owner")
        F.add(f"{pre}_label", p.label, f"{PLANS} | label | {pid} | shown as label_display", "text")
        F.facts[f"{pre}_label"]["display"] = p.label_display
        F.add(f"{pre}_flags", list(p["flags"]) or ["none"], f"{PLANS} | flags | {pid}")
        F.add(f"{pre}_action", p.action, f"{PLANS} | action | {pid}", "text")
        F.add(f"{pre}_trend", p.trend_word, f"{PLANS} | trend_word | {pid} | trailing 4 weeks vs the 4 before", "text")
        F.add(f"{pre}_weeks_open", p.weeks_open, f"{PLANS} | weeks_open | {pid}", "count")
        F.add(f"{pre}_baseline_recorded", p.baseline_recorded, f"{AP} | baseline_value | {pid}", kind)
        F.add(f"{pre}_baseline_panel", p.baseline_panel, f"{PLANS} | panel_base_1wk | {pid} | week {p.baseline_week}", kind)
        F.add(f"{pre}_baseline_panel_4wk", p.panel_base_4wk, f"{PLANS} | panel_base_4wk | {pid}", kind)
        F.add(f"{pre}_target", p.target_value, f"{AP} | target_value | {pid}", kind)
        F.add(f"{pre}_actual", p.actual, f"{PLANS} | actual | {pid} | trailing 4 weeks to {_w(latest)}", kind)
        F.add(f"{pre}_last_week", p.last_week, f"{CW} | {p.target_metric} | {p.location_name} | {_w(latest)}", kind)
        F.add(f"{pre}_progress_pct", None if p.progress is None or pd.isna(p.progress) else float(p.progress) * 100, f"{PLANS} | progress | {pid}", "pct0",
              note="code built plan lines only, never a slot")
        F.add(f"{pre}_progress_abs_pct", None if p.progress is None or pd.isna(p.progress) else abs(float(p.progress)) * 100, f"{PLANS} | progress, absolute value | {pid}", "pct0",
              note="code built plan lines only, never a slot")
        F.add(f"{pre}_window_elapsed_pct", float(p.window_elapsed) * 100, f"{PLANS} | window_elapsed | {pid}", "pct0",
              note="code built plan lines only, never a slot")
        F.add(f"{pre}_last_update", str(p.last_status_update), f"{AP} | last_status_update | {pid}", "date")
        F.add(f"{pre}_days_since_update", int(p.days_since_update), f"{PLANS} | days_since_update | {pid}", "count")
        F.add(f"{pre}_due", str(p.due_date), f"{AP} | due_date | {pid}", "date")
        F.add(f"{pre}_weeks_past_due", None if pd.isna(p.weeks_past_due) else int(p.weeks_past_due), f"{PLANS} | weeks_past_due | {pid} | whole weeks from due_date to 2026-05-04", "count",
              note="code built plan lines only, never a slot")
        F.add(f"{pre}_opened", str(p.opened_date), f"{AP} | opened_date | {pid}", "date")
        F.add(f"{pre}_robust", p.robust if p.robust is not None else "no panel row", f"{PLANS} | robust | {pid}", "bool" if p.robust is not None else "text")
        F.add(f"{pre}_linked", list(p.linked_metrics) or ["none"], f"{PLANS} | linked_metrics | {pid}")
        F.add(f"{pre}_ramp_suppressed", bool(p.ramp_suppressed_in_ranking), f"{PLANS} | ramp_suppressed_in_ranking | {pid}", "bool")

    # Item 1 Clinic 11
    mo = "Clinic 11"
    rc = series(mo, "record_completion_24h_pct")
    r_rc = srow(mo, "record_completion_24h_pct")
    for wk in ["2026-02-09", "2026-02-23", "2026-03-16", "2026-04-06", "2026-04-27"]:
        F.add(f"c11_rc_{wk.replace('-', '')}", rc[pd.Timestamp(wk)], f"{CW} | record_completion_24h_pct | {mo} | {wk}", "pct")
    F.add("c11_rc_long_mean", r_rc.long_mean, f"{RANKED} | long_mean | {mo} record_completion_24h_pct", "pct")
    F.add("c11_rc_avg8", r_rc.avg8, f"{RANKED} | avg8 | {mo} record_completion_24h_pct", "pct")
    F.add("c11_rc_drift_sd", r_rc.drift_sd, f"{RANKED} | drift_sd | {mo} record_completion_24h_pct", "sd2")
    F.add("c11_rc_spike_sd", r_rc.spike_sd, f"{RANKED} | spike_sd | {mo} record_completion_24h_pct", "sd2")
    F.add("c11_rc_run_weeks", int(r_rc.run_weeks_uncapped), f"{RANKED} | run_weeks_uncapped | {mo} record_completion_24h_pct", "count")
    F.add("c11_rc_run_start", r_rc.run_start_week, f"{RANKED} | run_start_week | {mo} record_completion_24h_pct", "date")
    F.add("c11_rc_score", r_rc.score, f"{RANKED} | score | {mo} record_completion_24h_pct", "sd2")
    mo_appts = series(mo, "appts_completed")
    F.add("c11_appts_last", mo_appts.iloc[-1], f"{CW} | appts_completed | {mo} | {_w(latest)}", "count")
    F.add("c11_records_late_last_week", int(round(mo_appts.iloc[-1] * (100 - rc.iloc[-1]) / 100)),
          f"{CW} | appts_completed times (100 minus record_completion_24h_pct) | {mo} | {_w(latest)}", "count",
          note="records not closed within 24 hours last week, computed")
    r_th = srow(mo, "appts_per_doctor_hour")
    F.add("c11_throughput_last", r_th.latest, f"{CW} | appts_per_doctor_hour | {mo} | {_w(latest)}", "rate2")
    F.add("c11_throughput_avg8", r_th.avg8, f"{RANKED} | avg8 | {mo} appts_per_doctor_hour", "rate2")
    opened_009 = plans.loc["AP_009", "opened_date"]
    bweek_009 = plans_mod.monday_on_or_before(pd.Timestamp(opened_009))
    mo_pw = pw[pw.location_id == ids[mo]]
    before = mo_pw[(mo_pw.week_start < bweek_009) & (mo_pw.week_start >= bweek_009 - pd.Timedelta(weeks=DURATION_WINDOW_WEEKS))]
    after_weeks = sorted(mo_pw.week_start.unique())[-DURATION_WINDOW_WEEKS:]
    after = mo_pw[mo_pw.week_start.isin(after_weeks)]
    dur_before, dur_after = _hw(before, "avg_appt_duration_min"), _hw(after, "avg_appt_duration_min")
    F.add("c11_duration_before", dur_before, f"{PW} | avg_appt_duration_min hours weighted | {mo} | 8 weeks before AP_009 opened, {_w(before.week_start.min())} to {_w(before.week_start.max())}", "minutes1", unit="minutes")
    F.add("c11_duration_after", dur_after, f"{PW} | avg_appt_duration_min hours weighted | {mo} | last 8 weeks, {_w(after_weeks[0])} to {_w(latest)}", "minutes1", unit="minutes")
    F.add("c11_duration_change", shown_diff(dur_after, dur_before, "minutes1"), f"{PW} | avg_appt_duration_min hours weighted | {mo} | after minus before, each rounded to one decimal first", "min_change", unit="minutes")
    F.add("c11_visits_shorter", bool(dur_after - dur_before < -1.0), f"{PW} | computed | {mo} | change below minus 1 minute", "bool")
    weeks = sorted(mo_pw.week_start.unique())
    early = mo_pw[mo_pw.week_start.isin(weeks[:8])].groupby("provider_id").apply(lambda g: _hw(g, "record_completion_24h_pct"), include_groups=False)
    late = mo_pw[mo_pw.week_start.isin(weeks[-8:])].groupby("provider_id").apply(lambda g: _hw(g, "record_completion_24h_pct"), include_groups=False)
    F.add("c11_doctors", int(len(late)), f"{PW} | provider_id | {mo} | distinct in last 8 weeks", "count")
    doctor_rows = []
    for pid in late.index:
        F.add(f"c11_{pid.lower()}_early", early[pid], f"{PW} | record_completion_24h_pct hours weighted | {pid} | first 8 provider weeks {_w(weeks[0])} to {_w(weeks[7])}", "pct")
        F.add(f"c11_{pid.lower()}_late", late[pid], f"{PW} | record_completion_24h_pct hours weighted | {pid} | last 8 weeks {_w(weeks[-8])} to {_w(latest)}", "pct")
        doctor_rows.append({"provider_id": pid, "early": f"c11_{pid.lower()}_early", "late": f"c11_{pid.lower()}_late"})
    F.add("c11_doctors_fell", int((late < early.reindex(late.index) - 5).sum()), f"{PW} | record_completion_24h_pct | {mo} | doctors down more than 5 points, first 8 vs last 8 weeks", "count")

    # Item 2 Clinic 05
    bc = "Clinic 05"
    r_bc = srow(bc, "recheck_compliance_pct")
    out = result["outliers"]
    o24 = out[(out.provider_id == "DVM_0024") & (out.metric == "recheck_compliance_pct")].iloc[0]
    o24r = out[(out.provider_id == "DVM_0024") & (out.metric == "record_completion_24h_pct")].iloc[0]
    peers_med = peer_median("recheck_compliance_pct", bc)
    F.add("c05_recheck_last", r_bc.latest, f"{CW} | recheck_compliance_pct | {bc} | {_w(latest)}", "pct")
    F.add("c05_recheck_avg8", r_bc.avg8, f"{RANKED} | avg8 | {bc} recheck_compliance_pct", "pct")
    F.add("c05_recheck_level_sd", r_bc.level_sd, f"{RANKED} | level_sd | {bc} recheck_compliance_pct", "sd2")
    F.add("c05_recheck_score", r_bc.score, f"{RANKED} | score | {bc} recheck_compliance_pct", "sd2")
    F.add("recheck_peer_median_avg8", peers_med, f"{RANKED} | avg8 | recheck_compliance_pct | median of the other 10 centers", "pct")
    F.add("c05_gap_to_peer_median", shown_diff(peers_med, r_bc.avg8, "pct"), f"{RANKED} | avg8 | peer median minus {bc} recheck_compliance_pct, each rounded to one decimal first", "points1", unit="points")
    F.add("dvm0024_recheck_hw", o24.rate_hours_weighted, f"{PW} | recheck_compliance_pct hours weighted | DVM_0024 | last 8 weeks", "pct")
    F.add("dvm0024_recheck_mean", o24.rate_simple_mean, f"{PW} | recheck_compliance_pct simple mean | DVM_0024 | last 8 weeks", "pct")
    F.add("dvm0024_peers_hw", o24.peers_hours_weighted, f"{PW} | recheck_compliance_pct hours weighted | {bc} other doctors | last 8 weeks", "pct")
    F.add("dvm0024_gap", shown_diff(o24.peers_hours_weighted, o24.rate_hours_weighted, "pct"), f"{ITEMS} | provider_outliers peers_hours_weighted minus rate_hours_weighted | DVM_0024 recheck", "points1", unit="points")
    F.add("dvm0024_weeks", int(o24.weeks), f"{PW} | week_start | DVM_0024 | weeks scheduled in last 8", "count")
    F.add("dvm0024_hours", o24.hours, f"{PW} | scheduled_hours | DVM_0024 | last 8 weeks sum", "hours1")
    F.add("dvm0024_tenure", int(o24.tenure_months), f"{PW} | tenure_months | DVM_0024", "count")
    F.add("dvm0024_employment", o24.employment_type.replace("_", " "), f"{PW} | employment_type | DVM_0024", "text")
    F.add("dvm0024_record_hw", o24r.rate_hours_weighted, f"{PW} | record_completion_24h_pct hours weighted | DVM_0024 | last 8 weeks", "pct")
    F.add("dvm0024_record_mean", o24r.rate_simple_mean, f"{PW} | record_completion_24h_pct simple mean | DVM_0024 | last 8 weeks", "pct")
    F.add("dvm0024_record_peers_hw", o24r.peers_hours_weighted, f"{PW} | record_completion_24h_pct hours weighted | {bc} other doctors | last 8 weeks", "pct")
    peer_rows = []
    for o in out[(out.location_id == ids[bc]) & (out.metric == "recheck_compliance_pct")].sort_values("rate_hours_weighted").itertuples():
        pid = o.provider_id.lower()
        F.add(f"c05_{pid}_recheck_hw", o.rate_hours_weighted, f"{PW} | recheck_compliance_pct hours weighted | {o.provider_id} | last 8 weeks", "pct")
        F.add(f"c05_{pid}_recheck_mean", o.rate_simple_mean, f"{PW} | recheck_compliance_pct simple mean | {o.provider_id} | last 8 weeks", "pct")
        F.add(f"c05_{pid}_tenure", int(o.tenure_months), f"{PW} | tenure_months | {o.provider_id}", "count")
        F.add(f"c05_{pid}_employment", o.employment_type.replace("_", " "), f"{PW} | employment_type | {o.provider_id}", "text")
        F.add(f"c05_{pid}_hours", o.hours, f"{PW} | scheduled_hours | {o.provider_id} | last 8 weeks sum", "hours1")
        peer_rows.append({"provider_id": o.provider_id, "employment": f"c05_{pid}_employment", "tenure": f"c05_{pid}_tenure",
                          "hours": f"c05_{pid}_hours", "recheck_hw": f"c05_{pid}_recheck_hw", "recheck_mean": f"c05_{pid}_recheck_mean"})
    last8 = sorted(pw.week_start.unique())[-8:]
    bc8 = pw[(pw.location_id == ids[bc]) & pw.week_start.isin(last8)].copy()
    clinic_hw = _hw(bc8, "recheck_compliance_pct")
    cf = bc8.copy()
    cf.loc[cf.provider_id == "DVM_0024", "recheck_compliance_pct"] = o24.peers_hours_weighted
    clinic_cf = _hw(cf, "recheck_compliance_pct")
    F.add("c05_clinic_recheck_8wk_hw", clinic_hw, f"{PW} | recheck_compliance_pct hours weighted all doctors | {bc} | last 8 weeks", "pct")
    F.add("c05_clinic_recheck_if_dvm0024_at_peers", clinic_cf, f"{PW} | recheck_compliance_pct hours weighted with DVM_0024 set to peers' rate | {bc} | last 8 weeks", "pct", note="counterfactual")
    F.add("c05_counterfactual_lift", shown_diff(clinic_cf, clinic_hw, "pct"), f"{PW} | computed | {bc} | counterfactual minus actual, each rounded to one decimal first", "points1", unit="points")
    F.add("c05_gap_to_target_after", shown_diff(plans.loc["AP_002", "target_value"], clinic_cf, "pct"), f"{AP} target_value minus counterfactual, each rounded to one decimal first | AP_002", "points1", unit="points")
    hrs_bc = series(bc, "doctor_hours_scheduled")
    F.add("c05_hours_start", hrs_bc.iloc[-12], f"{CW} | doctor_hours_scheduled | {bc} | {_w(hrs_bc.index[-12])}", "hours1")
    F.add("c05_hours_last", hrs_bc.iloc[-1], f"{CW} | doctor_hours_scheduled | {bc} | {_w(latest)}", "hours1")

    # Item 4 Clinic 09
    di = "Clinic 09"
    r_cb = srow(di, "callback_compliance_pct")
    cb = series(di, "callback_compliance_pct")
    F.add("c09_cb_last", r_cb.latest, f"{CW} | callback_compliance_pct | {di} | {_w(latest)}", "pct")
    F.add("c09_cb_avg8", r_cb.avg8, f"{RANKED} | avg8 | {di} callback_compliance_pct", "pct")
    F.add("c09_cb_weeks_at_ceiling_last10", int(r_cb.weeks_at_ceiling), f"{RANKED} | weeks_at_ceiling | {di} callback_compliance_pct | last 10 weeks at or above 99.5", "count")
    F.add("c09_cb_weeks_at_ceiling_all", int((cb >= rank.INTEGRITY_LEVEL).sum()), f"{CW} | callback_compliance_pct | {di} | weeks at or above 99.5, all history", "count")
    F.add("c09_weeks_available", int(len(cb)), f"{CW} | week_start | {di} | count", "count")
    F.add("c09_cb_score", r_cb.score, f"{RANKED} | score | {di} callback_compliance_pct", "sd2")
    F.add("c09_cb_required_count", "not in the data", f"{README} | callback_compliance_pct | a rate; the count of required callbacks is not given", "text")
    for m, key in [("recheck_compliance_pct", "recheck"), ("record_completion_24h_pct", "record"), ("appts_no_show", "noshow")]:
        r = srow(di, m)
        k = METRIC_KIND[m]
        F.add(f"c09_{key}_last", r.latest, f"{CW} | {m} | {di} | {_w(latest)}", k)
        F.add(f"c09_{key}_avg8", r.avg8, f"{RANKED} | avg8 | {di} {m}", "min1" if k == "count" else k)
        F.add(f"c09_{key}_level_sd", r.level_sd, f"{RANKED} | level_sd | {di} {m}", "sd2")
        F.add(f"c09_{key}_score", r.score, f"{RANKED} | score | {di} {m}", "sd2")
        F.add(f"{key}_peer_median_avg8_c09", peer_median(m, di), f"{RANKED} | avg8 | {m} | median of the other 10 centers", "min1" if k == "count" else k)
        F.add(f"c09_{key}_gap_to_peers", shown_diff(peer_median(m, di), r.avg8, "min1" if k == "count" else k) * (1 if METRICS[m]["better"] == "higher" else -1),
              f"{RANKED} | avg8 | peer median vs {di} {m}, adverse gap, each rounded as shown first", "points1", unit="points")

    # Item 5 Clinic 07
    ve = "Clinic 07"
    r_vc = srow(ve, "staff_call_outs")
    F.add("c07_opened", panel.locations.set_index("location_name").loc[ve, "opened_date"], f"{LOC} | opened_date | {ve}")
    F.add("c07_weeks", int(panel.locations.set_index("location_name").loc[ve, "weeks_available"]), f"{CW} | week_start | {ve} | count", "count")
    F.add("c07_callouts_last", r_vc.latest, f"{CW} | staff_call_outs | {ve} | {_w(latest)}", "count")
    F.add("c07_callouts_prior", series(ve, "staff_call_outs").iloc[-2], f"{CW} | staff_call_outs | {ve} | {_w(co.index[-2])}", "count")
    F.add("c07_callouts_avg8", r_vc.avg8, f"{RANKED} | avg8 | {ve} staff_call_outs", "min1")
    F.add("c07_callouts_long_mean", r_vc.long_mean, f"{RANKED} | long_mean | {ve} staff_call_outs", "min1")
    F.add("c07_callouts_run_weeks", int(r_vc.run_weeks_uncapped), f"{RANKED} | run_weeks_uncapped | {ve} staff_call_outs", "count")
    F.add("c07_callouts_spike_sd", r_vc.spike_sd, f"{RANKED} | spike_sd | {ve} staff_call_outs", "sd2")
    F.add("c07_callouts_score", r_vc.score, f"{RANKED} | score | {ve} staff_call_outs", "sd2")
    F.add("c07_callouts_suppressed", r_vc.suppressed or "none", f"{SUPP} | reason | {ve} staff_call_outs", "text")
    for m, key in [("open_dvm_requisitions", "reqs"), ("record_completion_24h_pct", "record")]:
        r = srow(ve, m)
        k = METRIC_KIND[m]
        F.add(f"c07_{key}_last", r.latest, f"{CW} | {m} | {ve} | {_w(latest)}", k)
        F.add(f"c07_{key}_avg8", r.avg8, f"{RANKED} | avg8 | {ve} {m}", "min1" if k == "count" else k)
        F.add(f"c07_{key}_long_mean", r.long_mean, f"{RANKED} | long_mean | {ve} {m}", "min1" if k == "count" else k)
        F.add(f"c07_{key}_score", r.score, f"{RANKED} | score | {ve} {m}", "sd2")
    hv = series(ve, "doctor_hours_scheduled")
    F.add("c07_hours_start", hv.iloc[-12], f"{CW} | doctor_hours_scheduled | {ve} | {_w(hv.index[-12])}", "hours1")
    F.add("c07_hours_last", hv.iloc[-1], f"{CW} | doctor_hours_scheduled | {ve} | {_w(latest)}", "hours1")
    r_va = srow(ve, "appts_per_doctor_hour")
    va = series(ve, "appts_completed")
    F.add("c07_appts_last", va.iloc[-1], f"{CW} | appts_completed | {ve} | {_w(latest)}", "count")
    F.add("c07_appts_avg8", va.iloc[-8:].mean(), f"{CW} | appts_completed | {ve} | trailing 8 week mean", "min1")
    r_vw = srow(ve, "avg_wait_time_min")
    F.add("c07_wait_last", r_vw.latest, f"{CW} | avg_wait_time_min | {ve} | {_w(latest)}", "minutes1")
    F.add("c07_wait_avg8", r_vw.avg8, f"{RANKED} | avg8 | {ve} avg_wait_time_min", "minutes1")
    r_vcs = srow(ve, "client_csat")
    F.add("c07_csat_last", r_vcs.latest, f"{CW} | client_csat | {ve} | {_w(latest)}", "csat")
    F.add("c07_csat_responses_last", int(r_vcs.denominator), f"{CW} | csat_responses | {ve} | {_w(latest)}", "count")
    F.add("c07_csat_responses_4wk", int(r_vcs.denominator_4wk), f"{RANKED} | denominator_4wk | {ve} client_csat", "count")
    F.add("c07_csat_4wk_weighted", r_vcs.csat_4wk_weighted, f"{RANKED} | csat_4wk_weighted | {ve} client_csat", "csat")
    F.add("c07_csat_long_mean", r_vcs.long_mean, f"{RANKED} | long_mean | {ve} client_csat", "csat")
    F.add("c07_csat_watch_4wk_score", r_vcs.watch_4wk_score, f"{RANKED} | watch_4wk_score | {ve} client_csat", "sd2")

    # Opportunity
    opp = result["opportunity"]
    opp_metric = opp["metric"]
    F.add("opp_center", opp["center"], f"{ITEMS} | opportunity.center", "text")
    F.add("opp_metric", METRIC_SHORT.get(opp_metric, opp["label"]), f"{ITEMS} | opportunity.label", "text")
    F.add("opp_last", opp["latest"], f"{CW} | {opp_metric} | {opp['center']} | {_w(latest)}", METRIC_KIND.get(opp_metric, "auto"))
    F.add("opp_avg8", opp["avg8"], f"{RANKED} | avg8 | {opp['center']} {opp_metric}", METRIC_KIND.get(opp_metric, "auto"))
    F.add("opp_long_mean", opp["long_mean"], f"{RANKED} | long_mean | {opp['center']} {opp_metric}", METRIC_KIND.get(opp_metric, "auto"))
    F.add("opp_peer_median_avg8", peer_median(opp_metric, opp["center"]), f"{RANKED} | avg8 | {opp_metric} | median of the other 10 centers", METRIC_KIND.get(opp_metric, "auto"))
    F.add("opp_spike_sd", opp["spike_sd"], f"{RANKED} | spike_sd | {opp['center']} {opp_metric}", "sd2")
    F.add("opp_level_sd", opp["level_sd"], f"{RANKED} | level_sd | {opp['center']} {opp_metric}", "sd2", note="negative means above the peer median")
    F.add("opp_held_weeks", int(opp["held_weeks_of_6"]), f"{ITEMS} | opportunity.held_weeks_of_6", "count")
    F.add("opp_needed_by", sorted({n["center"] for n in opp["needed_by"]}), f"{ITEMS} | opportunity.needed_by")
    F.add("opp_gap_to_clinic_05", shown_diff(opp["avg8"], r_bc.avg8, METRIC_KIND.get(opp_metric, "pct")), f"{RANKED} | avg8 | {opp['center']} minus {bc} recheck, each rounded as shown first", "points1", unit="points")

    # Plan check summary and groups
    s = result["plan_summary"]
    F.add("plans_total", s["plans"], f"{SUMMARY} | plans", "count")
    F.add("plans_mistargeted", s["mistargeted"], f"{SUMMARY} | mistargeted", "count")
    F.add("plans_baseline_mismatch", s["baseline_mismatch_flags"], f"{SUMMARY} | baseline_mismatch_flags", "count")
    F.add("plans_baseline_denominator", s["baseline_mismatch_denominator"], f"{SUMMARY} | baseline_mismatch_denominator | plans with a panel row", "count")
    F.add("plans_tracking_as_reported", s["plans_tracking_as_reported"], f"{SUMMARY} | plans_tracking_as_reported", "count")
    F.add("plans_in_span", s["in_span"], f"{SUMMARY} | in_span", "count")
    F.add("plans_met_at_open", s["met_at_open"], f"{SUMMARY} | met_at_open | src/plans.py HEADLINE, {s['at_open_window']}", "count")
    F.add("plans_at_open_denominator", s["at_open_denominator"], f"{SUMMARY} | at_open_denominator | plans with a panel row at open", "count")
    F.add("plans_no_data_at_open", s["no_data_at_open"] or ["none"], f"{SUMMARY} | no_data_at_open")
    groups = {
        "close": [pid for pid, p in plans.iterrows() if p.label == "achieved"],
        "close_and_reset": [pid for pid, p in plans.iterrows() if p.label == "mistargeted"],
        "challenge": [pid for pid, p in plans.iterrows() if p.label in ("contradicted", "stalled")],
        "too_early": [pid for pid, p in plans.iterrows() if p.label == "early"],
    }
    F.add("plans_tracking_word", "none" if s["plans_tracking_as_reported"] == 0 else str(s["plans_tracking_as_reported"]),
          f"{SUMMARY} | plans_tracking_as_reported | as a word when zero", "text")
    F.add("noise_band_definition", "1 sd of the center's weekly values in the 8 weeks before the plan opened",
          "src/plans.py | noise_band | BAND_WEEKS", "text")
    for k, v in groups.items():
        F.add(f"plans_group_{k}", v or ["none"], f"{PLANS} | label | in span | grouped")

    # Suppressed and watching
    sup = scores[~scores.eligible]
    F.add("membership_break_date", "2026-02-09", f"{CW} | membership_conversion_pct | definition change date ({README} note 5)", "date")
    F.add("membership_centers_suppressed", int((sup.metric == "membership_conversion_pct").sum()), f"{SUPP} | reason starts with definition change | count", "count")
    ramp = sup[sup.suppressed.str.startswith("ramp")]
    F.add("ramp_centers", sorted(ramp.center.unique().tolist()), f"{SUPP} | reason starts with ramp | centers")
    F.add("ramp_metrics", sorted({METRIC_SHORT[m] for m in ramp.metric.unique()}), f"{SUPP} | reason starts with ramp | metrics")
    for c in ramp.center.unique():
        F.add(f"ramp_weeks_{c.lower().replace(' ', '_')}", int(ramp[ramp.center == c].weeks_available.iloc[0]), f"{SUPP} | weeks_available | {c}", "count")
    watching = sup[sup.suppressed.str.startswith("watching")]
    F.add("watching_rows", [f"{r.center} {METRIC_SHORT.get(r.metric, r.label)}" for r in watching.itertuples()] or ["none"], f"{SUPP} | reason starts with watching")
    hard = sup[sup.suppressed.str.match(r"^(definition change|watch|ramp|insufficient history|no value)")]
    F.add("suppressed_pairs", [{"center": r.center, "metric_label": r.label} for r in hard.itertuples()], f"{SUPP} | center, metric | hard suppressions", "list")
    F.facts["suppressed_pairs"]["display"] = "; ".join(f"{p['center']} {p['metric_label']}" for p in F["suppressed_pairs"])
    F.add("suppressed_total_logged", int(len(sup) + 0) if False else int(json.loads((OUT_DIR / "suppressed.json").read_text(encoding="utf-8"))["count"]), f"{SUPP} | count", "count")

    # Other movements (appendix)
    om = []
    for m in result["other_movements"]:
        key = f"om_{m['center'].lower().replace(' ', '_')}_{m['metric']}"
        k = METRIC_KIND.get(m["metric"], "auto")
        F.add(f"{key}_last", m["latest"], f"{CW} | {m['metric']} | {m['center']} | {_w(latest)}", k)
        F.add(f"{key}_avg8", m["avg8"], f"{RANKED} | avg8 | {m['center']} {m['metric']}", "min1" if k == "count" else k)
        F.add(f"{key}_long_mean", m["long_mean"], f"{RANKED} | long_mean | {m['center']} {m['metric']}", "min1" if k == "count" else k)
        driver = "last week" if (m["spike_sd"] or 0) >= (m["drift_sd"] or 0) and m["direction"] == "adverse" else \
                 "slow drift" if m["direction"] == "adverse" else "improving"
        om.append({"center": m["center"], "metric_label": METRIC_SHORT.get(m["metric"], m["label"]), "direction": m["direction"],
                   "driver": driver, "last": f"{key}_last", "avg8": f"{key}_avg8", "long_mean": f"{key}_long_mean"})

    # Data notes (appendix)
    notes = {n["note"]: n for n in panel.notes}
    F.add("dn_duplicates", notes["duplicate_rows_removed"]["count"], f"{NOTES} | duplicate_rows_removed.count", "count")
    F.add("dn_negative_waits", notes["negative_wait_set_missing"]["count"], f"{NOTES} | negative_wait_set_missing.count", "count")
    F.add("dn_tier_mismatches", notes["maturity_tier_mismatch"]["count"], f"{NOTES} | maturity_tier_mismatch.count", "count")
    F.add("dn_tier_mismatch_centers", panel.tier_mismatches.location_name.tolist(), f"{NOTES} | maturity_tier_mismatch.rows")
    F.add("dn_state_code_errors", notes["state_code_mismatch"]["count"], f"{NOTES} | state_code_mismatch.count", "count")
    F.add("dn_fte_oddities", [r for r in notes["doctor_fte_target_oddity"]["rows"] if r.split(":")[0] in set(names[panel.my_ids])], f"{NOTES} | doctor_fte_target_oddity.rows | in span")

    # Items: headline, owner, signal, supporting lines, plan lines, slots
    item_specs = {
        "Clinic 11": dict(
            what_happened=["c11_rc_20260209", "c11_rc_20260223", "c11_rc_20260316", "c11_rc_20260406", "c11_rc_20260427", "c11_rc_long_mean",
                           "c11_rc_avg8", "c11_rc_run_weeks", "c11_rc_run_start", "c11_doctors", "c11_doctors_fell", "c11_throughput_last",
                           "c11_throughput_avg8", "ap_009_baseline_panel", "ap_009_target", "ap_009_status", "ap_009_label",
                           "c11_duration_before", "c11_duration_after", "c11_duration_change", "ap_001_status", "ap_001_label",
                           "ap_001_actual", "ap_001_target", "ap_001_baseline_panel"],
            why_it_matters=["c11_records_late_last_week", "c11_appts_last", "c11_rc_20260427"],
            what_to_do=["ap_001_owner", "next_digest_date", "c11_doctors"],
            plans=["AP_001", "AP_009"], owner="ap_001_owner", check="next_digest_date",
            tables={"weeks": True, "doctors": doctor_rows},
        ),
        "Clinic 05": dict(
            what_happened=["c05_recheck_last", "c05_recheck_avg8", "recheck_peer_median_avg8", "c05_gap_to_peer_median", "dvm0024_recheck_hw",
                           "dvm0024_recheck_mean", "dvm0024_peers_hw", "dvm0024_gap", "dvm0024_weeks", "dvm0024_tenure", "dvm0024_employment",
                           "dvm0024_record_hw", "dvm0024_record_peers_hw", "ap_002_status", "ap_002_label", "ap_002_actual", "ap_002_target"],
            why_it_matters=["c05_clinic_recheck_8wk_hw", "c05_clinic_recheck_if_dvm0024_at_peers", "c05_counterfactual_lift", "c05_gap_to_target_after",
                            "ap_002_target", "dvm0024_peers_hw", "recheck_peer_median_avg8"],
            what_to_do=["ap_002_owner", "two_week_check_date"],
            plans=["AP_002"], owner="ap_002_owner", check="two_week_check_date",
            tables={"peers": peer_rows},
        ),
        "Clinic 06": dict(
            what_happened=["c06_callouts_last4", "c06_callouts_last", "c06_callouts_prior", "c06_callouts_weeks_at_latest", "c06_callouts_before_march_median",
                           "c06_callouts_before_march_p75", "c06_callouts_long_mean", "c06_hours_start", "c06_hours_last", "c06_hours_change_pct",
                           "c06_open_reqs_last", "c06_open_reqs_avg8", "c06_wait_last", "c06_wait_avg8", "ap_003_status", "ap_003_label", "ap_024_label"],
            why_it_matters=["c06_appts_start", "c06_appts_last", "c06_appts_change_pct", "c06_weeks_7_or_more"],
            what_to_do=["rop_name", "next_digest_date"],
            plans=["AP_003", "AP_024"], owner="rop_name", check="next_digest_date", tables={},
        ),
        "Clinic 09": dict(
            what_happened=["c09_cb_last", "c09_cb_avg8", "c09_cb_weeks_at_ceiling_last10", "window_10wk", "integrity_level", "c09_cb_weeks_at_ceiling_all",
                           "c09_weeks_available", "c09_cb_required_count", "c09_recheck_avg8", "recheck_peer_median_avg8_c09", "c09_record_avg8",
                           "record_peer_median_avg8_c09", "c09_noshow_last", "c09_noshow_avg8"],
            why_it_matters=["c09_cb_required_count", "c09_recheck_gap_to_peers", "c09_record_gap_to_peers", "c09_recheck_avg8", "c09_record_avg8"],
            what_to_do=["rop_name", "next_digest_date", "c09_cb_required_count"],
            plans=[], owner="rop_name", check="next_digest_date", tables={},
        ),
        "Clinic 07": dict(
            what_happened=["c07_opened", "c07_weeks", "c07_callouts_last", "c07_callouts_prior", "c07_callouts_avg8", "c07_callouts_long_mean",
                           "c07_callouts_run_weeks", "c07_reqs_last", "c07_reqs_avg8", "c07_record_last", "c07_record_avg8", "c07_record_long_mean",
                           "c07_hours_start", "c07_hours_last"],
            why_it_matters=["c07_weeks", "c07_hours_start", "c07_hours_last", "c07_appts_last", "c07_appts_avg8", "c07_wait_last", "c07_wait_avg8"],
            what_to_do=["rop_name", "next_digest_date"],
            plans=[], owner="rop_name", check="next_digest_date", tables={},
        ),
    }
    # Item 2 why it matters is written by code: arithmetic restated is never a slot.
    # P1: the peers' number is left out of the sentence, it collides with the counterfactual on display.
    ITEM2_WHY_FACTS = ["c05_clinic_recheck_if_dvm0024_at_peers", "c05_gap_to_target_after", "ap_002_target"]
    CODE_WHY = {"Clinic 05": ("If DVM_0024 matched the other Clinic 05 doctors, the clinic would reach "
                                "{c05_clinic_recheck_if_dvm0024_at_peers}, still {c05_gap_to_target_after} points short of the "
                                "{ap_002_target} target.", ITEM2_WHY_FACTS)}
    slots = {}
    digest_items = []
    for it in items:
        n = it["rank"]
        lead = it["lead"]
        spec = item_specs[it["center"]]
        row = scores[(scores.center == it["center"]) & (scores.metric == lead["metric"])].iloc[0]
        stype = signal_type(row)
        short = METRIC_SHORT.get(lead["metric"], lead["label"].lower())
        F.add(f"item{n}_rank", n, f"{ITEMS} | items.rank", "count")
        F.add(f"item{n}_center", it["center"], f"{ITEMS} | items.center | rank {n}", "text")
        F.add(f"item{n}_score", it["score"], f"{ITEMS} | items.score | rank {n}", "sd2")
        F.add(f"item{n}_signal", stype, f"{RANKED} | spike_sd, drift_sd, level_sd, integrity | {it['center']} {lead['metric']} | largest term", "text")
        F.add(f"item{n}_owner", F[spec["owner"]], f"{AP} owner_name or {LOC} rop_name | item {n}", "owner")
        F.add(f"item{n}_check_date", F[spec["check"]], f"src/run.py | --date plus 7 or 14 days | item {n}", "date")
        F.add(f"item{n}_check_back", check_back(F[spec["check"]]), f"src/run.py | --date plus 7 or 14 days | item {n} | weekday month day", "text")
        F.add(f"item{n}_check_back_short", check_back(F[spec["check"]], weekday=False), f"src/run.py | --date plus 7 or 14 days | item {n} | month day", "text")
        adverse_word = "down" if METRICS[lead["metric"]]["better"] == "higher" else "up"
        run_n = int(row.run_weeks_uncapped)
        F.add(f"item{n}_run_weeks", run_n, f"{RANKED} | run_weeks_uncapped | {it['center']} {lead['metric']}", "count")
        if stype == "integrity":
            headline = f"{it['center']}: {short} at the ceiling, verify the logging"
        elif stype == "drift":
            norm_word = "below its norm" if adverse_word == "down" else "above its norm"
            headline = f"{it['center']}: {short} {norm_word} {run_n} weeks running"
        elif stype == "level":
            gap = (peer_median(lead["metric"], it["center"]) - row.avg8) * (1 if adverse_word == "down" else -1)
            F.add(f"item{n}_gap_to_peers", gap, f"{RANKED} | avg8 | peer median vs {it['center']} {lead['metric']}", "points1", unit="points")
            headline = f"{it['center']}: {short} {F.d(f'item{n}_gap_to_peers')} points {'below' if adverse_word == 'down' else 'above'} peers"
            outl = out[(out.location_id == it["location_id"]) & (out.metric == lead["metric"]) & out.outlier]
            if len(outl):
                worst = outl.sort_values("rate_hours_weighted").iloc[0]
                F.add(f"item{n}_one_doctor_rate", worst.rate_hours_weighted, f"{PW} | {lead['metric']} hours weighted | {worst.provider_id} | last 8 weeks", METRIC_KIND.get(lead["metric"], "pct"))
                headline += f"; one doctor at {F.d(f'item{n}_one_doctor_rate')}"
        else:
            latest_disp = display(row.latest, METRIC_KIND.get(lead["metric"], "auto"))
            headline = (f"{it['center']}: {short} at {latest_disp}, {adverse_word} {run_n} weeks running" if run_n > 1
                        else f"{it['center']}: {short} at {latest_disp} last week")
        F.add(f"item{n}_headline", headline, f"{ITEMS} | items | templated in src/facts.py by signal type", "text")
        support = []
        for s_ in it["supporting"]:
            srow_ = scores[(scores.center == it["center"]) & (scores.metric == s_["metric"])].iloc[0]
            terms = {"spike": srow_.spike_sd if pd.notna(srow_.spike_sd) else -9, "drift": srow_.drift_sd if pd.notna(srow_.drift_sd) else -9,
                     "level": srow_.level_sd if pd.notna(srow_.level_sd) else -9}
            driver = max(terms, key=terms.get)
            if terms[driver] < ALSO_MOVING_MIN_SD:
                continue  # the gap the reader would see is inside the noise band
            key = f"item{n}_also_{s_['metric']}"
            k = METRIC_KIND.get(s_["metric"], "auto")
            avgk = "min1" if k == "count" else k
            F.add(f"{key}_last", s_["latest"], f"{CW} | {s_['metric']} | {it['center']} | {_w(latest)}", k)
            F.add(f"{key}_avg8", s_["avg8"], f"{RANKED} | avg8 | {it['center']} {s_['metric']}", avgk)
            F.add(f"{key}_long_mean", srow_.long_mean, f"{RANKED} | long_mean | {it['center']} {s_['metric']}", avgk)
            F.add(f"{key}_peer_median", peer_median(s_["metric"], it["center"]), f"{RANKED} | avg8 | {s_['metric']} | median of the other 10 centers", avgk)
            support.append({"metric_label": METRIC_SHORT.get(s_["metric"], s_["label"]), "driver": driver,
                            "last": f"{key}_last", "avg8": f"{key}_avg8", "long_mean": f"{key}_long_mean", "peer_median": f"{key}_peer_median"})
        base_allowed = [f"item{n}_headline", f"item{n}_center", "latest_week", "window_8wk", "window_4wk"]
        if lead["metric"] == "record_completion_24h_pct" or any(s_["metric"] == "record_completion_24h_pct" for s_ in it["supporting"]):
            base_allowed.append("record_window_hours")
        code_why = CODE_WHY.get(it["center"])
        item_slots = []
        for part in ("what_happened", "why_it_matters", "what_to_do"):
            if part == "why_it_matters" and code_why:
                continue
            sid = f"item{n}_{part}"
            extra = [f"item{n}_owner", f"item{n}_check_date"] if part == "what_to_do" else []
            slots[sid] = {"cap": SLOT_CAPS[part], "section": f"item{n}", "allowed_facts": allowed(base_allowed + spec[part] + extra
                          + [x for s in support for x in (s["last"], s["avg8"], s["long_mean"], s["peer_median"])])}
            item_slots.append(sid)
        digest_items.append({
            "rank": n, "center": it["center"], "location_id": it["location_id"], "lane": it["lane"],
            "lead_metric": lead["metric"], "lead_label": lead["label"], "signal": stype,
            "headline": f"item{n}_headline", "owner": f"item{n}_owner", "check_date": f"item{n}_check_date",
            "check_back": f"item{n}_check_back", "check_back_short": f"item{n}_check_back_short",
            "slots": item_slots, "code_why": None if not code_why else {"template": code_why[0], "facts": code_why[1]},
            "also_moving": support, "plans": spec["plans"], "tables": spec["tables"],
        })

    # First surfaced on: the ranking run as of each prior Monday with data up to that Monday (src/surface.py).
    from src.surface import first_surfaced, last_week_items
    fs = first_surfaced(panel, items).set_index("item")
    last_week_items(panel, scores)  # outputs/last_week_items.csv for the page's last week strip
    for it in items:
        n = it["rank"]
        row = fs.loc[n] if n in fs.index else fs.loc[str(n)]
        F.add(f"item{n}_first_surfaced", row["first_monday"], f"outputs/first_surfaced.csv | first_monday | item {n} | src/surface.py, ranking as of each prior Monday",
              "date" if row["first_monday"] != "this week" else "text")
        F.add(f"item{n}_mondays_before_latest_week", int(row["mondays_before_latest_week"]), f"outputs/first_surfaced.csv | mondays_before_latest_week | item {n} | Mondays from first_monday to the latest complete week", "count")

    # One thing today: selection by rule
    one = None
    for it in items:
        row = scores[(scores.center == it["center"]) & (scores.metric == it["lead"]["metric"])].iloc[0]
        if it["lane"] == "ops" and pd.notna(row.spike_sd) and row.spike_sd >= 3.0 and row.run_weeks_uncapped >= 2:
            one = it
            break
    one = one or items[0]
    F.add("one_thing_item_rank", one["rank"], f"{ITEMS} | items.rank | selected by ONE_THING_RULE", "count")
    F.add("one_thing_center", one["center"], f"{ITEMS} | items.center | selected by ONE_THING_RULE", "text")
    one_allowed = allowed(["one_thing_center", "one_thing_item_rank", "rop_name", "next_digest_date", "c06_callouts_last", "c06_callouts_prior",
                           "c06_callouts_weeks_at_latest", "c06_weeks_7_or_more", "c06_hours_change_pct", "c06_appts_change_pct", "c06_hours_last"])
    slots["one_thing_today_why"] = {"cap": SLOT_CAPS["one_thing_today_why"], "section": "one_thing_today", "allowed_facts": one_allowed}
    slots["opportunity"] = {"cap": SLOT_CAPS["opportunity"], "section": "opportunity",
                            "allowed_facts": allowed(["opp_center", "opp_metric", "opp_last", "opp_avg8", "opp_held_weeks",
                                                      "window_6wk", "opp_needed_by", "opp_gap_to_clinic_05", "c05_recheck_avg8", "metric_def_recheck",
                                                      "item2_center"])}
    d = F.d
    suppressed_blocks = [
        {"id": "suppressed_membership", "what": "Membership conversion, every center",
         "lead": f"Membership conversion, all {d('membership_centers_suppressed')} centers: definition changed {d('membership_break_date')}, not ranked.",
         "allowed": ["membership_centers_suppressed", "membership_break_date"]},
        {"id": "suppressed_clinic_07_csat", "what": "Clinic 07 client satisfaction",
         "lead": (f"Clinic 07 client satisfaction: watch. {d('c07_csat_last')} on {d('c07_csat_responses_last')} responses last week, "
                  f"{d('c07_csat_4wk_weighted')} on {d('c07_csat_responses_4wk')} over {d('window_4wk')} weeks."),
         "allowed": ["c07_csat_last", "c07_csat_responses_last", "c07_csat_responses_4wk", "c07_csat_4wk_weighted",
                     "c07_csat_long_mean", "min_csat_responses", "next_digest_date", "c07_weeks", "window_4wk"]},
        {"id": "suppressed_ramp", "what": "Throughput, revenue and no shows at Clinic 10 and Clinic 07",
         "lead": (f"Throughput, revenue and no shows at Clinic 10 ({d('ramp_weeks_clinic_10')} weeks) and Clinic 07 ({d('ramp_weeks_clinic_07')} weeks): "
                  f"ramp, under {d('ramp_weeks')} weeks of history."),
         "allowed": ["ramp_centers", "ramp_metrics", "ramp_weeks_clinic_10", "ramp_weeks_clinic_07", "ramp_weeks", "ap_019_label", "ap_027_label"]},
    ]
    if F["watching_rows"] != ["none"]:
        suppressed_blocks.insert(1, {"id": "suppressed_watching", "what": "Watching at new centers",
                                     "lead": f"Watching at new centers: {d('watching_rows')}. Spike only, run under {d('new_center_min_run')} weeks.",
                                     "allowed": ["watching_rows", "new_center_min_run", "ramp_weeks", "next_digest_date"]})
    # No slots in Plan check or Watching and suppressed: code says everything there (part three).
    plan_check_facts = allowed(["plans_in_span", "plans_total", "plans_met_at_open", "plans_at_open_denominator", "plans_no_data_at_open",
                                "plans_mistargeted", "plans_tracking_word", "plans_group_close",
                                "plans_group_close_and_reset", "plans_group_challenge", "plans_group_too_early"])

    # Evidence index: every fact a slot or a code line may use, numbered E1.. in body order, plus the table facts.
    evidence_tables = []
    order = [("One thing today", slots["one_thing_today_why"]["allowed_facts"])]
    for di_ in digest_items:
        fids = []
        for sid in di_["slots"]:
            fids += slots[sid]["allowed_facts"]
        if di_["code_why"]:
            fids += di_["code_why"]["facts"]
        for t in di_["tables"].values():
            if isinstance(t, list):
                for rowd in t:
                    fids += [v for k, v in rowd.items() if k != "provider_id"]
        order.append((f"Item {di_['rank']}: {di_['center']}", allowed(fids)))
    order.append(("Plan check", plan_check_facts))
    order.append(("Opportunity", slots["opportunity"]["allowed_facts"]))
    for b in suppressed_blocks:
        order.append((b["what"], allowed(b["allowed"])))
    index = {}
    for title, fids in order:
        rows = []
        for fid in fids:
            if fid not in index:
                index[fid] = f"E{len(index) + 1}"
            rows.append(fid)
        evidence_tables.append({"title": title, "facts": rows})

    digest = {
        "header": {"leader": "leader", "date": "digest_date", "centers": "n_centers", "latest_week": "latest_week", "rop": "rop_name"},
        "at_a_glance": [{"rank": d["rank"], "headline": d["headline"], "owner": d["owner"], "check_date": d["check_date"],
                         "check_back_short": d["check_back_short"]} for d in digest_items],
        "one_thing_today": {"center": "one_thing_center", "item_rank": "one_thing_item_rank", "who_to_call": "rop_name",
                            "check_date": "next_digest_date", "check_back": "one_thing_check_back", "slot": "one_thing_today_why", "rule": ONE_THING_RULE},
        "items": digest_items,
        "plan_check": {"plans": list(plans.index), "cross_region": ["plans_total", "plans_mistargeted", "plans_tracking_word"],
                       "appendix_note": ["plans_baseline_mismatch", "plans_baseline_denominator"],
                       "groups": {k: f"plans_group_{k}" for k in groups}},
        "opportunity": {"slot": "opportunity", "held_test": "weeks in the last 6 above the center's own long run mean",
                        "good_center_test": "8 week average at or above the median of the other 10 centers"},
        "suppressed": suppressed_blocks,
        "other_movements": om,
        "data_notes": ["dn_duplicates", "dn_negative_waits", "dn_tier_mismatches", "dn_tier_mismatch_centers", "dn_state_code_errors",
                       "dn_fte_oddities", "c09_cb_weeks_at_ceiling_last10"],
        "feedback": {"file": FEEDBACK_FILE, "columns": FEEDBACK_COLUMNS, "verdicts": FEEDBACK_VERDICTS,
                     "items": [f"item{d['rank']}" for d in digest_items] + ["one_thing_today", "opportunity"],
                     "free_text_row": "Anything we missed? (free text)"},
        "slots": slots,
        "evidence_index": index,
        "evidence_tables": evidence_tables,
        "body_cap": BODY_CAP,
        "top_cap": TOP_CAP,
    }
    return {
        "meta": {"generated_for": str(panel.today.date()), "latest_week": str(latest.date()), "leader": panel.leader,
                 "source_format": "file | column | scope | window", "fact_count": len(F.facts), "slot_count": len(slots)},
        "facts": F.facts,
        "digest": digest,
    }


def slot_inputs(fx: dict) -> dict:
    """What the prose model receives: per slot, the cap, only the facts it may use, and the code written
    lines already in that section (already_stated), which it must not restate."""
    from src.write import already_stated
    F = fx["facts"]
    stated = already_stated(fx)
    return {sid: {"cap_words": s["cap"], "already_stated": stated.get(sid, []),
                  "facts": {fid: {"display": F[fid]["display"], "source": F[fid]["source"],
                                  **({"note": F[fid]["note"]} if "note" in F[fid] else {})}
                            for fid in s["allowed_facts"]}}
            for sid, s in fx["digest"]["slots"].items()}


def write_facts(facts: dict, path: Path = OUT_DIR / "digest_facts.json") -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(facts, f, indent=1, default=str)
    return path
