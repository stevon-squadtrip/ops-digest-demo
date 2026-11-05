"""Action plan check. One row per plan, every plan, with one label, stacked flags, and a reason.

Rules (CLAUDE.md final order):
- clinic_weekly is the source of truth. panel_baseline is the value in the week the plan opened
  (the Monday on or before opened_date). The recorded baseline is kept for comparison.
- actual is the trailing 4 week average. Last week is shown alongside.
- noise band is 1 sd of the center's weekly values on that metric in the 8 weeks before the plan
  opened. With fewer than 6 of those weeks, the median band across the leader's mature centers.
- Labels, first match wins: mistargeted, achieved, contradicted, stalled, early, confirmed.
  confirmed is the catch-all: no exception found against the owner's status.
- Flags stack and never replace a label: stale, past due, baseline mismatch, no panel row,
  linked metric slipped (target met or status complete, and a linked metric at the same center,
  from LINKED_METRICS, with an adverse run of at least LINKED_MIN_RUN_WEEKS that started after
  the plan opened and scores at or above the supporting line).
Suppression in the ranking never touches this file. Every plan is reported.
"""
import pandas as pd

from src.clean import Panel
from src.metrics import METRICS

STALE_DAYS = 28
EARLY_WEEKS = 6
ACTUAL_WEEKS = 4
BAND_WEEKS = 8
BAND_MIN_WEEKS = 6
STALLED_PROGRESS = 0.5
STALLED_ELAPSED = 0.5
RANKED_ITEM_SCORE = 3.0  # same line as rank.ITEM_THRESHOLD; rank passes its own value in
LINKED_MIN_SCORE = 2.0  # same line as rank.SUPPORT_THRESHOLD; rank passes its own value in
# Minimum evidence for the linked metric slipped flag: the adverse run must be at least this many
# weeks. A one-week run is normal variation and this flag reaches the digest.
LINKED_MIN_RUN_WEEKS = 3
# Metric pairs that share doctor time or the client visit. A plan on one may carry the flag when
# the other slips. Unordered. Nothing outside this list is ever linked.
LINKED_METRICS = frozenset({
    frozenset({"appts_per_doctor_hour", "record_completion_24h_pct"}),
    frozenset({"appts_per_doctor_hour", "avg_wait_time_min"}),
    frozenset({"appts_per_doctor_hour", "client_csat"}),
    frozenset({"appts_per_doctor_hour", "recheck_compliance_pct"}),
    frozenset({"client_csat", "avg_wait_time_min"}),
})
LINKED_FLAG = "linked metric slipped"
LABEL_ORDER = ["mistargeted", "achieved", "contradicted", "stalled", "early", "confirmed"]

# Headline definitions, pinned 2026-10-21. One place. The submission email's two lead numbers come from these.
HEADLINE = {
    # "met target at open": the panel value in the last clinic week on or before opened_date (the Monday on or
    # before it), compared with target_value in the metric's own direction (higher is better for rates and
    # throughput, lower is better for avg_wait_time_min and counts of bad events). Direction comes from
    # src/metrics.py METRICS[metric]["better"], applied through meets().
    "at_open_window": "last_week_on_or_before_open",
    "at_open_windows_considered": ["last_week_on_or_before_open", "first_week_on_or_after_open", "trailing_4_week_mean", "trailing_8_week_mean"],
    # A plan whose center has no clinic row on or before the opening Monday is "no data at open" and leaves the
    # denominator. It is never counted as not met.
    "no_data_at_open_label": "no data at open",
    # "tracking as reported without a flag": label is confirmed, achieved or early, and the flag list is empty.
    # Flags: stale (last update more than STALE_DAYS ago), past due (due date passed, status not complete),
    # baseline mismatch (recorded baseline_value differs from the panel value at open by more than the noise band),
    # no panel row, linked metric slipped (a paired metric slid 3 or more weeks after the plan opened).
    "tracking_labels": ["confirmed", "achieved", "early"],
}
# Plain words for the reader. Code labels stay in every output file; these are what the digest shows.
LABEL_DISPLAY = {"contradicted": "data disagrees", "mistargeted": "target met at open", "stalled": "stalled",
                 "achieved": "done", "early": "too soon to judge", "confirmed": "confirmed"}
FLAGS = ["stale", "past due", "baseline mismatch", "no panel row", LINKED_FLAG]
LINKED_ACTION = "Check the linked metric first."
MISTARGETED_PAST_TARGET_ACTION = "Close it. If the metric still needs to move, re-open with a target set from the panel."
ACTIONS = {
    "mistargeted": "Re-set the target from the panel baseline, then decide if the plan is still needed.",
    "achieved": "Close it.",
    "contradicted": "Update the status to match the data, or flag the data as wrong.",
    "stalled": "Change the approach or close it.",
    "early": "Too soon to judge. Note the direction.",
    "confirmed": "Keep going.",
}


def better(metric: str) -> str:
    return METRICS.get(metric, {}).get("better", "higher")


def adverse_sign(metric: str) -> int:
    return -1 if better(metric) == "higher" else 1


def meets(value: float, target: float, metric: str) -> bool:
    return value >= target if better(metric) == "higher" else value <= target


def monday_on_or_before(date: pd.Timestamp) -> pd.Timestamp:
    return (date - pd.Timedelta(days=date.weekday())).normalize()


def _window_mean(series: pd.Series, end_week, weeks: int) -> float:
    window = series[series.index <= end_week].iloc[-weeks:]
    return float(window.mean()) if len(window) else float("nan")


def noise_band(panel: Panel, lid: str, metric: str, baseline_week: pd.Timestamp) -> tuple:
    """1 sd of the center's weekly values in the 8 weeks before the plan opened, or the mature median."""
    clinic = panel.clinic.sort_values("week_start")

    def own(center_id):
        s = clinic[clinic.location_id == center_id].set_index("week_start")[metric]
        hist = s[s.index < baseline_week].iloc[-BAND_WEEKS:].dropna()
        return float(hist.std()) if len(hist) >= BAND_MIN_WEEKS else float("nan")

    band = own(lid)
    if pd.notna(band):
        return band, "own 8 weeks before opening"
    mature = panel.my_locations[panel.my_locations.maturity_tier == "mature"].location_id
    bands = pd.Series([own(c) for c in mature]).dropna()
    if len(bands):
        return float(bands.median()), f"median of {len(bands)} mature centers in the leader's span"
    return float("nan"), "no band available"


def plan_check(panel: Panel, scores: pd.DataFrame = None, in_span_only: bool = False,
               actual_weeks: int = ACTUAL_WEEKS, item_threshold: float = RANKED_ITEM_SCORE,
               linked_threshold: float = LINKED_MIN_SCORE, band_multiplier: float = 1.0) -> pd.DataFrame:
    """Every plan with recomputed baseline, actual, label, flags, reason and action.

    scores: the ranking table, used only for the linked metric slipped flag. Without it the flag
    cannot fire, which is fine for the first pass that feeds the ranking context.
    band_multiplier: scales the noise band, for the tolerance report only. 1.0 is the rule.
    """
    names = panel.locations.set_index("location_id").location_name
    rmp = panel.locations.set_index("location_id").rmp_name
    clinic = panel.clinic.sort_values(["location_id", "week_start"])
    plans = panel.my_plans if in_span_only else panel.plans
    rows = []
    for r in plans.itertuples():
        g = clinic[clinic.location_id == r.location_id].set_index("week_start")
        metric = r.target_metric
        spec = METRICS.get(metric, {})
        s = g[metric] if metric in g.columns else pd.Series(dtype=float)
        sign = adverse_sign(metric)
        bweek = monday_on_or_before(r.opened_date)

        panel_baseline = float(s[bweek]) if bweek in s.index and pd.notna(s.get(bweek)) else float("nan")
        no_panel_row = pd.isna(panel_baseline)
        # Robustness check (CLAUDE.md Reporting rules): the same two tests on the trailing 4 weeks
        # ending at the opening week. Labels and flags still use the single week. robust says whether
        # both windows agree on mistargeted and on baseline mismatch.
        panel_base_4wk = _window_mean(s, bweek, ACTUAL_WEEKS) if not no_panel_row else float("nan")
        base = float(r.baseline_value) if no_panel_row else panel_baseline
        band, band_source = noise_band(panel, r.location_id, metric, bweek)
        if pd.notna(band):
            band = band * band_multiplier
        diff = float("nan") if no_panel_row else float(r.baseline_value) - panel_baseline

        latest_week = s.index[-1] if len(s) else panel.latest_week
        actual = _window_mean(s, latest_week, actual_weeks)
        last_week = float(s.iloc[-1]) if len(s) else float("nan")
        prior_end = s.index[-actual_weeks - 1] if len(s) > actual_weeks else None
        prior = _window_mean(s, prior_end, actual_weeks) if prior_end is not None else float("nan")

        weeks_open = (panel.today - r.opened_date).days / 7
        days_since_update = (panel.today - r.last_status_update).days
        window_days = max((r.due_date - r.opened_date).days, 1)
        elapsed = (panel.today - r.opened_date).days / window_days
        past_due = r.due_date < panel.today and r.status != "complete"
        target_met = pd.notna(actual) and meets(actual, r.target_value, metric)
        # Headline field: the panel value at open against the target, in the metric's direction. HEADLINE block above.
        met_at_open = HEADLINE["no_data_at_open_label"] if no_panel_row else ("yes" if meets(panel_baseline, r.target_value, metric) else "no")
        span = r.target_value - base
        progress = (actual - base) / span if pd.notna(actual) and span != 0 else float("nan")
        worse_by = (actual - base) * sign if pd.notna(actual) else float("nan")

        # Linked metric slipped: target met or status complete, and a linked metric (LINKED_METRICS)
        # at this center scoring at or above the supporting line, adverse, with a run of at least
        # LINKED_MIN_RUN_WEEKS that started after the plan opened. Suppressed rows never count.
        effects = []
        if scores is not None and (r.status == "complete" or target_met):
            at_center = scores[(scores.location_id == r.location_id) & (scores.metric != metric)
                               & scores.eligible & (scores.score >= linked_threshold) & (scores.direction == "adverse")]
            for e in at_center.itertuples():
                if frozenset({metric, e.metric}) not in LINKED_METRICS:
                    continue
                start = pd.Timestamp(e.run_start_week) if e.run_start_week else None
                if start is not None and start > r.opened_date and e.run_weeks_uncapped >= LINKED_MIN_RUN_WEEKS:
                    effects.append(f"{e.label} slid from {e.long_mean:.1f} to {e.latest:.1f}, "
                                   f"adverse run of {int(e.run_weeks_uncapped)} weeks since {start.date()}")

        # One label, first match wins.
        if not no_panel_row and meets(panel_baseline, r.target_value, metric):
            label = "mistargeted"
            reason = (f"panel baseline {panel_baseline:.2f} in week {bweek.date()} already met target "
                      f"{r.target_value:.2f}; recorded baseline was {r.baseline_value:.2f}")
        elif target_met and r.status != "complete":
            label = "achieved"
            reason = f"actual {actual:.2f} meets target {r.target_value:.2f} while status says {r.status}"
        elif r.status in ("on_track", "complete") and pd.notna(band) and worse_by > band:
            label = "contradicted"
            reason = (f"status {r.status} but actual {actual:.2f} is worse than the panel baseline {base:.2f} "
                      f"by {worse_by:.2f}, over the noise band {band:.2f}")
        elif (past_due and not target_met) or (elapsed > STALLED_ELAPSED and pd.notna(progress) and progress < STALLED_PROGRESS):
            label = "stalled"
            bits = []
            if past_due and not target_met:
                bits.append(f"due {r.due_date.date()} has passed with actual {actual:.2f} short of target {r.target_value:.2f}")
            if elapsed > STALLED_ELAPSED and pd.notna(progress) and progress < STALLED_PROGRESS:
                bits.append(f"{elapsed:.0%} of the window gone with progress {progress:.0%}")
            reason = "; ".join(bits)
        elif weeks_open < EARLY_WEEKS:
            label = "early"
            reason = f"opened {weeks_open:.1f} weeks ago; actual {actual:.2f} vs panel baseline {base:.2f}, progress {progress:.0%}" \
                if pd.notna(progress) else f"opened {weeks_open:.1f} weeks ago; actual {actual:.2f} vs baseline {base:.2f}"
        else:
            label = "confirmed"
            reason = (f"status {r.status} matches the data: actual {actual:.2f} vs baseline {base:.2f}, "
                      f"target {r.target_value:.2f}, progress {progress:.0%}, {elapsed:.0%} of the window gone")

        flags = []
        if days_since_update > STALE_DAYS:
            flags.append("stale")
        if past_due:
            flags.append("past due")
        if not no_panel_row and pd.notna(band) and abs(diff) > band:
            flags.append("baseline mismatch")
        if no_panel_row:
            flags.append("no panel row")
        if effects:
            flags.append(LINKED_FLAG)
        base_action = MISTARGETED_PAST_TARGET_ACTION if (label == "mistargeted" and target_met) else ACTIONS[label]
        action = base_action + (f" {LINKED_ACTION}" if effects else "")
        # Trend word for the reader's plan line: how the trailing 4 weeks compare with the 4 before.
        if pd.notna(prior) and pd.notna(actual):
            move = (actual - prior) * sign
            if move > 0:
                trend_word = "falling" if better(metric) == "higher" else "rising"
            elif move < 0:
                trend_word = "recovering"
            else:
                trend_word = "flat"
        else:
            trend_word = "flat"
        if no_panel_row or pd.isna(panel_base_4wk) or pd.isna(band):
            robust, robust_note = None, "no panel row, no check possible" if no_panel_row else "no band, no check possible"
        else:
            mis_1wk = meets(panel_baseline, r.target_value, metric)
            mis_4wk = meets(panel_base_4wk, r.target_value, metric)
            mm_1wk = abs(float(r.baseline_value) - panel_baseline) > band
            mm_4wk = abs(float(r.baseline_value) - panel_base_4wk) > band
            robust = bool(mis_1wk == mis_4wk and mm_1wk == mm_4wk)
            robust_note = (f"mistargeted 1wk {mis_1wk} 4wk {mis_4wk}; baseline mismatch 1wk {mm_1wk} 4wk {mm_4wk} "
                           f"(4wk base {panel_base_4wk:.2f} vs target {r.target_value:.2f}, recorded {r.baseline_value:.2f}, band {band:.2f})")
        note = ""
        if pd.notna(prior) and pd.notna(actual):
            move = (actual - prior) * sign
            note = f"vs the prior {actual_weeks} weeks: {prior:.2f} to {actual:.2f}, {'worse' if move > 0 else 'better'} by {abs(move):.2f}"
        if no_panel_row:
            note = (note + "; " if note else "") + "center opened after the plan, recorded baseline used"

        rows.append({
            "plan_id": r.plan_id, "location_id": r.location_id, "location_name": names[r.location_id],
            "rmp_name": rmp[r.location_id], "in_span": r.location_id in panel.my_ids,
            "owner_name": r.owner_name, "target_metric": metric, "metric_label": spec.get("label", metric),
            "lane": spec.get("lane", ""), "status": r.status,
            "baseline_recorded": float(r.baseline_value), "baseline_week": bweek.date(),
            "baseline_panel": None if no_panel_row else round(panel_baseline, 2),
            "baseline_diff": None if no_panel_row else round(diff, 2), "baseline_used": round(base, 2),
            "panel_base_1wk": None if no_panel_row else round(panel_baseline, 2),
            "panel_base_4wk": None if pd.isna(panel_base_4wk) else round(panel_base_4wk, 2),
            "robust": robust, "robust_note": robust_note,
            "noise_band": None if pd.isna(band) else round(band, 2), "band_source": band_source,
            "target_value": float(r.target_value), "actual": None if pd.isna(actual) else round(actual, 2),
            "actual_weeks": actual_weeks, "last_week": None if pd.isna(last_week) else round(last_week, 2),
            "prior": None if pd.isna(prior) else round(prior, 2),
            "worse_than_baseline_by": None if pd.isna(worse_by) else round(worse_by, 2),
            "progress": None if pd.isna(progress) else round(progress, 3), "window_elapsed": round(elapsed, 3),
            "target_met": bool(target_met), "met_at_open": met_at_open,
            "baseline_gap_pct": None if no_panel_row or not panel_baseline else round(diff / panel_baseline * 100, 1),
            "opened_date": r.opened_date.date(), "weeks_open": round(weeks_open, 1),
            "due_date": r.due_date.date(), "weeks_past_due": int((panel.today - r.due_date).days // 7) if past_due else None,
            "last_status_update": r.last_status_update.date(),
            "days_since_update": int(days_since_update), "latest_week": latest_week.date(),
            "label": label, "label_display": LABEL_DISPLAY[label], "flags": flags, "reason": reason, "note": note,
            "action": action, "trend_word": trend_word, "linked_metrics": effects,
        })
    return pd.DataFrame(rows)


def plan_summary(plans: pd.DataFrame) -> dict:
    """Counts the memo uses: plans per label, plans per flag, and the two headline shares."""
    n = len(plans)
    per_label = {lab: int((plans.label == lab).sum()) for lab in LABEL_ORDER}
    per_flag = {flag: int(plans["flags"].apply(lambda f: flag in f).sum()) for flag in FLAGS}
    tracking = plans[plans.label.isin(HEADLINE["tracking_labels"]) & plans["flags"].apply(lambda f: len(f) == 0)]
    with_data = plans[plans.met_at_open != HEADLINE["no_data_at_open_label"]]
    return {
        "plans": n,
        "mistargeted": per_label["mistargeted"],
        "met_at_open": int((with_data.met_at_open == "yes").sum()),
        "at_open_denominator": int(len(with_data)),
        "no_data_at_open": plans[plans.met_at_open == HEADLINE["no_data_at_open_label"]].plan_id.tolist(),
        "at_open_window": HEADLINE["at_open_window"],
        "baseline_mismatch_flags": per_flag["baseline mismatch"],
        "baseline_mismatch_denominator": n - per_flag["no panel row"],
        "plans_tracking_as_reported": int(len(tracking)),
        "plans_tracking_as_reported_ids": tracking.plan_id.tolist(),
        "per_label": per_label,
        "per_flag": per_flag,
        "in_span": int(plans.in_span.sum()),
        "ramp_suppressed_in_ranking": int(plans["ramp_suppressed_in_ranking"].sum()) if "ramp_suppressed_in_ranking" in plans else 0,
    }


def at_open_sensitivity(panel: Panel) -> pd.DataFrame:
    """Met target at open under each candidate window. One row per window: met, denominator, no data plans."""
    import numpy as np
    cw = panel.clinic.sort_values(["location_id", "week_start"])
    rows = []
    for window in HEADLINE["at_open_windows_considered"]:
        met = denom = 0
        no_data = []
        for r in panel.plans.itertuples():
            s = cw[cw.location_id == r.location_id].set_index("week_start")[r.target_metric]
            bw = monday_on_or_before(pd.Timestamp(r.opened_date))
            if window == "last_week_on_or_before_open":
                w = s[s.index <= bw].iloc[-1:]
            elif window == "first_week_on_or_after_open":
                w = s[s.index >= bw].iloc[:1]
            elif window == "trailing_4_week_mean":
                w = s[s.index <= bw].iloc[-4:]
            else:
                w = s[s.index <= bw].iloc[-8:]
            if not len(w) or not w.notna().any():
                no_data.append(r.plan_id)
                continue
            denom += 1
            met += int(meets(float(w.mean()), r.target_value, r.target_metric))
        rows.append({"window": window, "met": met, "denominator": denom, "no_data": no_data})
    return pd.DataFrame(rows)


def label_plans(panel: Panel, scores: pd.DataFrame = None, item_threshold: float = RANKED_ITEM_SCORE,
                linked_threshold: float = LINKED_MIN_SCORE) -> pd.DataFrame:
    """Plans in the leader's span, for the ranking context and the digest."""
    return plan_check(panel, scores=scores, in_span_only=True, item_threshold=item_threshold,
                      linked_threshold=linked_threshold)


def plan_context(plans: pd.DataFrame) -> dict:
    """(location_id, metric) to off-track tags for the ranking context term.

    Counts contradicted, and stale plus past due together, on plans that are not achieved or
    mistargeted. A plan that is only past its due date does not move the ranking.
    """
    out = {}
    for r in plans.itertuples():
        if r.label in ("achieved", "mistargeted"):
            continue
        tags = []
        if r.label == "contradicted":
            tags.append(f"{r.plan_id} contradicted")
        if "stale" in r.flags and "past due" in r.flags:
            tags.append(f"{r.plan_id} stale past due")
        if r.status == "at_risk":
            tags.append(f"{r.plan_id} at_risk")
        if tags:
            out.setdefault((r.location_id, r.target_metric), []).extend(tags)
    return out
