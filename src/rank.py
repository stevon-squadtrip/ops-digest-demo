"""Ranking and suppression. Decides what earns the leader's attention this week.

Rules are in CLAUDE.md "Ranking rules" and its approved additions. Every (center, metric)
that is considered and not shown is written to outputs/suppressed.json with the reason.
Code computes every number here. Nothing in this file writes a sentence for the digest.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import ROOT, Panel
from src.metrics import METRICS, MIN_APPTS, MIN_CSAT_RESPONSES
from src.plans import label_plans, plan_check, plan_context, plan_summary

OUT_DIR = ROOT / "outputs"
SOURCE = "data/raw/clinic_weekly.csv"

ITEM_THRESHOLD = 3.0
SUPPORT_THRESHOLD = 2.0
OTHER_MOVEMENT_SD = 1.5
OTHER_MOVEMENTS_CAP = 4
# Ramp rule extension: at a center with fewer than RAMP_WEEKS of history, an adverse signal whose
# run is under this many weeks goes to Watching, never to an item. Any metric.
NEW_CENTER_MIN_RUN = 2
FAVORABLE_CONTEXT_SD = 1.0
CAP = 5
SIZE_CAP = 4.0
MIN_WEEKS = 12
MIN_OWN_SD_WEEKS = 10
POOLED_SD_CONFIDENCE = 0.8
PERSIST_FLOOR_SD = 0.5
PERSIST_MAX_WEEKS = 10
INTEGRITY_LEVEL = 99.5
INTEGRITY_WEEKS = 6
RAMP_WEEKS = 26
RAMP_METRICS = {"appts_per_doctor_hour", "revenue_per_appt", "appts_no_show", "membership_conversion_pct"}
QUALITY_METRICS = {"recheck_compliance_pct", "record_completion_24h_pct", "callback_compliance_pct"}
PROVIDER_GAP_POINTS = 15.0
PROVIDER_MIN_WEEKS = 4
PROVIDER_MIN_HOURS = 100.0
PROVIDER_WINDOW = 8
CONTEXT_METRICS = {"doctor_hours_scheduled": "Doctor hours scheduled", "appts_completed": "Appointments completed"}
CONTEXT_WINDOW = 12
# Fitted change over 12 weeks, as a share of the 12 week mean. Clinic 06's doctor hours slide
# is 8.5 fitted and 17 raw, which is the kind of move a leader should see next to a staffing item.
CONTEXT_MOVE_PCT = 8.0
OPP_HELD_WEEKS = 4
OPP_WINDOW = 6
# Opportunity candidates: quality and client metrics only, at a center that sits at or above the
# median of the leader's other centers on that metric. We copy from centers that are good, not just improving.
OPPORTUNITY_METRICS = {"recheck_compliance_pct", "record_completion_24h_pct", "callback_compliance_pct",
                       "avg_wait_time_min", "client_csat"}
# Raw first-to-last change over the context window, as a share of the first value. Either test can trigger.
CONTEXT_RAW_MOVE_PCT = 10.0


def adverse_sign(metric: str) -> int:
    """Multiply a change by this so that adverse movement is positive."""
    return -1 if METRICS[metric]["better"] == "higher" else 1


def _fmt_week(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def evidence(center: str, metric: str, series: pd.Series, weeks: int = 8, columns=None) -> dict:
    tail = series.iloc[-weeks:]
    return {
        "metric": metric,
        "center": center,
        "weeks": [_fmt_week(w) for w in tail.index],
        "values": [None if pd.isna(v) else round(float(v), 2) for v in tail.values],
        "source": SOURCE,
        "columns": columns or ["location_id", "week_start", metric],
    }


def provider_outliers(panel: Panel) -> pd.DataFrame:
    """Doctors whose 8 week hours-weighted quality rate sits far below their center's other doctors."""
    pw = panel.my_providers
    names = panel.locations.set_index("location_id").location_name
    last = sorted(pw.week_start.unique())[-PROVIDER_WINDOW:]
    p8 = pw[pw.week_start.isin(last)]
    rows = []
    for lid, g in p8.groupby("location_id"):
        for metric in ["recheck_compliance_pct", "record_completion_24h_pct"]:
            for pid, gp in g.groupby("provider_id"):
                weeks = gp.week_start.nunique()
                hours = gp.scheduled_hours.sum()
                others = g[g.provider_id != pid]
                if weeks < PROVIDER_MIN_WEEKS or hours < PROVIDER_MIN_HOURS or others.scheduled_hours.sum() == 0:
                    continue
                mine_w = (gp[metric] * gp.scheduled_hours).sum() / hours
                peers_w = (others[metric] * others.scheduled_hours).sum() / others.scheduled_hours.sum()
                peer_rates = others.groupby("provider_id").apply(
                    lambda x: (x[metric] * x.scheduled_hours).sum() / x.scheduled_hours.sum(), include_groups=False
                )
                rows.append({
                    "location_id": lid, "center": names[lid], "metric": metric, "provider_id": pid,
                    "employment_type": gp.employment_type.iloc[-1], "tenure_months": int(gp.tenure_months.max()),
                    "weeks": int(weeks), "hours": round(float(hours), 1),
                    "rate_hours_weighted": round(float(mine_w), 1), "rate_simple_mean": round(float(gp[metric].mean()), 1),
                    "peers_hours_weighted": round(float(peers_w), 1),
                    "peers_min": round(float(peer_rates.min()), 1), "peers_max": round(float(peer_rates.max()), 1),
                    "gap_points": round(float(peers_w - mine_w), 1),
                    "outlier": bool(peers_w - mine_w >= PROVIDER_GAP_POINTS),
                })
    return pd.DataFrame(rows)


def score_panel(panel: Panel, plan_ctx: dict, outliers: pd.DataFrame, thresholds: dict = None) -> pd.DataFrame:
    """One row per (center, metric) with every component, the score, and any suppression reason."""
    thresholds = thresholds or {}
    cw = panel.my_clinic.sort_values(["location_id", "week_start"])
    names = panel.locations.set_index("location_id").location_name
    weeks_available = panel.locations.set_index("location_id").weeks_available
    provider_hits = {
        (r.location_id, r.metric): f"{r.provider_id} {r.rate_hours_weighted} vs peers {r.peers_hours_weighted}"
        for r in outliers.itertuples() if r.outlier
    } if len(outliers) else {}

    # Pooled sd per metric, for centers with too little history for their own.
    pooled_sd = {}
    for metric in METRICS:
        parts = [g[metric].iloc[:-8] for _, g in cw.groupby("location_id")]
        pooled_sd[metric] = pd.concat(parts).std()

    rows = []
    for lid, g in cw.groupby("location_id"):
        g = g.set_index("week_start")
        center = names[lid]
        n_weeks = int(weeks_available[lid])
        for metric, spec in METRICS.items():
            s = g[metric]
            sign = adverse_sign(metric)
            latest = s.iloc[-1]
            trailing8 = s.iloc[-8:]
            base8 = trailing8.mean()
            history = s.iloc[:-8]
            long_mean = s.iloc[:-10].mean()
            own_sd = history.std() if history.notna().sum() >= MIN_OWN_SD_WEEKS else np.nan
            sd_source = "own" if pd.notna(own_sd) else "pooled"
            sd = own_sd if sd_source == "own" else pooled_sd[metric]
            if not sd or pd.isna(sd) or sd == 0:
                sd = np.nan
            denom = float(g[spec["denom"]].iloc[-1]) if spec["denom"] else np.nan
            denom_4wk = float(g[spec["denom"]].iloc[-4:].sum()) if spec["denom"] else np.nan
            # The watch rule's 4 week check: the denominator-weighted value over the trailing 4 weeks.
            # For client_csat this is the response-weighted CSAT (csat_4wk_weighted), for every center.
            if spec["denom"]:
                last4 = g[[metric, spec["denom"]]].iloc[-4:].dropna()
                wsum = last4[spec["denom"]].sum()
                value_4wk_weighted = float((last4[metric] * last4[spec["denom"]]).sum() / wsum) if wsum else np.nan
            else:
                value_4wk_weighted = np.nan

            spike = (latest - base8) / sd * sign if pd.notna(sd) else np.nan
            drift = (base8 - long_mean) / sd * sign if pd.notna(sd) else np.nan
            run_uncapped = 0
            if pd.notna(sd) and pd.notna(long_mean):
                for v in s.iloc[::-1]:
                    if pd.notna(v) and (v - long_mean) * sign >= PERSIST_FLOOR_SD * sd:
                        run_uncapped += 1
                    else:
                        break
            run = min(run_uncapped, PERSIST_MAX_WEEKS)
            run_start_week = _fmt_week(s.index[-run_uncapped]) if run_uncapped else ""
            weeks_at_ceiling = int((s.iloc[-10:] >= INTEGRITY_LEVEL).sum()) if metric in QUALITY_METRICS else 0
            integrity = weeks_at_ceiling >= INTEGRITY_WEEKS
            # Informational only (CLAUDE.md Reporting rules). The spike term anchors on last week and the
            # integrity rule on a 10 week count. These columns show the trailing 4 week view. Not scored.
            prior8_before_last4 = s.iloc[-12:-4]
            spike_4wk = ((s.iloc[-4:].mean() - prior8_before_last4.mean()) / sd * sign
                         if pd.notna(sd) and prior8_before_last4.notna().sum() >= 4 else np.nan)
            ceiling_last4 = int((s.iloc[-4:] >= INTEGRITY_LEVEL).sum()) if metric in QUALITY_METRICS else 0
            last10 = s.iloc[-10:]
            confidence = float(last10.notna().mean()) * (POOLED_SD_CONFIDENCE if sd_source == "pooled" else 1.0)
            plan_tags = plan_ctx.get((lid, metric), [])
            provider_tag = provider_hits.get((lid, metric), "")
            context = (2.0 if plan_tags else 0.0) + (2.0 if provider_tag else 0.0)
            fb = thresholds.get((lid, metric), {})
            row_threshold = fb.get("threshold", ITEM_THRESHOLD) if isinstance(fb, dict) else float(fb)
            fb_suppressed = isinstance(fb, dict) and fb.get("suppressed_until") and str(panel.today.date()) <= fb["suppressed_until"]
            fb_reasons = "; ".join(fb.get("reasons", [])) if isinstance(fb, dict) else ""

            # Suppression. Hides the metric from the ranking only.
            suppressed = ""
            if spec.get("suppressed"):
                suppressed = f"definition change: {spec['suppressed']}"
            elif n_weeks < MIN_WEEKS:
                suppressed = f"insufficient history: {n_weeks} weeks, need {MIN_WEEKS}"
            elif pd.isna(latest):
                suppressed = "no value this week"
            elif metric == "client_csat" and denom < MIN_CSAT_RESPONSES:
                suppressed = (f"watch: {int(denom)} responses this week, {int(denom_4wk)} over the trailing 4 weeks, "
                              f"need {MIN_CSAT_RESPONSES} a week")
            elif spec["denom"] == "appts_completed" and denom < MIN_APPTS:
                suppressed = (f"watch: {int(denom)} appointments this week, {int(denom_4wk)} over the trailing 4 weeks, "
                              f"need {MIN_APPTS} a week")
            elif metric in RAMP_METRICS and n_weeks < RAMP_WEEKS:
                suppressed = f"ramp: {n_weeks} weeks of history, need {RAMP_WEEKS} for volume and commercial metrics"
            elif (n_weeks < RAMP_WEEKS and run_uncapped < NEW_CENTER_MIN_RUN and pd.notna(spike) and spike > 0
                  and max(spike, drift if pd.notna(drift) else 0) >= 1.0):
                suppressed = (f"watching: spike only at a new center, run {run_uncapped} week{'s' if run_uncapped != 1 else ''}, "
                              f"{n_weeks} weeks of history, need a run of {NEW_CENTER_MIN_RUN} or {RAMP_WEEKS} weeks of history")
            elif fb_suppressed:
                suppressed = f"feedback: already handled, hidden until {fb['suppressed_until']} ({'; '.join(fb.get('reasons', []))})"
            # Robustness of the watch rule: does the 4 week average denominator agree with last week?
            if spec["denom"] and pd.notna(denom):
                floor = MIN_CSAT_RESPONSES if metric == "client_csat" else MIN_APPTS
                watch_robust = bool((denom < floor) == (denom_4wk / 4 < floor))
            else:
                watch_robust = True

            rows.append({
                "location_id": lid, "center": center, "metric": metric, "label": spec["label"],
                "lane": spec["lane"], "weight": spec["weight"], "weeks_available": n_weeks,
                "latest": latest, "avg8": base8, "long_mean": long_mean, "sd": sd, "sd_source": sd_source,
                "denominator": denom, "denominator_4wk": denom_4wk, "watch_robust": watch_robust,
                "value_4wk_weighted": value_4wk_weighted,
                "csat_4wk_weighted": value_4wk_weighted if metric == "client_csat" else np.nan,
                "spike_sd": spike, "spike_4wk_sd": spike_4wk, "drift_sd": drift, "run_weeks": run,
                "run_weeks_uncapped": run_uncapped, "run_start_week": run_start_week,
                "weeks_at_ceiling": weeks_at_ceiling, "ceiling_weeks_last4": ceiling_last4, "integrity": integrity,
                "plan_context": "; ".join(plan_tags), "provider_context": provider_tag,
                "context": context, "confidence": confidence, "suppressed": suppressed,
                "threshold": row_threshold, "threshold_reasons": fb_reasons,
            })
    df = pd.DataFrame(rows)

    # Level: trailing 8 week average vs the median of the leader's other centers, in peer sd.
    df["level_sd"] = np.nan
    for metric, g in df.groupby("metric"):
        sign = adverse_sign(metric)
        for i, r in g.iterrows():
            others = g.loc[g.index != i, "avg8"]
            psd = others.std()
            if pd.notna(psd) and psd > 0:
                df.loc[i, "level_sd"] = (r.avg8 - others.median()) / psd * sign

    # Movement is spike or drift. Level is a standing gap to peers: it counts in the score,
    # never in the direction, so an improving center with a long way to go still reads as improving.
    movement = df[["spike_sd", "drift_sd"]].clip(lower=0).max(axis=1).fillna(0)
    fav = df[["spike_sd", "drift_sd"]].clip(upper=0).min(axis=1).abs().fillna(0).clip(upper=SIZE_CAP)
    df["movement_sd"] = movement.clip(upper=SIZE_CAP)
    df["size"] = pd.concat([movement, df.level_sd.clip(lower=0).fillna(0)], axis=1).max(axis=1).clip(upper=SIZE_CAP)
    df.loc[df.integrity, "size"] = SIZE_CAP
    df["persistence"] = df.run_weeks / PERSIST_MAX_WEEKS * 4.0
    df.loc[df.integrity, "persistence"] = 0.0
    df["score"] = df.weight * df.confidence * (df["size"] + df.persistence + df.context)
    df["favorable_size"] = fav
    df["favorable_score"] = df.weight * df.confidence * fav
    df["direction"] = np.where(
        (movement > 0) & (movement >= fav), "adverse", np.where(fav > 0, "favorable", "flat")
    )
    df.loc[df.integrity, "direction"] = "adverse"

    # Watch rows: say what the 4 week weighted value would score, so the log line carries the evidence.
    # Thresholds are never bent toward inclusion (CLAUDE.md Reporting rules); this is a report only.
    df["watch_4wk_score"] = np.nan
    for i, r in df[df.suppressed.str.startswith("watch") & df.value_4wk_weighted.notna()].iterrows():
        sign = adverse_sign(r.metric)
        others = df[(df.metric == r.metric) & (df.index != i)].avg8
        spike_w = (r.value_4wk_weighted - r.avg8) / r.sd * sign if pd.notna(r.sd) else np.nan
        level_w = (r.value_4wk_weighted - others.median()) / others.std() * sign if others.std() else np.nan
        size_w = min(max(spike_w if pd.notna(spike_w) else 0, r.drift_sd if pd.notna(r.drift_sd) else 0,
                         level_w if pd.notna(level_w) else 0, 0), SIZE_CAP)
        score_w = r.weight * r.confidence * (size_w + r.persistence + r.context)
        df.loc[i, "watch_4wk_score"] = round(score_w, 3)
        verdict = "clears the supporting line" if score_w >= SUPPORT_THRESHOLD else "under the alert line, watching"
        unit = "CSAT" if r.metric == "client_csat" else r.label
        df.loc[i, "suppressed"] = (
            f"{r.suppressed}. 4 week weighted {unit} {r.value_4wk_weighted:.2f} on {int(r.denominator_4wk)} "
            f"{'responses' if r.metric == 'client_csat' else 'appointments'}, long run {r.long_mean:.2f}, "
            f"would score {score_w:.2f}: {verdict}"
        )
    # Feedback raised the line and the score no longer clears it: the pair goes to the hidden list with the
    # verdict dates in the reason, and it never drops into other movements through the back door.
    raised = df.suppressed.eq("") & (df.threshold > ITEM_THRESHOLD) & (df.score < df.threshold)
    for i, r in df[raised].iterrows():
        df.loc[i, "suppressed"] = (f"feedback: line raised to {r.threshold:.2f} from {ITEM_THRESHOLD:.1f} ({r.threshold_reasons}), "
                                   f"this line scored {r.score:.1f}")
    df["eligible"] = df.suppressed.eq("")
    for col in ["latest", "avg8", "long_mean", "sd", "spike_sd", "spike_4wk_sd", "drift_sd", "level_sd", "movement_sd", "size",
                "value_4wk_weighted", "csat_4wk_weighted",
                "persistence", "score", "favorable_size", "favorable_score", "confidence"]:
        df[col] = df[col].astype(float).round(3)
    return df


def _held_favorable(series: pd.Series, metric: str) -> int:
    """Weeks in the last 6 on the favorable side of the center's long run mean."""
    long_mean = series.iloc[:-10].mean()
    sign = adverse_sign(metric)
    last = series.iloc[-OPP_WINDOW:]
    return int(((last - long_mean) * sign < 0).sum())


def context_trends(panel: Panel, lid: str) -> list:
    """Unscored context: doctor hours and appointments, shown when they moved over 12 weeks."""
    g = panel.my_clinic[panel.my_clinic.location_id == lid].sort_values("week_start").set_index("week_start")
    out = []
    for metric, label in CONTEXT_METRICS.items():
        s = g[metric].iloc[-CONTEXT_WINDOW:]
        if len(s) < 4 or s.mean() == 0:
            continue
        x = np.arange(len(s))
        slope = np.polyfit(x, s.values, 1)[0]
        fitted_change = slope * (len(s) - 1)
        pct = fitted_change / s.mean() * 100
        raw_pct = (s.iloc[-1] - s.iloc[0]) / s.iloc[0] * 100 if s.iloc[0] else 0.0
        if abs(pct) >= CONTEXT_MOVE_PCT or abs(raw_pct) >= CONTEXT_RAW_MOVE_PCT:
            out.append({
                "metric": metric, "label": label, "weeks": int(len(s)),
                "start_week": _fmt_week(s.index[0]), "start": round(float(s.iloc[0]), 1),
                "latest_week": _fmt_week(s.index[-1]), "latest": round(float(s.iloc[-1]), 1),
                "fitted_change_pct": round(float(pct), 1), "raw_change_pct": round(float(raw_pct), 1),
                "direction": "down" if (pct + raw_pct) < 0 else "up",
                "evidence": evidence(panel.locations.set_index("location_id").location_name[lid], metric, g[metric], CONTEXT_WINDOW),
            })
    return out


def _row_dict(r: pd.Series) -> dict:
    d = r.to_dict()
    for k, v in list(d.items()):
        if isinstance(v, (np.floating, float)) and pd.isna(v):
            d[k] = None
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            d[k] = bool(v)
    return d


def build_items(panel: Panel, scores: pd.DataFrame, plans: pd.DataFrame, outliers: pd.DataFrame) -> dict:
    """Top five items, one per center, plus the opportunity and other movements."""
    cw = panel.my_clinic.sort_values(["location_id", "week_start"])
    series = {(lid, m): g.set_index("week_start")[m] for lid, g in cw.groupby("location_id") for m in METRICS}
    eligible = scores[scores.eligible]
    # Score is only ever built from adverse size, persistence and context, so no direction filter here.
    # A center that is improving but still far below peers with a contradicted plan earns an item.
    # The line is ITEM_THRESHOLD unless feedback moved it for that center and metric (threshold column).
    adverse = eligible[eligible.score >= eligible.threshold].copy()
    adverse["sort_spike"] = adverse.spike_sd.fillna(-99)
    adverse["sort_drift"] = adverse.drift_sd.fillna(-99)
    adverse["sort_level"] = adverse.level_sd.fillna(-99)
    adverse = adverse.sort_values(
        ["score", "sort_spike", "sort_drift", "sort_level", "integrity"], ascending=[False] * 5
    )
    leads = adverse.drop_duplicates("location_id")
    ranked_centers = leads.location_id.tolist()
    top_ids = ranked_centers[:CAP]

    items = []
    for rank, lid in enumerate(top_ids, start=1):
        lead = leads[leads.location_id == lid].iloc[0]
        center = lead.center
        at_center = eligible[eligible.location_id == lid]
        supporting = at_center[(at_center.metric != lead.metric) & (at_center.score >= SUPPORT_THRESHOLD)]
        supporting = supporting.sort_values("score", ascending=False)
        center_plans = plans[plans.location_id == lid]
        plan_metrics = set(center_plans.target_metric)
        favorable = at_center[
            (at_center.metric != lead.metric) & (at_center.direction == "favorable")
            & ((at_center.favorable_size >= FAVORABLE_CONTEXT_SD) | at_center.metric.isin(plan_metrics))
        ].sort_values("favorable_score", ascending=False)
        center_outliers = outliers[(outliers.location_id == lid) & outliers.outlier] if len(outliers) else outliers
        items.append({
            "rank": rank, "center": center, "location_id": lid, "lane": lead.lane,
            "score": float(lead.score), "integrity": bool(lead.integrity),
            "lead": _row_dict(lead), "lead_evidence": evidence(center, lead.metric, series[(lid, lead.metric)]),
            "supporting": [
                {**_row_dict(r), "evidence": evidence(center, r.metric, series[(lid, r.metric)])}
                for _, r in supporting.iterrows()
            ],
            "favorable_context": [
                {**_row_dict(r), "evidence": evidence(center, r.metric, series[(lid, r.metric)])}
                for _, r in favorable.iterrows()
            ],
            "unscored_context": context_trends(panel, lid),
            "plans": center_plans.to_dict("records"),
            "providers": center_outliers.to_dict("records") if len(center_outliers) else [],
        })

    # Metrics where another center needs help: a ranked adverse line (not integrity) or an off-track plan.
    need = {}
    for it in items:
        if not it["integrity"]:
            need.setdefault(it["lead"]["metric"], []).append((it["center"], f"item {it['rank']}"))
        for s in it["supporting"]:
            if not s["integrity"]:
                need.setdefault(s["metric"], []).append((it["center"], f"supporting line in item {it['rank']}"))
    for r in plans.itertuples():
        off_track = r.label in ("contradicted", "stalled") or r.status == "at_risk" \
            or ("stale" in r.flags and "past due" in r.flags)
        if off_track:
            need.setdefault(r.target_metric, []).append(
                (r.location_name, f"{r.plan_id} {r.status}, {r.label}" + (f" ({', '.join(r.flags)})" if r.flags else ""))
            )

    not_ranked = eligible[
        ~eligible.location_id.isin(top_ids) & (eligible.direction == "favorable") & ~eligible.integrity
        & eligible.metric.isin(OPPORTUNITY_METRICS)
    ]
    candidates = []
    for _, r in not_ranked.iterrows():
        held = _held_favorable(series[(r.location_id, r.metric)], r.metric)
        needs = [(c, why) for c, why in need.get(r.metric, []) if c != r.center]
        candidates.append((r, held, needs))
    good_center = lambda r: pd.notna(r.level_sd) and r.level_sd <= 0  # at or above the peer median
    qualified = [(r, held, needs) for r, held, needs in candidates
                 if held >= OPP_HELD_WEEKS and needs and good_center(r)]
    opportunity = None
    if qualified:
        r, held, needs = sorted(qualified, key=lambda t: (-t[0].favorable_score, -(t[0].denominator or 0)))[0]
        opportunity = {
            "rule": "held and needed", **_row_dict(r), "held_weeks_of_6": held,
            "needed_by": [{"center": c, "why": w} for c, w in needs],
            "evidence": evidence(r.center, r.metric, series[(r.location_id, r.metric)]),
        }
    elif candidates:
        r, held, needs = sorted(candidates, key=lambda t: (-t[0].favorable_score, -(t[0].denominator or 0)))[0]
        opportunity = {
            "rule": "fallback: largest favorable score, ties by denominator", **_row_dict(r),
            "held_weeks_of_6": held, "needed_by": [],
            "evidence": evidence(r.center, r.metric, series[(r.location_id, r.metric)]),
        }

    # Other movements: centers outside the top five, scoring at or above the supporting line in
    # either direction, strongest first, capped. A 0.01 move never reaches the reader.
    others = eligible[~eligible.location_id.isin(top_ids)].copy()
    others["movement_any"] = others[["score", "favorable_score"]].max(axis=1)
    other_movements = others[others.movement_any >= SUPPORT_THRESHOLD].sort_values("movement_any", ascending=False)
    other_movements = [
        {**_row_dict(r), "evidence": evidence(r.center, r.metric, series[(r.location_id, r.metric)])}
        for _, r in other_movements.iterrows()
        if not (opportunity and r.location_id == opportunity["location_id"] and r.metric == opportunity["metric"])
    ][:OTHER_MOVEMENTS_CAP]
    opportunity_candidates = [
        {"center": r.center, "metric": r.metric, "latest": float(r.latest), "avg8": float(r.avg8),
         "favorable_score": float(r.favorable_score), "level_sd": None if pd.isna(r.level_sd) else float(r.level_sd),
         "at_or_above_peer_median": bool(good_center(r)), "held_weeks_of_6": held,
         "needed_by": sorted({c for c, _ in needs}), "qualified": bool(held >= OPP_HELD_WEEKS and needs and good_center(r))}
        for r, held, needs in sorted(candidates, key=lambda t: -t[0].favorable_score)
    ]
    return {
        "items": items, "opportunity": opportunity, "opportunity_candidates": opportunity_candidates,
        "other_movements": other_movements, "ranked_centers": ranked_centers,
    }


def why_not(panel: Panel, scores: pd.DataFrame, result: dict) -> list:
    """One plain line per center on why it did or did not make the top five."""
    ranked = {it["location_id"]: it for it in result["items"]}
    beyond_cap = result["ranked_centers"][CAP:]
    lines = []
    for lid in panel.my_ids:
        center = panel.locations.set_index("location_id").location_name[lid]
        rows = scores[scores.location_id == lid]
        if lid in ranked:
            it = ranked[lid]
            lines.append({"center": center, "in_top_five": True,
                          "why": f"Item {it['rank']}: {it['lead']['label']} scored {it['score']:.1f}, above the {ITEM_THRESHOLD} line."})
            continue
        el = rows[rows.eligible]
        if lid in beyond_cap:
            best = el.sort_values("score", ascending=False).iloc[0]
            lines.append({"center": center, "in_top_five": False,
                          "why": f"Ranked {CAP + beyond_cap.index(lid) + 1} on {best.label} at {best.score:.1f}. The cap is {CAP} items."})
            continue
        if el.empty:
            lines.append({"center": center, "in_top_five": False, "why": "Every metric is suppressed this week."})
            continue
        best = el.sort_values("score", ascending=False).iloc[0]
        suppressed_n = int((~rows.eligible).sum())
        extra = f" {suppressed_n} metric{'s' if suppressed_n != 1 else ''} suppressed." if suppressed_n else ""
        lines.append({"center": center, "in_top_five": False,
                      "why": f"Best adverse movement is {best.label} at {best.score:.1f}, below the {ITEM_THRESHOLD} line.{extra}"})
    return lines


def _json_dump(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, default=str)


def write_outputs(panel: Panel, scores: pd.DataFrame, result: dict, why: list, plans: pd.DataFrame,
                  outliers: pd.DataFrame, out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(exist_ok=True)
    paths = {}
    ranked = scores.sort_values(["eligible", "score"], ascending=[False, False])
    paths["ranked"] = out_dir / "ranked.csv"
    ranked.to_csv(paths["ranked"], index=False, lineterminator="\n")
    suppressed = [
        {"center": r.center, "metric": r.metric, "reason": r.suppressed, "latest": r.latest,
         "denominator": None if pd.isna(r.denominator) else r.denominator,
         "denominator_4wk": None if pd.isna(r.denominator_4wk) else r.denominator_4wk,
         "value_4wk_weighted": None if pd.isna(r.value_4wk_weighted) else r.value_4wk_weighted,
         "watch_4wk_score": None if pd.isna(r.watch_4wk_score) else r.watch_4wk_score,
         "watch_robust": bool(r.watch_robust), "weeks_available": int(r.weeks_available)}
        for r in scores[~scores.eligible].itertuples()
    ]
    shown = {(it["location_id"], it["lead"]["metric"]) for it in result["items"]}
    for it in result["items"]:
        shown |= {(it["location_id"], s["metric"]) for s in it["supporting"]}
        shown |= {(it["location_id"], s["metric"]) for s in it["favorable_context"]}
    if result["opportunity"]:
        shown.add((result["opportunity"]["location_id"], result["opportunity"]["metric"]))
    shown |= {(m["location_id"], m["metric"]) for m in result["other_movements"]}
    top_ids = {it["location_id"] for it in result["items"]}
    for r in scores[scores.eligible].itertuples():
        if (r.location_id, r.metric) in shown:
            continue
        if r.location_id in top_ids:
            reason = f"bundled: {r.center} already has an item; this line scored {r.score:.1f} (adverse) or {r.favorable_score:.1f} (favorable), below the lines for a supporting or context line"
        elif r.location_id in result["ranked_centers"] and r.score >= ITEM_THRESHOLD:
            reason = f"cap: {r.center} ranked beyond the top {CAP} centers, this line scored {r.score:.1f}"
        elif max(r.score, r.favorable_score) >= SUPPORT_THRESHOLD:
            reason = f"other movements cap: {OTHER_MOVEMENTS_CAP} lines shown, this one scored {max(r.score, r.favorable_score):.1f} ({r.direction})"
        elif r.direction == "favorable":
            reason = f"favorable: {r.favorable_size:.1f} sd in the good direction, not the opportunity pick"
        else:
            reason = f"below threshold: score {r.score:.1f}, item line {ITEM_THRESHOLD}, other movements need {OTHER_MOVEMENT_SD} sd"
        suppressed.append({"center": r.center, "metric": r.metric, "reason": reason, "latest": r.latest,
                           "denominator": None if pd.isna(r.denominator) else r.denominator,
                           "weeks_available": int(r.weeks_available)})
    paths["suppressed"] = out_dir / "suppressed.json"
    _json_dump(paths["suppressed"], {"latest_week": str(panel.latest_week.date()), "count": len(suppressed), "suppressed": suppressed})
    paths["why_not"] = out_dir / "why_not.json"
    _json_dump(paths["why_not"], why)
    paths["items"] = out_dir / "items.json"
    _json_dump(paths["items"], {"latest_week": str(panel.latest_week.date()), "leader": panel.leader, **result,
                                 "plans": plans.to_dict("records"), "provider_outliers": outliers.to_dict("records")})
    return paths


def feedback_thresholds(panel: Panel, feedback_path=None) -> dict:
    """Milestone 5e wiring: feedback/feedback.csv adjusts the line per (center, metric). Item ids are mapped
    through the last published items.json. Every decision goes to outputs/decision_log.json."""
    from src import feedback
    items_path = OUT_DIR / "items.json"
    item_map = {}
    if items_path.exists():
        prev = json.loads(items_path.read_text(encoding="utf-8"))
        for it in prev.get("items", []):
            item_map[f"item{it['rank']}"] = (it["location_id"], it["lead"]["metric"])
        if prev.get("opportunity"):
            item_map["opportunity"] = (prev["opportunity"]["location_id"], prev["opportunity"]["metric"])
    path = feedback_path or feedback.FEEDBACK_CSV
    return feedback.thresholds(panel.today, item_map, ITEM_THRESHOLD, path=path)


def run(panel: Panel, thresholds: dict = None, write: bool = True, feedback_path=None) -> dict:
    # Two passes. Plan labels feed the ranking context; ranked items feed the side-effect label.
    if thresholds is None:
        thresholds = feedback_thresholds(panel, feedback_path)
    first_pass = label_plans(panel)
    outliers = provider_outliers(panel)
    scores = score_panel(panel, plan_context(first_pass), outliers, thresholds)
    plans = label_plans(panel, scores=scores, item_threshold=ITEM_THRESHOLD, linked_threshold=SUPPORT_THRESHOLD)
    all_plans = plan_check(panel, scores=scores, in_span_only=False, item_threshold=ITEM_THRESHOLD,
                           linked_threshold=SUPPORT_THRESHOLD)
    ramp = {(r.location_id, r.metric) for r in scores.itertuples() if r.suppressed.startswith("ramp")}
    for df in (plans, all_plans):
        df["ramp_suppressed_in_ranking"] = [(r.location_id, r.target_metric) in ramp for r in df.itertuples()]
    result = build_items(panel, scores, plans, outliers)
    why = why_not(panel, scores, result)
    paths = write_outputs(panel, scores, result, why, plans, outliers) if write else {}
    summary = plan_summary(all_plans)
    if write:
        paths["plan_check"] = OUT_DIR / "plan_check.csv"
        out = all_plans.copy()
        out["flags"] = out["flags"].apply("; ".join)
        out["linked_metrics"] = out["linked_metrics"].apply("; ".join)
        out.to_csv(paths["plan_check"], index=False, lineterminator="\n")
        # Summary block at the end, every line starting with '#', so pd.read_csv(..., comment="#") still works.
        lines = [
            "", f"# summary: {summary['plans']} plans, latest week {panel.latest_week.date()}",
            f"# mistargeted: {summary['mistargeted']} of {summary['plans']}",
            f"# baseline mismatch flags: {summary['baseline_mismatch_flags']} of {summary['plans']}",
            "# per label: " + ", ".join(f"{k} {v}" for k, v in summary["per_label"].items()),
            "# per flag: " + ", ".join(f"{k} {v}" for k, v in summary["per_flag"].items()),
            f"# in {panel.leader}'s span: {summary['in_span']}; ramp suppressed in the ranking: {summary['ramp_suppressed_in_ranking']}",
        ]
        with open(paths["plan_check"], "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        paths["plan_summary"] = OUT_DIR / "plan_check_summary.json"
        _json_dump(paths["plan_summary"], {"latest_week": str(panel.latest_week.date()), **summary})
    return {"scores": scores, "plans": plans, "all_plans": all_plans, "plan_summary": summary,
            "outliers": outliers, "why_not": why, "paths": paths, **result}
