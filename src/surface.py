"""First surfaced on: the earliest Monday on which the ranking, run as of that date with only the
data up to it, would have placed each item above the item line.

Long run means, baselines, peer medians and provider windows all come from the truncated panel,
so nothing looks ahead. One caveat that cannot be removed with this data: action plan status is a
single snapshot, so a plan's label as of an earlier Monday uses today's recorded status.
"""
import csv
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.clean import ROOT, Panel
from src import rank
from src.plans import label_plans, plan_context

OUT_PATH = ROOT / "outputs" / "first_surfaced.csv"
MIN_WEEKS_AS_OF = rank.MIN_WEEKS


def panel_as_of(panel: Panel, monday: pd.Timestamp) -> Panel:
    """A copy of the panel that knows nothing after `monday`."""
    clinic = panel.clinic[panel.clinic.week_start <= monday].copy()
    providers = panel.providers[panel.providers.week_start <= monday].copy()
    plans = panel.plans[panel.plans.opened_date <= monday].copy()
    loc = panel.locations.drop(columns=["first_week", "weeks_available"], errors="ignore").copy()
    first = clinic.groupby("location_id").week_start.agg(first_week="min", weeks_available="nunique")
    loc = loc.merge(first, left_on="location_id", right_index=True, how="left")
    loc["weeks_available"] = loc["weeks_available"].fillna(0).astype(int)
    return replace(panel, today=monday + pd.Timedelta(days=7), latest_week=monday, clinic=clinic,
                   providers=providers, plans=plans, locations=loc)


def scores_as_of(panel: Panel, monday: pd.Timestamp) -> pd.DataFrame:
    p = panel_as_of(panel, monday)
    outliers = rank.provider_outliers(p) if len(p.my_providers) else pd.DataFrame()
    plans = label_plans(p) if len(p.my_plans) else pd.DataFrame(columns=["plan_id"])
    ctx = plan_context(plans) if len(plans) else {}
    return rank.score_panel(p, ctx, outliers)


def first_surfaced(panel: Panel, items: list, cache: Path = OUT_PATH, use_cache: bool = True) -> pd.DataFrame:
    """One row per item: first_monday (the start of the current run at or above the item line) and
    mondays_before_latest_week, the count of Mondays from first_monday to the latest complete week.
    2026-02-23 to 2026-04-27 is 9, so the date and the number agree on the page."""
    pairs = [(it["rank"], it["center"], it["lead"]["metric"]) for it in items]
    if use_cache and cache.exists():
        old = pd.read_csv(cache, dtype=str)
        same = ("mondays_before_latest_week" in old.columns and len(old) == len(pairs)
                and (old.get("latest_week", pd.Series(dtype=str)) == str(panel.latest_week.date())).all()
                and all((str(r), c, m) in set(zip(old["item"], old["center"], old["metric"])) for r, c, m in pairs))
        if same:
            old["mondays_before_latest_week"] = old["mondays_before_latest_week"].astype(int)
            return old
    mondays = [pd.Timestamp(m) for m in sorted(panel.clinic.week_start.unique()) if pd.Timestamp(m) < panel.latest_week]
    above = {p: [] for p in pairs}  # per item, one bool per prior Monday
    for monday in mondays:
        sc = scores_as_of(panel, monday)
        for p in pairs:
            rows = sc[(sc.center == p[1]) & (sc.metric == p[2]) & sc.eligible & (sc.score >= rank.ITEM_THRESHOLD)]
            above[p].append(bool(len(rows)))
    rows = []
    for (r, c, m) in pairs:
        flags = above[(r, c, m)]
        ever = next((mondays[i] for i, f in enumerate(flags) if f), None)
        # Start of the unbroken run of Mondays above the line that ends last Monday. This is the episode
        # the reader is looking at; the earliest ever Monday can be an old, separate episode.
        run_start = None
        for i in range(len(flags) - 1, -1, -1):
            if flags[i]:
                run_start = mondays[i]
            else:
                break
        base = {"item": r, "center": c, "metric": m, "latest_week": str(panel.latest_week.date()),
                "first_monday_ever": "this week" if ever is None else str(ever.date()),
                "mondays_above_line_before_latest_week": int(sum(flags))}
        if run_start is None:
            rows.append({**base, "first_monday": "this week", "mondays_before_latest_week": 0})
        else:
            rows.append({**base, "first_monday": str(run_start.date()),
                         "mondays_before_latest_week": int((panel.latest_week - run_start).days // 7)})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False, lineterminator="\n")
    return df


LAST_WEEK_PATH = ROOT / "outputs" / "last_week_items.csv"


def last_week_items(panel: Panel, scores_now: pd.DataFrame, cache: Path = LAST_WEEK_PATH, use_cache: bool = True) -> pd.DataFrame:
    """The five items last week's digest would have carried (ranking as of the Monday before the latest
    complete week), and whether each pair is still above the item line this week. Written to outputs."""
    prior = panel.latest_week - pd.Timedelta(days=7)
    digest_date = str((panel.today - pd.Timedelta(days=7)).date())
    if use_cache and cache.exists():
        old = pd.read_csv(cache, dtype=str)
        if "latest_week" in old.columns and (old["latest_week"] == str(panel.latest_week.date())).all():
            return old
    sc = scores_as_of(panel, prior)
    adverse = sc[sc.eligible & (sc.score >= rank.ITEM_THRESHOLD)].copy()
    for col in ["spike_sd", "drift_sd", "level_sd"]:
        adverse[f"sort_{col}"] = adverse[col].fillna(-99)
    adverse = adverse.sort_values(["score", "sort_spike_sd", "sort_drift_sd", "sort_level_sd", "integrity"], ascending=[False] * 5)
    leads = adverse.drop_duplicates("location_id").head(rank.CAP)
    rows = []
    for n, r in enumerate(leads.itertuples(), start=1):
        now = scores_now[(scores_now.location_id == r.location_id) & (scores_now.metric == r.metric)]
        still = bool(len(now) and now.iloc[0].eligible and now.iloc[0].score >= rank.ITEM_THRESHOLD)
        rows.append({"digest_date": digest_date, "as_of_week": str(prior.date()), "latest_week": str(panel.latest_week.date()),
                     "rank": n, "center": r.center, "metric": r.metric, "metric_label": r.label,
                     "score_then": round(float(r.score), 2), "still_above_line": still})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False, lineterminator="\n")
    return df
