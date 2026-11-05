"""Feedback loop rules and their plain word effect lines.

feedback/feedback.csv is the pipeline's input (digest_date, item_id, verdict, note). In week one,
PostHog feedback events are exported to it by hand; week two automates it. Rules (CLAUDE.md):
- not_useful on a (center, metric) raises its threshold by 25 percent for 4 weeks
- already_handled suppresses that exact item for 2 weeks
- useful lowers the threshold by 10 percent
Every adjustment is written to outputs/decision_log.json.
"""
import csv
import json
from pathlib import Path

import pandas as pd

from src.clean import ROOT

FEEDBACK_CSV = ROOT / "feedback" / "feedback.csv"
DECISION_LOG = ROOT / "outputs" / "decision_log.json"
COLUMNS = ["digest_date", "item_id", "verdict", "note"]
RULES = {
    "useful": {"factor": 0.90, "weeks": 4, "suppress": False},
    "not_useful": {"factor": 1.25, "weeks": 4, "suppress": False},
    "already_handled": {"factor": 1.0, "weeks": 2, "suppress": True},
}
# What the reader sees after a click. Written from the rule, not by a model.
EFFECT_LINES = {
    "useful": "Noted. Next week this item needs to clear a lower line to surface, for four weeks.",
    "not_useful": "Noted. Next week this item needs to clear a higher line to surface, for four weeks.",
    "already_handled": "Noted. This item is hidden for two weeks, then checked again.",
}
VERDICT_LABELS = {"useful": "useful", "not_useful": "not useful", "already_handled": "already handled"}


def ensure_file() -> Path:
    FEEDBACK_CSV.parent.mkdir(exist_ok=True)
    if not FEEDBACK_CSV.exists():
        with open(FEEDBACK_CSV, "w", encoding="utf-8", newline="\n") as f:
            f.write(",".join(COLUMNS) + "\n")
    return FEEDBACK_CSV


def load(path: Path = FEEDBACK_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feedback.csv is missing columns {missing}; expected {COLUMNS}")
    return df


def thresholds(today, item_map: dict, base_threshold: float, path: Path = FEEDBACK_CSV, log_path: Path = DECISION_LOG) -> dict:
    """Per (location_id, metric) threshold adjustments in force on `today`.

    item_map: item_id -> (location_id, metric) from the digest the feedback was given on.
    Returns {(location_id, metric): {"threshold": float, "suppressed_until": date or None, "reasons": [...]}}
    and writes every adjustment to the decision log.
    """
    today = pd.Timestamp(today)
    df = load(path)
    out, log = {}, []
    for r in df.itertuples():
        if r.verdict not in RULES or r.item_id not in item_map:
            log.append({"digest_date": r.digest_date, "item_id": r.item_id, "verdict": r.verdict, "applied": False,
                        "why": "unknown verdict or item"})
            continue
        rule = RULES[r.verdict]
        given = pd.Timestamp(r.digest_date)
        until = given + pd.Timedelta(weeks=rule["weeks"])
        key = item_map[r.item_id]
        # A verdict given on one digest applies to the next `weeks` digests, then stops. Already handled on
        # the 2026-05-04 digest hides the pair on 05-11 and 05-18, and it comes back on 05-25.
        active = given < today <= until
        entry = out.setdefault(key, {"threshold": base_threshold, "suppressed_until": None, "reasons": []})
        if active:
            if rule["suppress"]:
                entry["suppressed_until"] = str(until.date())
            else:
                entry["threshold"] = round(entry["threshold"] * rule["factor"], 3)
            entry["reasons"].append(f"{r.verdict} on {r.digest_date} ({r.item_id})")
        log.append({"digest_date": r.digest_date, "item_id": r.item_id, "verdict": r.verdict, "center_metric": list(key),
                    "applied": bool(active), "until": str(until.date()),
                    "effect": EFFECT_LINES[r.verdict], "note": r.note})
    with open(log_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"today": str(today.date()), "base_threshold": base_threshold, "decisions": log}, f, indent=1)
    return out
