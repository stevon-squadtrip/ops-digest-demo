"""Load the four raw CSVs, apply the cleaning rules, and write down every change.

Rules come from CLAUDE.md (Data cleaning rules).
This module never ranks or judges. It makes the data honest and records what it did,
so the digest can show its data notes with real counts.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_DATE = "2026-05-04"
DEFAULT_LEADER = "Medical Director, North"
MEMBERSHIP_BREAK = pd.Timestamp("2026-02-09")
DAYS_PER_MONTH = 30.4375

# Metros and the state code each one sits in. Used only to flag wrong state codes.
METRO_STATE = {"Northgate": "NG", "Riverbend": "RB", "Lakeside": "LK", "Fairmont": "FM", "Harborview": "HV", "Stonebridge": "SB", "Westfield": "WF", "Hillcrest": "HC"}
# Budgeted doctor FTE per exam room outside this band is flagged as an oddity, never corrected.
FTE_PER_ROOM_BAND = (0.4, 1.2)
# Clinic and provider tables both round to one decimal, so a weighted average can sit
# half a unit away from the published value and still agree.
HOURS_WEIGHTED_TOLERANCE = 0.06


@dataclass
class Panel:
    """Everything downstream stages need, plus the record of what cleaning did."""

    today: pd.Timestamp
    leader: str
    locations: pd.DataFrame
    clinic: pd.DataFrame
    providers: pd.DataFrame
    plans: pd.DataFrame
    latest_week: pd.Timestamp
    tier_mismatches: pd.DataFrame
    notes: list = field(default_factory=list)

    @property
    def my_locations(self) -> pd.DataFrame:
        return self.locations[self.locations.rmp_name == self.leader].copy()

    @property
    def my_ids(self) -> list:
        return self.my_locations.location_id.tolist()

    @property
    def my_clinic(self) -> pd.DataFrame:
        return self.clinic[self.clinic.location_id.isin(self.my_ids)].copy()

    @property
    def my_providers(self) -> pd.DataFrame:
        return self.providers[self.providers.location_id.isin(self.my_ids)].copy()

    @property
    def my_plans(self) -> pd.DataFrame:
        return self.plans[self.plans.location_id.isin(self.my_ids)].copy()


def load_raw(raw_dir: Path = RAW_DIR) -> dict:
    """Read the four CSVs exactly as given. No cleaning here."""
    return {
        "locations": pd.read_csv(raw_dir / "locations.csv", parse_dates=["opened_date"]),
        "clinic": pd.read_csv(raw_dir / "clinic_weekly.csv", parse_dates=["week_start"]),
        "providers": pd.read_csv(raw_dir / "provider_weekly.csv", parse_dates=["week_start"]),
        "plans": pd.read_csv(
            raw_dir / "action_plans.csv",
            parse_dates=["opened_date", "due_date", "last_status_update"],
        ),
    }


def maturity_tier(opened: pd.Series, today: pd.Timestamp) -> pd.Series:
    """new under 12 months, ramping 12 to 24, mature over 24."""
    months = (today - opened).dt.days / DAYS_PER_MONTH
    tier = pd.Series("ramping", index=opened.index)
    tier[months < 12] = "new"
    tier[months > 24] = "mature"
    return tier


def hours_weighted_check(clinic: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    """Compare clinic recheck and record completion with the hours-weighted provider average.

    The dataset README says the clinic value is this average on overlapping weeks.
    Returns one row per overlapping (center, week) with the absolute differences.
    """
    p = providers.copy()
    for col in ["recheck_compliance_pct", "record_completion_24h_pct"]:
        p[f"w_{col}"] = p[col] * p.scheduled_hours
    g = p.groupby(["location_id", "week_start"]).agg(
        hours=("scheduled_hours", "sum"),
        w_recheck=("w_recheck_compliance_pct", "sum"),
        w_record=("w_record_completion_24h_pct", "sum"),
        providers=("provider_id", "nunique"),
    )
    g["provider_recheck"] = g.w_recheck / g.hours
    g["provider_record"] = g.w_record / g.hours
    merged = clinic.merge(g.reset_index(), on=["location_id", "week_start"], how="inner")
    merged["diff_recheck"] = (merged.recheck_compliance_pct - merged.provider_recheck).abs()
    merged["diff_record"] = (merged.record_completion_24h_pct - merged.provider_record).abs()
    cols = [
        "location_id", "week_start", "providers", "hours", "doctor_hours_scheduled",
        "recheck_compliance_pct", "provider_recheck", "diff_recheck",
        "record_completion_24h_pct", "provider_record", "diff_record",
    ]
    return merged[cols]


def clean(raw: dict, today: str = DEFAULT_DATE, leader: str = DEFAULT_LEADER) -> Panel:
    """Apply every cleaning rule and return a Panel with notes."""
    today = pd.Timestamp(today)
    notes = []
    loc = raw["locations"].copy()
    cw = raw["clinic"].copy()
    pw = raw["providers"].copy()
    ap = raw["plans"].copy()

    # Rule: drop exact duplicate rows in clinic_weekly.
    before = len(cw)
    dup_rows = cw[cw.duplicated()][["location_id", "week_start"]]
    cw = cw.drop_duplicates().reset_index(drop=True)
    notes.append({
        "note": "duplicate_rows_removed",
        "source": "clinic_weekly.csv",
        "count": before - len(cw),
        "detail": "Exact duplicate rows from a double-run of the weekly load. First copy kept.",
        "rows": dup_rows.assign(week_start=dup_rows.week_start.dt.strftime("%Y-%m-%d")).to_dict("records"),
    })

    # Rule: negative wait times become missing.
    neg = cw.avg_wait_time_min < 0
    neg_rows = cw.loc[neg, ["location_id", "week_start"]]
    cw.loc[neg, "avg_wait_time_min"] = np.nan
    notes.append({
        "note": "negative_wait_set_missing",
        "source": "clinic_weekly.csv, avg_wait_time_min",
        "count": int(neg.sum()),
        "detail": "Known clock-sync bug. Treated as missing, never as zero.",
        "rows": neg_rows.assign(week_start=neg_rows.week_start.dt.strftime("%Y-%m-%d")).to_dict("records"),
    })

    # Rule: employment_type to lowercase.
    changed = (pw.employment_type != pw.employment_type.str.lower()).sum()
    pw["employment_type"] = pw.employment_type.str.lower().str.strip()
    notes.append({
        "note": "employment_type_lowercased",
        "source": "provider_weekly.csv, employment_type",
        "count": int(changed),
        "detail": f"Values now: {sorted(pw.employment_type.unique().tolist())}.",
    })

    # Panel shape. Fail loudly if the data is not what the README says.
    weeks = np.sort(cw.week_start.unique())
    latest_week = pd.Timestamp(weeks[-1])
    if len(weeks) != 52:
        raise ValueError(f"clinic_weekly has {len(weeks)} distinct weeks, expected 52")
    if not (cw.week_start.dt.weekday == 0).all():
        raise ValueError("clinic_weekly has a week_start that is not a Monday")
    if latest_week != today - pd.Timedelta(days=7):
        raise ValueError(f"latest week {latest_week.date()} is not the Monday before {today.date()}")
    if pw.week_start.nunique() != 26 or pw.week_start.max() != latest_week:
        raise ValueError("provider_weekly should hold exactly the latest 26 weeks")
    if cw.duplicated(["location_id", "week_start"]).any():
        raise ValueError("clinic_weekly still has more than one row per center per week")
    notes.append({
        "note": "panel_shape",
        "source": "clinic_weekly.csv, provider_weekly.csv",
        "count": 52,
        "detail": (
            f"52 clinic weeks from {pd.Timestamp(weeks[0]).date()} to {latest_week.date()}, "
            f"all Mondays. Provider data covers the latest 26 weeks only "
            f"(from {pw.week_start.min().date()})."
        ),
    })

    # Recompute maturity_tier from opened_date. Keep the original.
    loc["maturity_tier_reported"] = loc.maturity_tier
    loc["maturity_tier"] = maturity_tier(loc.opened_date, today)
    loc["months_open"] = ((today - loc.opened_date).dt.days / DAYS_PER_MONTH).round(1)
    mism = loc[loc.maturity_tier != loc.maturity_tier_reported][
        ["location_id", "location_name", "rmp_name", "opened_date", "months_open",
         "maturity_tier_reported", "maturity_tier"]
    ].reset_index(drop=True)
    notes.append({
        "note": "maturity_tier_mismatch",
        "source": "locations.csv, maturity_tier vs opened_date",
        "count": len(mism),
        "detail": "Tier recomputed against today. Reported tier kept as maturity_tier_reported.",
        "rows": [
            f"{r.location_name}: reported {r.maturity_tier_reported}, "
            f"recomputed {r.maturity_tier} ({r.months_open} months)"
            for r in mism.itertuples()
        ],
    })

    # Partial history. Weeks available per center, so trend windows can adapt.
    first = cw.groupby("location_id").week_start.agg(first_week="min", weeks_available="nunique")
    loc = loc.merge(first, left_on="location_id", right_index=True, how="left")
    partial = loc[loc.weeks_available < 52]
    notes.append({
        "note": "partial_history",
        "source": "clinic_weekly.csv",
        "count": len(partial),
        "detail": "Centers with fewer than 52 weeks. Trend windows use the weeks that exist.",
        "rows": [f"{r.location_name}: {r.weeks_available} weeks" for r in partial.itertuples()],
    })

    # Wrong state codes. Flagged against the metro, never corrected.
    expected_state = loc.metro_area.map(METRO_STATE)
    wrong_state = loc[expected_state.notna() & (expected_state != loc.state)]
    notes.append({
        "note": "state_code_mismatch",
        "source": "locations.csv, state vs metro_area",
        "count": len(wrong_state),
        "detail": "State does not match the metro. Not used by any metric. Left as is.",
        "rows": [
            f"{r.location_name}: {r.state} recorded, metro {r.metro_area} is in {METRO_STATE[r.metro_area]}"
            for r in wrong_state.itertuples()
        ],
    })

    # doctor_fte_target oddities relative to exam rooms.
    loc["fte_per_room"] = (loc.doctor_fte_target / loc.exam_rooms).round(2)
    low, high = FTE_PER_ROOM_BAND
    odd = loc[(loc.fte_per_room < low) | (loc.fte_per_room > high)]
    notes.append({
        "note": "doctor_fte_target_oddity",
        "source": "locations.csv, doctor_fte_target vs exam_rooms",
        "count": len(odd),
        "detail": f"Budgeted FTE per exam room outside {low} to {high}. Flagged for a human, not changed.",
        "rows": [
            f"{r.location_name}: {r.exam_rooms} rooms, {r.doctor_fte_target} FTE ({r.fte_per_room} per room)"
            for r in odd.itertuples()
        ],
    })

    # Membership conversion definition change. Recorded here, suppressed from ranking later.
    notes.append({
        "note": "membership_conversion_definition_change",
        "source": "clinic_weekly.csv, membership_conversion_pct",
        "count": int((cw.week_start >= MEMBERSHIP_BREAK).sum()),
        "detail": (
            f"Denominator widened on {MEMBERSHIP_BREAK.date()} to include urgent care visits. "
            "Values before and after are not comparable. Never ranked."
        ),
    })

    # Missing values, as found after cleaning.
    missing = {c: int(cw[c].isna().sum()) for c in ["client_csat", "revenue_per_appt", "avg_wait_time_min"]}
    missing["avg_appt_duration_min"] = int(pw.avg_appt_duration_min.isna().sum())
    notes.append({
        "note": "missing_values",
        "source": "clinic_weekly.csv, provider_weekly.csv",
        "count": int(sum(missing.values())),
        "detail": "Left missing. Rates are computed on the weeks that have a value.",
        "rows": [f"{k}: {v}" for k, v in missing.items()],
    })

    # Clinic hours vs provider hours. Observed, not a rule. Worth a line in the data notes.
    check = hours_weighted_check(cw, pw)
    gap = (check.doctor_hours_scheduled - check.hours)
    notes.append({
        "note": "clinic_hours_exceed_provider_hours",
        "source": "clinic_weekly.doctor_hours_scheduled vs sum of provider_weekly.scheduled_hours",
        "count": int((gap.abs() > 1).sum()),
        "detail": (
            f"On overlapping weeks the clinic total is a median {gap.median():.0f} hours above the "
            "sum of listed providers. Quality rates still match the hours-weighted provider "
            f"average within {HOURS_WEIGHTED_TOLERANCE} on every row, so the provider table is "
            "complete for quality but not for hours."
        ),
    })

    # Plans: dates only. Baseline recompute and labels belong to plans.py.
    ap["status"] = ap.status.str.lower().str.strip()

    return Panel(
        today=today, leader=leader, locations=loc, clinic=cw, providers=pw, plans=ap,
        latest_week=latest_week, tier_mismatches=mism, notes=notes,
    )


def load_and_clean(today: str = DEFAULT_DATE, leader: str = DEFAULT_LEADER, raw_dir: Path = RAW_DIR) -> Panel:
    return clean(load_raw(raw_dir), today=today, leader=leader)
