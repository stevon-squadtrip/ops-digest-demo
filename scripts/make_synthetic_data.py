"""Write the four raw CSVs in data/raw from a fixed seed. Nothing in them comes from a real company.

Run from the repo root: python scripts/make_synthetic_data.py

The panel is 24 clinics in four regions. The Medical Director, North owns Clinic 01 to Clinic 11 and reads
the digest. Every number is drawn from the generator below, then the planted cases are laid on top:

- Clinic 11: record completion drifts down for eleven weeks while its plan (AP_001) is marked on track, and
  its throughput plan (AP_009) is marked complete with a target the panel had already met when it opened.
  All four doctors slide. The plan marked on track while the metric worsens.
- Clinic 05: recheck compliance far below peers, driven by one doctor (DVM_0024) whose record completion is
  fine. Plan AP_002 carries a recorded baseline that does not match the panel.
- Clinic 06: staff call outs jump from 1 or 2 a week to 4, 7, 9, 9 while doctor hours fall. The same day call.
- Clinic 09: callback compliance sits at 100 nearly every week while recheck and record completion sit well
  below peers. A perfect score that points at a logging gap, never praise.
- Clinic 07: a new clinic with 7 CSAT responses last week. Too few to trust, so it is held back and watched.
  Its call outs ran 3 then 5, which is the two week run the new center rule needs.
- Clinic 02: a client satisfaction plan (AP_007) whose target was met at open, while throughput, the metric
  paired with it, slips for five weeks. The related metric that slips while the tracked one looks fine.
- membership_conversion_pct changes definition on 2026-02-09 at every clinic. Flagged, never ranked.
- Data faults the cleaning step must catch: 9 exact duplicate rows, 11 negative wait times, mixed case
  employment_type, 7 maturity tiers that disagree with opened_date, 7 state codes that disagree with the metro,
  two doctor FTE targets out of line with the exam rooms, and scattered missing values.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SEED = 20260504
TODAY = pd.Timestamp("2026-05-04")
N_WEEKS = 52
PROVIDER_WEEKS = 26
WEEKS = pd.date_range(end=TODAY - pd.Timedelta(days=7), periods=N_WEEKS, freq="7D")
LATEST = WEEKS[-1]
MEMBERSHIP_BREAK = pd.Timestamp("2026-02-09")
UNLISTED_HOURS = 15.0  # clinic hours above the sum of listed provider hours (unlisted relief cover)

W = {w.strftime("%Y-%m-%d"): w for w in WEEKS}
# Unplanted series end quietly: in the last two weeks their noise is clipped to one sd and their counts to the base
# plus one, so every line in the demo digest is a planted case and not a dice roll. History keeps its full noise.
QUIET_WEEKS = 2

REGIONS = {
    "North": {"metros": [("Northgate", "NG"), ("Riverbend", "RB")], "rmp": "Medical Director, North", "rop": "Regional Ops Partner, North"},
    "South": {"metros": [("Lakeside", "LK"), ("Fairmont", "FM")], "rmp": "Medical Director, South", "rop": "Regional Ops Partner, South"},
    "East": {"metros": [("Harborview", "HV"), ("Stonebridge", "SB")], "rmp": "Medical Director, East", "rop": "Regional Ops Partner, East"},
    "West": {"metros": [("Westfield", "WF"), ("Hillcrest", "HC")], "rmp": "Medical Director, West", "rop": "Regional Ops Partner, West"},
}
METRO_STATE = {m: s for r in REGIONS.values() for m, s in r["metros"]}

# One row per clinic: region, opened date, reported tier (may disagree with the date on purpose), rooms, FTE,
# wrong_state (plant a state code that disagrees with the metro), providers, and the per clinic base levels.
# Bases: hours, throughput, recheck, record, callback, csat, wait, revenue, membership, call outs, open reqs.
CLINICS = [
    # id, region, metro idx, opened, reported tier, rooms, fte, wrong_state, n_providers, bases
    ("CL_001", "North", 0, "2022-11-30", "mature", 7, 5.5, False, 5, dict(hours=190, thr=2.60, recheck=84.0, record=91.0, callback=86.0, csat=4.60, wait=11.0, rev=262, member=21.0, callouts=1.2, reqs=0.3)),
    ("CL_002", "North", 0, "2024-11-23", "mature", 6, 5.7, True, 5, dict(hours=170, thr=2.60, recheck=82.0, record=90.0, callback=85.0, csat=4.78, wait=12.0, rev=255, member=22.0, callouts=1.4, reqs=0.5)),
    ("CL_003", "North", 1, "2024-02-15", "mature", 8, 5.6, True, 6, dict(hours=200, thr=2.50, recheck=76.5, record=89.0, callback=84.0, csat=4.52, wait=12.5, rev=250, member=20.0, callouts=1.6, reqs=0.8)),
    ("CL_004", "North", 0, "2024-12-27", "mature", 4, 3.5, True, 3, dict(hours=120, thr=2.45, recheck=81.0, record=90.5, callback=87.0, csat=4.58, wait=13.0, rev=258, member=23.0, callouts=1.0, reqs=0.4)),
    ("CL_005", "North", 1, "2025-10-07", "mature", 8, 4.8, False, 5, dict(hours=165, thr=2.55, recheck=65.0, record=89.0, callback=83.0, csat=4.50, wait=12.0, rev=248, member=19.0, callouts=1.5, reqs=1.0)),
    ("CL_006", "North", 1, "2024-08-09", "mature", 5, 4.2, True, 5, dict(hours=165, thr=2.50, recheck=83.0, record=88.5, callback=85.5, csat=4.55, wait=9.0, rev=252, member=21.5, callouts=1.5, reqs=1.0)),
    ("CL_007", "North", 0, "2025-12-05", "new", 4, 2.4, False, 3, dict(hours=78, thr=2.30, recheck=80.0, record=89.5, callback=84.5, csat=4.45, wait=10.5, rev=240, member=18.0, callouts=0.6, reqs=1.0)),
    ("CL_008", "North", 1, "2021-11-15", "ramping", 5, 4.5, True, 5, dict(hours=160, thr=2.65, recheck=83.5, record=92.0, callback=88.0, csat=4.62, wait=12.0, rev=265, member=24.0, callouts=1.3, reqs=0.2)),
    ("CL_009", "North", 0, "2025-08-27", "ramping", 4, 6.4, True, 4, dict(hours=135, thr=2.40, recheck=74.0, record=86.0, callback=100.0, csat=4.50, wait=11.5, rev=245, member=19.5, callouts=1.2, reqs=0.6)),
    ("CL_010", "North", 1, "2026-01-03", "mature", 5, 5.5, True, 4, dict(hours=130, thr=2.50, recheck=79.5, record=90.0, callback=86.0, csat=4.55, wait=11.0, rev=250, member=20.5, callouts=1.1, reqs=0.5)),
    ("CL_011", "North", 0, "2023-03-23", "mature", 7, 2.7, False, 4, dict(hours=150, thr=2.50, recheck=82.5, record=92.0, callback=86.5, csat=4.60, wait=11.0, rev=260, member=22.0, callouts=1.3, reqs=0.4)),
    ("CL_012", "South", 0, "2022-07-10", "mature", 8, 4.0, False, 5, dict(hours=175, thr=2.55, recheck=80.0, record=89.0, callback=85.0, csat=4.55, wait=12.0, rev=255, member=21.0, callouts=1.5, reqs=0.6)),
    ("CL_013", "South", 1, "2024-07-02", "ramping", 7, 4.6, False, 5, dict(hours=170, thr=2.50, recheck=79.0, record=88.0, callback=84.0, csat=4.50, wait=13.0, rev=250, member=20.0, callouts=1.6, reqs=0.8)),
    ("CL_014", "South", 0, "2022-08-30", "mature", 8, 5.4, False, 5, dict(hours=185, thr=2.60, recheck=83.0, record=90.0, callback=87.0, csat=4.62, wait=11.0, rev=262, member=22.0, callouts=1.2, reqs=0.4)),
    ("CL_015", "South", 1, "2025-10-15", "new", 8, 3.6, False, 5, dict(hours=140, thr=2.40, recheck=78.0, record=88.0, callback=83.0, csat=4.48, wait=12.5, rev=245, member=19.0, callouts=1.4, reqs=1.0)),
    ("CL_016", "South", 0, "2023-11-03", "mature", 7, 6.1, False, 5, dict(hours=195, thr=2.55, recheck=81.0, record=90.0, callback=86.0, csat=4.57, wait=11.5, rev=258, member=21.0, callouts=1.5, reqs=0.5)),
    ("CL_017", "East", 0, "2022-05-21", "mature", 4, 2.8, False, 5, dict(hours=110, thr=2.50, recheck=82.0, record=91.0, callback=86.0, csat=4.60, wait=10.5, rev=260, member=23.0, callouts=1.0, reqs=0.3)),
    ("CL_018", "East", 1, "2025-06-11", "new", 8, 5.2, False, 5, dict(hours=175, thr=2.45, recheck=78.0, record=88.0, callback=84.0, csat=4.50, wait=13.0, rev=248, member=19.5, callouts=1.6, reqs=1.2)),
    ("CL_019", "East", 0, "2025-11-24", "new", 7, 4.0, False, 5, dict(hours=130, thr=2.35, recheck=79.0, record=89.0, callback=85.0, csat=4.52, wait=12.0, rev=246, member=19.0, callouts=1.3, reqs=1.0)),
    ("CL_020", "East", 1, "2023-06-14", "mature", 5, 5.4, False, 5, dict(hours=180, thr=2.38, recheck=81.0, record=90.5, callback=86.5, csat=4.58, wait=11.5, rev=257, member=21.5, callouts=1.4, reqs=0.5)),
    ("CL_021", "West", 0, "2020-10-21", "mature", 8, 3.5, False, 5, dict(hours=150, thr=2.60, recheck=83.0, record=91.0, callback=87.0, csat=4.63, rev=265, wait=10.5, member=23.5, callouts=1.1, reqs=0.3)),
    ("CL_022", "West", 1, "2022-05-06", "mature", 7, 3.4, False, 5, dict(hours=120, thr=2.55, recheck=81.5, record=90.0, callback=86.0, csat=4.58, wait=11.0, rev=258, member=22.0, callouts=1.2, reqs=0.4)),
    ("CL_023", "West", 0, "2023-07-16", "mature", 7, 3.6, False, 5, dict(hours=140, thr=2.50, recheck=80.5, record=89.5, callback=85.5, csat=4.55, wait=12.0, rev=252, member=21.0, callouts=1.4, reqs=0.6)),
    ("CL_024", "West", 1, "2025-09-21", "new", 8, 3.5, False, 5, dict(hours=135, thr=2.40, recheck=78.0, record=88.5, callback=84.0, csat=4.50, wait=12.5, rev=247, member=19.5, callouts=1.5, reqs=0.9)),
]

# Planted paths. A path maps week -> value; weeks not listed follow the base. Values are clinic level targets.
# Clinic 11 record completion: five steps down over eleven weeks, no single week alarming.
C11_RECORD = {"2026-02-09": 92.8, "2026-02-16": 90.6, "2026-02-23": 86.4, "2026-03-02": 85.2, "2026-03-09": 83.6,
              "2026-03-16": 81.6, "2026-03-23": 80.4, "2026-03-30": 78.3, "2026-04-06": 74.8, "2026-04-13": 77.2,
              "2026-04-20": 78.6, "2026-04-27": 79.5}
# Clinic 11 throughput: rises from mid February and stays up. 2.66 in the week AP_009 opened, 2.98 last week.
C11_THR = {"2026-01-26": 2.55, "2026-02-02": 2.60, "2026-02-09": 2.66, "2026-02-16": 2.72, "2026-02-23": 2.80,
           "2026-03-02": 2.86, "2026-03-09": 2.90, "2026-03-16": 2.88, "2026-03-23": 2.93, "2026-03-30": 2.95,
           "2026-04-06": 2.91, "2026-04-13": 2.96, "2026-04-20": 2.94, "2026-04-27": 2.98}
# Clinic 06 staff call outs: 1 or 2 a week until March, then 4, 7, 9, 9.
C06_CALLOUTS = {"2026-03-02": 1, "2026-03-09": 2, "2026-03-16": 2, "2026-03-23": 3, "2026-03-30": 3,
                "2026-04-06": 4, "2026-04-13": 7, "2026-04-20": 9, "2026-04-27": 9}
# Clinic 06 doctor hours: 165 twelve weeks ago down to 138 last week.
C06_HOURS = {w.strftime("%Y-%m-%d"): round(165 - (165 - 138) * i / 11, 1) for i, w in enumerate(WEEKS[-12:])}
# Clinic 02 throughput: a five week slip after the CSAT plan opened. Enough for the supporting line, not an item.
C02_THR = {"2026-03-30": 2.44, "2026-04-06": 2.42, "2026-04-13": 2.45, "2026-04-20": 2.43, "2026-04-27": 2.44}
# Clinic 02 callbacks: up to 91.8 last week.
C02_CALLBACK = {"2026-04-13": 88.0, "2026-04-20": 90.1, "2026-04-27": 91.8}
# Clinic 02 CSAT at the week AP_007 opened: 4.80 (target 4.55 already met).
C02_CSAT = {"2025-11-10": 4.62, "2025-11-17": 4.66, "2025-11-24": 4.70, "2025-12-01": 4.80}
# Clinic 01 recheck: held above its long run average for the last eight weeks. The opportunity source.
C01_RECHECK = {w.strftime("%Y-%m-%d"): v for w, v in zip(WEEKS[-8:], [87.0, 87.8, 88.4, 87.6, 88.9, 88.1, 88.7, 88.3])}
# Clinic 03 no shows: 56 last week against an 8 week average near 32.
C03_NOSHOW = {"2026-04-27": 52}
# Clinic 08 no shows 37 last week against 20; wait time improved over the last four weeks.
C08_NOSHOW = {"2026-04-27": 36}
C08_WAIT = {"2026-04-06": 9.4, "2026-04-13": 8.8, "2026-04-20": 9.1, "2026-04-27": 8.6}
# Clinic 04 wait time improved.
C04_WAIT = {"2026-04-13": 10.8, "2026-04-20": 10.3, "2026-04-27": 10.6}
# Clinic 07 call outs 3 then 5; CSAT 3.54 on 7 responses last week.
C07_QUIET = [1, 1, 0, 1, 2, 1, 1, 0, 1, 1, 2, 1, 0, 1, 1, 1, 2, 0, 1, 1]
C07_CALLOUTS = {"2026-04-20": 3, "2026-04-27": 5}
C07_CSAT = {"2026-04-06": 4.50, "2026-04-13": 4.45, "2026-04-20": 4.48, "2026-04-27": 3.54}
C07_CALLBACK = {"2026-04-13": 84.8, "2026-04-20": 85.1, "2026-04-27": 84.6}
C07_CSAT_N = {"2026-04-06": 11, "2026-04-13": 12, "2026-04-20": 9, "2026-04-27": 7}
# Clinic 07 throughput 2.29 in the week AP_027 opened (recorded baseline says 2.5).
C07_THR = {"2026-03-23": 2.31, "2026-03-30": 2.29, "2026-04-06": 2.27, "2026-04-13": 2.30, "2026-04-20": 2.28, "2026-04-27": 2.29}
# Clinic 10 callbacks: 88.5 in the week AP_004 opened, the four weeks before it lower, then above 88 since.
C10_CALLBACK = {"2026-01-19": 84.0, "2026-01-26": 85.2, "2026-02-02": 86.1, "2026-02-09": 88.5, "2026-02-16": 87.4,
                "2026-02-23": 88.9, "2026-03-02": 89.6, "2026-03-09": 89.0, "2026-03-16": 90.2, "2026-03-23": 89.8,
                "2026-03-30": 90.5, "2026-04-06": 90.1, "2026-04-13": 90.8, "2026-04-20": 89.9, "2026-04-27": 90.4}
# Clinic 10 throughput: 2.56 over the last four weeks, past AP_019's 2.33.
C10_THR = {"2026-04-06": 2.54, "2026-04-13": 2.57, "2026-04-20": 2.55, "2026-04-27": 2.58}
# Clinic 03 recheck near 76 in the week AP_020 opened.
C03_RECHECK = {"2026-03-30": 76.6, "2026-04-06": 77.2, "2026-04-13": 76.9, "2026-04-20": 77.4, "2026-04-27": 77.0}
# Clinic 24 recheck and Clinic 20 throughput improve on schedule, so AP_012 and AP_018 read confirmed with no flag.
C24_RECHECK = {w.strftime("%Y-%m-%d"): round(78.0 + 8.0 * min(1.0, i / 9), 1) for i, w in enumerate(WEEKS[-12:])}
C20_THR = {w.strftime("%Y-%m-%d"): round(2.38 + 0.11 * min(1.0, i / 12), 3) for i, w in enumerate(WEEKS[-16:])}
# Clinic 09 callbacks: 100 almost every week; one week off the ceiling inside the last ten.
C09_CALLBACK_OFF = {"2026-03-16": 98.4, "2025-11-24": 97.9, "2025-10-13": 98.8}

# Provider level overlays. DVM_0024 at Clinic 05: recheck near 41.8, record near 90.9, six of the last eight weeks.
DVM_0024_RECHECK = 41.5
DVM_0024_RECORD = 90.9
DVM_0024_ABSENT = {"2026-03-23", "2026-04-13"}
# Peers at Clinic 05 spread from the high sixties to the low eighties.
C05_PEER_RECHECK = [66.0, 71.0, 75.0, 79.0]

PLANS = [
    # plan_id, location_id, metric, recorded baseline (None: the panel value in the opening week), target, opened, due, status, last update
    ("AP_001", "CL_011", "record_completion_24h_pct", 91.8, 94.0, "2026-01-28", "2026-05-25", "on_track", "2026-04-28"),
    ("AP_002", "CL_005", "recheck_compliance_pct", 74.0, 88.0, "2026-02-18", "2026-06-13", "on_track", "2026-04-23"),
    ("AP_003", "CL_006", "avg_wait_time_min", 9.2, 7.0, "2025-12-05", "2026-04-19", "at_risk", "2026-02-22"),
    ("AP_004", "CL_010", "callback_compliance_pct", 79.0, 88.0, "2026-02-13", "2026-05-14", "on_track", "2026-04-30"),
    ("AP_005", "CL_021", "appts_per_doctor_hour", None, 2.73, "2025-12-31", "2026-03-06", "not_started", "2026-03-31"),
    ("AP_006", "CL_016", "record_completion_24h_pct", 80.2, 89.6, "2025-12-17", "2026-03-11", "not_started", "2026-04-02"),
    ("AP_007", "CL_002", "client_csat", 4.3, 4.55, "2025-12-05", "2026-03-29", "on_track", "2026-03-30"),
    ("AP_008", "CL_013", "avg_wait_time_min", 18.0, 13.4, "2026-03-04", "2026-04-19", "on_track", "2026-04-04"),
    ("AP_009", "CL_011", "appts_per_doctor_hour", 2.2, 2.52, "2026-02-15", "2026-04-13", "complete", "2026-04-18"),
    ("AP_010", "CL_022", "recheck_compliance_pct", 64.5, 73.0, "2026-01-07", "2026-03-16", "not_started", "2026-04-25"),
    ("AP_011", "CL_015", "avg_wait_time_min", None, 7.9, "2025-11-23", "2026-03-16", "complete", "2026-04-11"),
    ("AP_012", "CL_024", "recheck_compliance_pct", None, 89.9, "2026-01-26", "2026-05-24", "on_track", "2026-04-18"),
    ("AP_013", "CL_023", "appts_per_doctor_hour", None, 2.58, "2026-01-27", "2026-05-08", "at_risk", "2026-04-03"),
    ("AP_014", "CL_018", "client_csat", None, 4.65, "2026-02-09", "2026-06-03", "not_started", "2026-04-25"),
    ("AP_015", "CL_017", "appts_per_doctor_hour", 2.15, 2.41, "2025-11-20", "2026-01-24", "not_started", "2026-04-25"),
    ("AP_016", "CL_019", "callback_compliance_pct", 83.3, 92.6, "2025-10-30", "2026-03-13", "on_track", "2026-04-15"),
    ("AP_017", "CL_014", "client_csat", 4.25, 4.5, "2025-12-14", "2026-02-10", "at_risk", "2026-04-21"),
    ("AP_018", "CL_020", "appts_per_doctor_hour", None, 2.53, "2026-01-16", "2026-05-29", "on_track", "2026-04-08"),
    ("AP_019", "CL_010", "appts_per_doctor_hour", 2.07, 2.33, "2025-12-27", "2026-04-28", "on_track", "2026-03-27"),
    ("AP_020", "CL_003", "recheck_compliance_pct", 66.0, 78.6, "2026-03-31", "2026-07-21", "on_track", "2026-05-01"),
    ("AP_021", "CL_012", "callback_compliance_pct", None, 90.0, "2026-01-20", "2026-04-03", "on_track", "2026-04-24"),
    ("AP_022", "CL_017", "recheck_compliance_pct", None, 84.6, "2025-12-28", "2026-02-11", "on_track", "2026-04-05"),
    ("AP_023", "CL_014", "record_completion_24h_pct", 65.4, 77.8, "2026-01-03", "2026-04-09", "on_track", "2026-04-22"),
    ("AP_024", "CL_006", "recheck_compliance_pct", 65.9, 74.6, "2026-01-10", "2026-04-16", "on_track", "2026-04-25"),
    ("AP_025", "CL_019", "callback_compliance_pct", None, 91.7, "2026-03-29", "2026-06-12", "on_track", "2026-04-20"),
    ("AP_026", "CL_020", "appts_per_doctor_hour", None, 2.47, "2025-12-07", "2026-03-30", "on_track", "2026-04-17"),
    ("AP_027", "CL_007", "appts_per_doctor_hour", 2.5, 2.72, "2026-04-01", "2026-06-07", "on_track", "2026-04-03"),
    ("AP_028", "CL_022", "avg_wait_time_min", 16.9, 11.9, "2025-12-03", "2026-02-18", "on_track", "2026-04-07"),
]


def monday_on_or_before(d: pd.Timestamp) -> pd.Timestamp:
    return (d - pd.Timedelta(days=d.weekday())).normalize()


def path_value(path: dict, week: pd.Timestamp, default: float) -> float:
    return path.get(week.strftime("%Y-%m-%d"), default)


def noise(rng, sd: float, quiet: bool, size=None):
    """Gaussian noise; in the quiet tail it is clipped to one sd."""
    x = rng.normal(0, sd, size)
    return np.clip(x, -0.5 * sd, 0.5 * sd) if quiet else x


def make_locations() -> pd.DataFrame:
    rows = []
    for lid, region, mi, opened, tier, rooms, fte, wrong, _, _ in CLINICS:
        metro, state = REGIONS[region]["metros"][mi]
        if wrong:
            # Borrow the other metro's code in the same region, so the code never matches the metro.
            state = REGIONS[region]["metros"][1 - mi][1]
        rows.append({"location_id": lid, "location_name": f"Clinic {lid[-2:]}", "metro_area": metro, "state": state,
                     "region": region, "rmp_name": REGIONS[region]["rmp"], "rop_name": REGIONS[region]["rop"],
                     "opened_date": opened, "maturity_tier": tier, "exam_rooms": rooms, "doctor_fte_target": fte})
    return pd.DataFrame(rows)


def make_providers(rng: np.random.Generator) -> list:
    """One dict per provider: id, clinic, type, tenure, base hours, recheck and record offsets, presence rate."""
    out, n = [], 0
    for lid, _, _, _, _, _, _, _, n_prov, base in CLINICS:
        for k in range(n_prov):
            n += 1
            pid = f"DVM_{n:04d}"
            etype = rng.choice(["full_time", "part_time", "relief"], p=[0.66, 0.18, 0.16])
            hours = {"full_time": rng.uniform(28, 36), "part_time": rng.uniform(14, 22), "relief": rng.uniform(8, 16)}[etype]
            presence = {"full_time": 0.95, "part_time": 0.85, "relief": 0.55}[etype]
            p = {"provider_id": pid, "location_id": lid, "employment_type": etype, "tenure_months": int(rng.integers(3, 96)),
                 "hours": hours, "presence": presence, "recheck_off": rng.normal(0, 4.0), "record_off": rng.normal(0, 2.5),
                 "duration": rng.uniform(24, 30)}
            if lid == "CL_005":
                p["employment_type"], p["presence"], p["hours"] = "full_time", 0.97, rng.uniform(28, 34)
                if pid == "DVM_0024":
                    p.update(tenure_months=89, hours=31.2, recheck_abs=DVM_0024_RECHECK, record_abs=DVM_0024_RECORD)
                else:
                    p.update(recheck_abs=C05_PEER_RECHECK[k], record_off=rng.normal(-1.0, 1.0))
            if lid == "CL_011":
                p["employment_type"], p["presence"], p["duration"] = "full_time", 0.96, 27.0
                p["recheck_off"], p["record_off"] = rng.normal(0, 2.0), rng.normal(0, 1.5)
            if lid == "CL_009":
                p["recheck_off"], p["record_off"] = rng.normal(0, 3.0), rng.normal(0, 2.0)
            out.append(p)
    for lid in {p["location_id"] for p in out}:
        group = [p for p in out if p["location_id"] == lid and "recheck_abs" not in p]
        for key in ("recheck_off", "record_off"):
            m = float(np.mean([p[key] for p in group])) if group else 0.0
            for p in group:
                p[key] -= m
    return out


def generate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    loc = make_locations()
    providers = make_providers(rng)
    by_clinic = {}
    for p in providers:
        by_clinic.setdefault(p["location_id"], []).append(p)

    clinic_rows, provider_rows = [], []
    for lid, region, _, opened, _, _, _, _, _, base in CLINICS:
        first = monday_on_or_before(pd.Timestamp(opened))
        weeks = [w for w in WEEKS if w >= first]
        # Mild week to week memory so series are not white noise, pulled back to zero so nothing drifts by accident.
        wobble = np.zeros(len(weeks))
        for i in range(1, len(weeks)):
            wobble[i] = 0.3 * wobble[i - 1] + rng.normal(0, 0.3)
        for i, w in enumerate(weeks):
            wk = w.strftime("%Y-%m-%d")
            quiet = i >= len(weeks) - QUIET_WEEKS
            if quiet:
                wobble[i] = 0.0
            # Clinic level targets for the quality rates, then providers are drawn around them.
            recheck_t = base["recheck"] + wobble[i] * 2
            record_t = base["record"] + wobble[i] * 1.2
            if lid == "CL_011":
                record_t = path_value(C11_RECORD, w, base["record"] + noise(rng, 0.6, quiet))
            if lid == "CL_001":
                recheck_t = path_value(C01_RECHECK, w, base["recheck"] + noise(rng, 1.2, quiet))
            if lid == "CL_003":
                recheck_t = path_value(C03_RECHECK, w, recheck_t)
            if lid == "CL_024":
                recheck_t = path_value(C24_RECHECK, w, recheck_t)
            hours_t = base["hours"]
            if lid == "CL_006":
                hours_t = path_value(C06_HOURS, w, base["hours"])
            hours_t = hours_t * (1.0 + noise(rng, 0.03, quiet))
            thr = base["thr"] + noise(rng, 0.06, quiet)
            thr = {"CL_011": path_value(C11_THR, w, thr), "CL_002": path_value(C02_THR, w, thr),
                   "CL_007": path_value(C07_THR, w, thr), "CL_010": path_value(C10_THR, w, thr),
                   "CL_020": path_value(C20_THR, w, thr)}.get(lid, thr)

            # Providers present this week.
            present = []
            for p in by_clinic[lid]:
                if p["provider_id"] == "DVM_0024" and wk in DVM_0024_ABSENT:
                    continue
                if rng.random() > p["presence"] and not quiet:
                    continue
                present.append(p)
            listed_hours = max(hours_t - UNLISTED_HOURS, 20.0)
            if present:
                shares = np.array([p["hours"] for p in present])
                shares = shares / shares.sum()
                hrs = np.round(listed_hours * shares * (1.0 + noise(rng, 0.04, quiet, len(present))), 1)
                rc = np.array([round(min(100.0, max(25.0, (p.get("recheck_abs", recheck_t + p["recheck_off"])) + noise(rng, 1.6, quiet))), 1) for p in present])
                rd = np.array([round(min(100.0, max(55.0, (p.get("record_abs", record_t + p["record_off"])) + noise(rng, 1.2, quiet))), 1) for p in present])
                for p, h, a, b in zip(present, hrs, rc, rd):
                    if w < WEEKS[-PROVIDER_WEEKS]:
                        continue  # the provider table covers the latest 26 weeks only; the clinic rate above still uses them
                    provider_rows.append({
                        "provider_id": p["provider_id"], "location_id": lid, "week_start": wk,
                        "employment_type": p["employment_type"], "tenure_months": p["tenure_months"], "scheduled_hours": float(h),
                        "appts_completed": int(round(h * thr * rng.normal(1.0, 0.05))),
                        "recheck_compliance_pct": float(a), "record_completion_24h_pct": float(b),
                        "avg_appt_duration_min": round(p["duration"] + rng.normal(0, 0.8), 1),
                    })
                recheck = round(float((rc * hrs).sum() / hrs.sum()), 1)
                record = round(float((rd * hrs).sum() / hrs.sum()), 1)
                doctor_hours = round(float(hrs.sum()) + UNLISTED_HOURS + rng.normal(0, 2.0), 1)
            else:
                recheck = round(recheck_t + rng.normal(0, 1.5), 1)
                record = round(record_t + rng.normal(0, 1.0), 1)
                doctor_hours = round(hours_t, 1)

            appts = int(round(doctor_hours * thr))
            callback = base["callback"] + wobble[i] * 1.5 + noise(rng, 2.2, quiet)
            if lid == "CL_009":
                callback = path_value(C09_CALLBACK_OFF, w, 100.0)
            if lid == "CL_002":
                callback = path_value(C02_CALLBACK, w, callback)
            if lid == "CL_010":
                callback = path_value(C10_CALLBACK, w, callback)
            if lid == "CL_007":
                callback = path_value(C07_CALLBACK, w, callback)
            csat = base["csat"] + noise(rng, 0.08, quiet)
            csat_n = int(round(appts * rng.uniform(0.07, 0.11)))
            if lid == "CL_002":
                csat = path_value(C02_CSAT, w, csat)
            if lid == "CL_007":
                csat = path_value(C07_CSAT, w, csat)
                csat_n = int(path_value(C07_CSAT_N, w, rng.integers(8, 16)))
            wait = base["wait"] + wobble[i] * 0.8 + noise(rng, 1.1, quiet)
            if lid == "CL_006":
                wait = 9.0 + noise(rng, 0.5, quiet)
            wait = {"CL_008": path_value(C08_WAIT, w, wait), "CL_004": path_value(C04_WAIT, w, wait)}.get(lid, wait)
            noshow_base = appts * 0.075
            if lid == "CL_003":
                noshow_base = 32
            if lid == "CL_008":
                noshow_base = 20
            if lid == "CL_011":
                noshow_base = 28
            noshow = int(noshow_base + noise(rng, 6.5 if lid == "CL_008" else max(2.0, noshow_base * 0.17), quiet))
            noshow = {"CL_003": int(path_value(C03_NOSHOW, w, noshow)), "CL_008": int(path_value(C08_NOSHOW, w, noshow))}.get(lid, noshow)
            member = base["member"] + wobble[i] * 1.5 + noise(rng, 1.8, quiet) + (8.0 if w >= MEMBERSHIP_BREAK else 0.0)
            callouts = int(min(rng.poisson(base["callouts"]), round(base["callouts"]) + 1) if quiet else rng.poisson(base["callouts"]))
            if lid == "CL_006":
                callouts = int(path_value(C06_CALLOUTS, w, 1 + (1 if rng.random() < 0.35 else 0)))
            if lid == "CL_007":
                callouts = int(path_value(C07_CALLOUTS, w, C07_QUIET[i % len(C07_QUIET)]))
            reqs = int(round(base["reqs"])) + (1 if rng.random() < 0.08 and not quiet else 0)
            clinic_rows.append({
                "location_id": lid, "week_start": wk, "doctor_hours_scheduled": doctor_hours, "appts_completed": appts,
                "appts_no_show": max(0, noshow), "appts_cancelled": max(0, int(rng.normal(appts * 0.052, 4))),
                "appts_per_doctor_hour": round(appts / doctor_hours, 3),
                "recheck_compliance_pct": recheck, "record_completion_24h_pct": record,
                "callback_compliance_pct": round(min(100.0, callback), 1),
                "client_csat": round(min(5.0, csat), 2), "csat_responses": max(csat_n, 5),
                "avg_wait_time_min": round(max(3.0, wait), 1), "revenue_per_appt": round(base["rev"] + wobble[i] * 6 + noise(rng, 7, quiet), 2),
                "membership_conversion_pct": round(member, 1), "staff_call_outs": callouts,
                "open_dvm_requisitions": reqs,
            })

    cw = pd.DataFrame(clinic_rows)
    pw = pd.DataFrame(provider_rows)

    # Faults for the cleaning step.
    north = cw.location_id.isin([c[0] for c in CLINICS if c[1] == "North"])
    old = cw.week_start < "2025-12-01"
    neg_pool = cw.index[(~north) | old]
    neg_idx = rng.choice(neg_pool, 11, replace=False)
    cw.loc[neg_idx, "avg_wait_time_min"] = -cw.loc[neg_idx, "avg_wait_time_min"]
    open_weeks = {(lid, monday_on_or_before(pd.Timestamp(o)).strftime("%Y-%m-%d")) for _, lid, _, _, _, o, _, _, _ in PLANS}
    not_open = ~pd.Series([(a, b) in open_weeks for a, b in zip(cw.location_id, cw.week_start)], index=cw.index)
    csat_idx = rng.choice(cw.index[(~north) & not_open], 46, replace=False)
    cw.loc[csat_idx, "client_csat"] = np.nan
    rev_idx = rng.choice(cw.index[(~north) & not_open], 30, replace=False)
    cw.loc[rev_idx, "revenue_per_appt"] = np.nan
    dur_idx = rng.choice(pw.index, 101, replace=False)
    pw.loc[dur_idx, "avg_appt_duration_min"] = np.nan
    caps = rng.choice(pw.index, int(len(pw) * 0.04), replace=False)
    pw.loc[caps, "employment_type"] = pw.loc[caps, "employment_type"].str.upper()
    dups = cw.sample(9, random_state=seed)
    cw = pd.concat([cw, dups], ignore_index=True)

    ap = pd.DataFrame(PLANS, columns=["plan_id", "location_id", "target_metric", "baseline_value", "target_value",
                                      "opened_date", "due_date", "status", "last_status_update"])
    panel = cw.drop_duplicates(["location_id", "week_start"]).set_index(["location_id", "week_start"])
    for i, r in ap.iterrows():
        if pd.isna(r.baseline_value):
            bweek = monday_on_or_before(pd.Timestamp(r.opened_date)).strftime("%Y-%m-%d")
            ap.loc[i, "baseline_value"] = round(float(panel.loc[(r.location_id, bweek), r.target_metric]), 2)
    rmp = loc.set_index("location_id").rmp_name
    ap.insert(2, "owner_name", ap.location_id.map(rmp))
    ap["notes"] = ""
    return {"locations": loc, "clinic": cw, "providers": pw, "plans": ap}


def write(data: dict, raw: Path = RAW) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    data["locations"].to_csv(raw / "locations.csv", index=False, lineterminator="\n")
    data["clinic"].to_csv(raw / "clinic_weekly.csv", index=False, lineterminator="\n")
    data["providers"].to_csv(raw / "provider_weekly.csv", index=False, lineterminator="\n")
    data["plans"].to_csv(raw / "action_plans.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    d = generate()
    write(d)
    for k, v in d.items():
        print(f"{k:<10} {len(v):>6} rows")
    print(f"clinic weeks {d['clinic'].week_start.nunique()}, provider weeks {d['providers'].week_start.nunique()}, seed {SEED}")
    sys.exit(0)
