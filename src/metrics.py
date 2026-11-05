"""The metric layer: which clinic metrics the digest watches, and a long panel of them.

One row per (center, week, metric). Later stages score this table.
Weights and directions come from CLAUDE.md "Ranking rules".
"""
import pandas as pd

from src.clean import MEMBERSHIP_BREAK, Panel

# better: which direction is good. weight: clinical weight. denom: column that sizes the rate.
# lane: medical or ops. A label only, confirmed with the client Oct 21. Never changes a score or a suppression.
METRICS = {
    "recheck_compliance_pct": dict(label="Recheck compliance", lane="medical", better="higher", weight=1.5, denom="appts_completed"),
    "record_completion_24h_pct": dict(label="Record completion within 24h", lane="medical", better="higher", weight=1.5, denom="appts_completed"),
    "callback_compliance_pct": dict(label="Callback compliance", lane="medical", better="higher", weight=1.5, denom="appts_completed"),
    "avg_wait_time_min": dict(label="Average wait time", lane="ops", better="lower", weight=1.0, denom="appts_completed"),
    "client_csat": dict(label="Client satisfaction", lane="ops", better="higher", weight=1.0, denom="csat_responses"),
    "appts_per_doctor_hour": dict(label="Throughput per doctor hour", lane="ops", better="higher", weight=0.7, denom="appts_completed"),
    "revenue_per_appt": dict(label="Revenue per appointment", lane="ops", better="higher", weight=0.7, denom="appts_completed"),
    "appts_no_show": dict(label="No shows", lane="ops", better="lower", weight=0.7, denom=None),
    "staff_call_outs": dict(label="Staff call outs", lane="ops", better="lower", weight=1.2, denom=None),
    "open_dvm_requisitions": dict(label="Open doctor requisitions", lane="ops", better="lower", weight=1.2, denom=None),
    "membership_conversion_pct": dict(label="Membership conversion", lane="ops", better="higher", weight=0.7, denom="appts_completed",
                                      suppressed=f"definition changed on {MEMBERSHIP_BREAK.date()}"),
}
LANES = {"medical", "ops"}

# Below these, a rate is a watch item, never an alarm.
MIN_CSAT_RESPONSES = 10
MIN_APPTS = 30


def long_panel(panel: Panel, mine_only: bool = True) -> pd.DataFrame:
    """Melt the clinic table into one row per (center, week, metric) with its denominator."""
    cw = panel.my_clinic if mine_only else panel.clinic.copy()
    names = panel.locations.set_index("location_id").location_name
    rows = []
    for metric, spec in METRICS.items():
        part = cw[["location_id", "week_start", metric]].rename(columns={metric: "value"})
        part["metric"] = metric
        part["denominator"] = cw[spec["denom"]].values if spec["denom"] else pd.NA
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["location_name"] = out.location_id.map(names)
    out = out[["location_id", "location_name", "week_start", "metric", "value", "denominator"]]
    return out.sort_values(["location_id", "metric", "week_start"]).reset_index(drop=True)
