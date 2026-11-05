# Synthetic dataset

Every file in data/raw is written by scripts/make_synthetic_data.py from a fixed seed. No row comes from a real
company. Regenerate with `python scripts/make_synthetic_data.py`.

Treat Monday 2026-05-04 as today. The weekly panel covers 52 weeks; week_start is always a Monday and the most
recent complete week begins 2026-04-27. Provider rows exist for the latest 26 weeks only.

## Files

| File | Rows | What it holds |
|---|---|---|
| locations.csv | 24 | One row per clinic: region, metro, opened_date, reported maturity tier, exam rooms, doctor FTE target, and the medical director (rmp_name) and ops partner (rop_name) who own it |
| clinic_weekly.csv | about 1,080 | One row per clinic per week: the operating panel |
| provider_weekly.csv | about 2,500 | One row per doctor per week, latest 26 weeks |
| action_plans.csv | 28 | Improvement plans with owner, baseline, target, dates and a self reported status |

locations.rmp_name defines a leader's span. The digest runs for "Medical Director, North", who owns Clinic 01 to
Clinic 11.

## Column notes

- recheck_compliance_pct: share of visits where a recheck was scheduled before the client left. Clinic level
  values are the hours weighted average of the clinic's providers in weeks where both tables overlap.
- record_completion_24h_pct: share of visit records closed within 24 hours. Same clinic level rule.
- callback_compliance_pct: share of required client callbacks completed. A rate only; the count of required
  callbacks is not in the data.
- client_csat and csat_responses: mean survey score and the number of responses that week.
- avg_wait_time_min: minutes from check in to room. A known clock sync bug writes some weeks as negative.
- membership_conversion_pct: the denominator widened on 2026-02-09 to include urgent care visits. Values before
  and after that date are not comparable.
- staff_call_outs and open_dvm_requisitions: weekly counts.
- action_plans.status is self reported by the owner and is never corrected by the data.

## Known faults, on purpose

Nine exact duplicate rows in clinic_weekly, eleven negative wait times, mixed case employment_type, seven
maturity tiers that disagree with opened_date, seven state codes that disagree with the metro, two doctor FTE
targets out of line with the exam rooms, and scattered missing values. The cleaning step logs each one.
