"""Cleaning tests. Counts come from data/README.md and tests/fixtures/expected.json, never from the code under test."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.clean import HOURS_WEIGHTED_TOLERANCE, hours_weighted_check, load_and_clean, load_raw
from src.metrics import METRICS, long_panel

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected.json").read_text(encoding="utf-8"))
NOTES = EXPECTED["data_notes"]
NORTH_CENTERS = set(NOTES["north_centers"])


@pytest.fixture(scope="module")
def panel():
    return load_and_clean()


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def note(panel, name):
    return next(n for n in panel.notes if n["note"] == name)


def test_four_files_load(raw):
    assert len(raw["locations"]) == NOTES["locations"]
    assert len(raw["plans"]) == NOTES["plans"]
    assert raw["clinic"].week_start.dtype.kind == "M"


def test_nine_duplicates_removed(raw, panel):
    assert note(panel, "duplicate_rows_removed")["count"] == NOTES["duplicates"]
    assert len(panel.clinic) == len(raw["clinic"]) - NOTES["duplicates"]
    assert not panel.clinic.duplicated(["location_id", "week_start"]).any()


def test_eleven_negative_waits_set_missing(raw, panel):
    assert (raw["clinic"].avg_wait_time_min < 0).sum() == NOTES["negative_waits"]
    assert note(panel, "negative_wait_set_missing")["count"] == NOTES["negative_waits"]
    assert not (panel.clinic.avg_wait_time_min < 0).any()
    assert panel.clinic.avg_wait_time_min.isna().sum() == NOTES["negative_waits"]


def test_employment_type_lowercase(raw, panel):
    assert raw["providers"].employment_type.str.contains("[A-Z]").any()
    assert set(panel.providers.employment_type.unique()) == {"full_time", "part_time", "relief"}


def test_panel_shape(panel):
    assert panel.today == pd.Timestamp("2026-05-04")
    assert panel.latest_week == pd.Timestamp("2026-04-27")
    assert panel.clinic.week_start.nunique() == 52
    assert (panel.clinic.week_start.dt.weekday == 0).all()
    assert panel.providers.week_start.nunique() == 26
    assert panel.providers.week_start.max() == panel.latest_week


def test_wrong_date_fails_loudly():
    with pytest.raises(ValueError):
        load_and_clean(today="2026-05-11")


def test_maturity_tier_recomputed(panel):
    loc = panel.locations.set_index("location_name")
    assert "maturity_tier_reported" in loc.columns
    assert loc.loc["Clinic 10", "maturity_tier"] == "new"
    assert loc.loc["Clinic 08", "maturity_tier"] == "mature"
    assert loc.loc["Clinic 05", "maturity_tier"] == "new"
    assert loc.loc["Clinic 01", "maturity_tier"] == "mature"
    flagged = sorted(panel.tier_mismatches.location_name)
    assert flagged == NOTES["tier_mismatch_centers"]
    assert note(panel, "maturity_tier_mismatch")["count"] == len(flagged)


def test_leader_span_by_rmp_name(panel):
    assert len(panel.my_ids) == 11
    assert set(panel.my_locations.location_name) == NORTH_CENTERS
    assert (panel.my_locations.rop_name == "Regional Ops Partner, North").all()
    assert set(panel.my_clinic.location_id) == set(panel.my_ids)


def test_other_leader_runs_with_same_logic(panel):
    other = panel.locations.loc[panel.locations.rmp_name != panel.leader, "rmp_name"].iloc[0]
    p2 = load_and_clean(leader=other)
    assert p2.leader == other
    assert len(p2.my_ids) > 0
    assert not set(p2.my_ids) & set(panel.my_ids)


def test_partial_history_recorded(panel):
    weeks = panel.locations.set_index("location_name").weeks_available
    for center, n in NOTES["partial_history"].items():
        assert weeks[center] == n, center


def test_clinic_quality_is_hours_weighted_provider_average(panel):
    check = hours_weighted_check(panel.clinic, panel.providers)
    assert len(check) > 500
    assert check.diff_recheck.max() <= HOURS_WEIGHTED_TOLERANCE
    assert check.diff_record.max() <= HOURS_WEIGHTED_TOLERANCE
    # The test has teeth: a plain unweighted mean does not match.
    plain = panel.providers.groupby(["location_id", "week_start"]).recheck_compliance_pct.mean()
    merged = panel.clinic.merge(plain.rename("plain"), on=["location_id", "week_start"])
    assert (merged.recheck_compliance_pct - merged.plain).abs().max() > 1.0


def test_data_notes_cover_readme_issues(panel):
    names = {n["note"] for n in panel.notes}
    for required in [
        "duplicate_rows_removed", "negative_wait_set_missing", "employment_type_lowercased",
        "membership_conversion_definition_change", "partial_history", "missing_values",
        "maturity_tier_mismatch", "state_code_mismatch", "doctor_fte_target_oddity",
    ]:
        assert required in names
    odd = " ".join(note(panel, "doctor_fte_target_oddity")["rows"])
    for center in NOTES["fte_oddity_centers"]:
        assert center in odd
    assert note(panel, "state_code_mismatch")["count"] > 0


def test_long_panel_shape(panel):
    lp = long_panel(panel)
    assert set(lp.metric) == set(METRICS)
    assert lp.location_id.nunique() == 11
    assert len(lp) == len(panel.my_clinic) * len(METRICS)
    csat = lp[lp.metric == "client_csat"]
    assert csat.denominator.notna().all()
