"""Site tests: the static page is built from the accepted digest, carries its evidence, and makes no network call."""
import json
import re

import pandas as pd
import pytest

from src import site
from src.clean import ROOT


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "index.html"
    site.build(out)
    return out.read_text(encoding="utf-8")


def test_site_builds_from_accepted_digest(page):
    assert "<h1>Monday digest: Medical Director, North</h1>" in page
    accepted = (ROOT / "outputs" / "digest_accepted.md").read_text(encoding="utf-8")
    current = (ROOT / "outputs" / "digest.md").read_text(encoding="utf-8")
    assert accepted == current, "digest.md must be the accepted run (python -m src.run --stage digest --cached)"
    for sid in ["at_a_glance", "one_thing_today", "item1", "item5", "plan_check", "opportunity", "watching", "feedback", "appendix"]:
        assert f"id='{sid}'" in page


def test_every_citation_has_an_evidence_row(page):
    md = (ROOT / "outputs" / "digest.md").read_text(encoding="utf-8")
    cites = set(re.findall(r"\[(E\d+)\]", md))
    assert cites
    m = re.search(r"var EVIDENCE=(\{.*?\}); var EFFECTS", page, re.S)
    assert m
    ev = json.loads(m.group(1))
    for e in cites:
        assert e in ev and ev[e]["value"] and ev[e]["source"], e
        assert f"data-e='{e}'" in page


def test_page_makes_no_network_call(page):
    """Static demo: one inline script, no external script, no analytics, no reader parameter, no identify call."""
    assert page.count("<script") == 1 and "<script src=" not in page
    for needle in ["posthog", "identify(", "URLSearchParams", "sendBeacon", "fetch(", "XMLHttpRequest", "phc" + "_"]:
        assert needle not in page, needle
    assert "function track(name,props){}" in page
    assert "This page is static. It makes no network call, keeps no cookie, and records nothing about you." in page


def test_banner_and_repo_link(page):
    assert "<p class='banner'>Demo on synthetic data. Built by <a href='https://stevonjudd.com'>Stevon Judd</a>." in page
    assert f"<a href='{site.REPO_URL}'>" in page
    assert "Demo on synthetic data. Built by Stevon Judd." in page
    assert "<title>Monday digest, 2026-05-04 (demo)</title>" in page


def test_no_hardcoded_key_in_repo():
    marker = "phc" + "_"
    for p in list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py")) + [ROOT / "CLAUDE.md", ROOT / ".env.example", ROOT / "README.md"]:
        assert marker not in p.read_text(encoding="utf-8"), p
    assert "ANTHROPIC_API_KEY=\n" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_first_surfaced_rows(page):
    fs = pd.read_csv(ROOT / "outputs" / "first_surfaced.csv", dtype=str)
    assert len(fs) == 5
    assert set(fs["item"].astype(int)) == {1, 2, 3, 4, 5}
    for d in fs["first_monday"]:
        assert d == "this week" or pd.Timestamp(d) <= pd.Timestamp("2026-04-27")
    assert "This would have surfaced on" in page or "First surfaced this week." in page


def test_why_here_terms_use_plain_words(page):
    assert "Size of the move" in page and "How many weeks it has persisted" in page
    assert "Clinical weight" in page and "Confidence from the denominator" in page
    assert "Line it had to clear" in page
    panels = re.findall(r"id='panel-why_here-item\d+' hidden>(.*?)</div>", page, re.S)
    assert len(panels) == 5
    for p in panels:
        for col in ["spike_sd", "drift_sd", "level_sd", "run_weeks_uncapped", "avg8", "long_mean"]:
            assert col not in p, col


def test_print_rules_and_phone_width(page):
    assert "@media print" in page and "#plan_check{page-break-before:always}" in page
    assert "@media (max-width:480px)" in page
    assert "overflow-x:auto" in page


def test_items_load_collapsed_and_plan_table_too(page):
    """Reader first: every numbers block and the plan table carry the hidden attribute on first load."""
    for n in range(1, 6):
        assert f"<div class='panel numbers' id='panel-numbers-item{n}' hidden>" in page, n
        sec = re.search(rf"<section id='item{n}'.*?</section>", page, re.S).group(0)
        for marker in ["<div class='tablewrap'>", "Also moving here:", "Plan AP_", "Why is this here"]:
            assert marker not in sec.split("<div class='panel numbers'")[0], (n, marker)
    assert "<div class='panel plantable' id='panel-plan_table' hidden>" in page
    assert page.count("Show the numbers</button>") == 5 and page.count("Show the plan table</button>") == 1
    assert page.count("Show all the detail</button>") == 1
    script = page.split("<script>", 1)[-1]
    assert "setAll(true)" not in script, "nothing opens every panel on load"


def test_grouped_plan_lines_come_from_plan_check_csv(page):
    df = pd.read_csv(ROOT / "outputs" / "plan_check.csv", dtype=str)
    df = df[df.in_span.astype(str).str.lower() == "true"]
    line = site.grouped_plan_lines(ROOT / "outputs" / "plan_check.csv")
    assert f"<p>{line}</p>" in page
    assert re.match(r"^Close: .+\. Close and re-set: .+\. Challenge: .+\. Too early to judge: .+\.$", line)
    for r in df.itertuples():
        assert f"{r.location_name} " in line and f"({r.plan_id})" in line


def test_hidden_panel_reads_in_plain_words(page):
    """The Show what I hid panel: group counts sum to suppressed.json's total, every metric in the groups is a
    display label, and the raw log is still there, one click deeper, hidden."""
    from src.facts import METRIC_SHORT
    total = json.loads((ROOT / "outputs" / "suppressed.json").read_text(encoding="utf-8"))["count"]
    panel = re.search(r"<div class='panel' id='panel-show_hidden' hidden>(.*?)<div class='panel' id='panel-hidden_log' hidden>", page, re.S)
    assert panel, "plain word panel missing"
    counts = [int(x) for x in re.findall(r"data-count='(\d+)'", panel.group(1))]
    assert counts and sum(counts) == total, (counts, total)
    text = re.sub(r"<[^>]+>", " ", panel.group(1))
    for raw in METRIC_SHORT:
        assert raw not in text, raw
    for label in ["recheck compliance", "client satisfaction", "throughput"]:
        assert label in text, label
    assert "Show the full log with scores</button>" in page
    log = re.search(r"<div class='panel' id='panel-hidden_log' hidden><ul>(.*?)</ul></div>", page, re.S)
    assert log and log.group(1).count("<li>") >= total


def test_how_this_date_was_found_binds_first_surfaced_csv(page):
    """The panel under each item's first-surfaced date: the method, the date, the earlier runs, the caveat."""
    fs = pd.read_csv(ROOT / "outputs" / "first_surfaced.csv", dtype=str).set_index("item")
    panels = re.findall(r"id='panel-first_surfaced-item(\d)' hidden><p>(.*?)</p>", page, re.S)
    assert len(panels) == 5
    for n, text in panels:
        r = fs.loc[n]
        assert text.startswith("How this date was found: the ranking was run again as of each earlier Monday")
        assert "doctor history all stop there" in text and "scored at or above the 3.0 line." in text
        assert text.endswith("One caveat: plan status is a single snapshot, so earlier Mondays use the status recorded today.")
        count = int(r["mondays_above_line_before_latest_week"])
        expect = count > 0 and r["first_monday_ever"] != r["first_monday"]
        sentence = f"It had cleared the line on {count} Mondays in all, the first on {r['first_monday_ever']}, but the earlier runs broke."
        assert (sentence in text) == expect, (n, text)
