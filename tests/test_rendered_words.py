"""The page is read as words, not as structure. This builds the site, strips the markup, and fails on any leaked
value, placeholder or em dash, then checks the lines that are built from the output files."""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from src import site
from src.clean import ROOT

LEAKS = [r"\bnan\b", r"\bNone\b", r"\(nan\)", r"\binf\b", r"\bNaN\b", r"\bnull\b"]
PLACEHOLDER = re.compile(r"\{[A-Za-z0-9_]*\}")
EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected.json").read_text(encoding="utf-8"))


def visible_text(html: str) -> str:
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"<[^>]+>", " ", html)


@pytest.fixture(scope="module")
def page_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("site") / "index.html"
    site.build(out)
    return out.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def page_text(page_html):
    return visible_text(page_html)


def test_site_text_has_no_leaked_values(page_text):
    for pat in LEAKS:
        hits = [m.group(0) for m in re.finditer(pat, page_text)]
        assert not hits, f"{pat}: {hits[:5]}"
    assert not PLACEHOLDER.search(page_text), PLACEHOLDER.search(page_text).group(0)
    assert chr(0x2014) not in page_text


def test_digest_markdown_has_no_leaked_values():
    text = (ROOT / "outputs" / "digest.md").read_text(encoding="utf-8")
    for pat in LEAKS:
        assert not re.search(pat, text), pat
    assert not PLACEHOLDER.search(text), PLACEHOLDER.search(text).group(0)


def prose_without_code(md: str) -> str:
    return re.sub(r"```.*?```", " ", md, flags=re.S)


@pytest.mark.parametrize("name", ["README.md", "CLAUDE.md", "data/README.md"])
def test_documents_have_no_leaked_values(name):
    text = prose_without_code((ROOT / name).read_text(encoding="utf-8"))
    for pat in LEAKS:
        hits = [m.group(0) for m in re.finditer(pat, text)]
        assert not hits, f"{name} {pat}: {hits[:5]}"
    assert not PLACEHOLDER.search(text), PLACEHOLDER.search(text).group(0)
    assert chr(0x2014) not in text, f"{name} has an em dash"


def test_readme_is_under_300_words():
    text = prose_without_code((ROOT / "README.md").read_text(encoding="utf-8"))
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    n = len(re.findall(r"\S+", text))
    assert 0 < n <= 300, f"README is {n} words"


def test_read_time_line_is_sane(page_text):
    m = re.search(r"About (\d+) minutes to read\. Sources and hidden items are one click away\.", page_text)
    assert m, "read time line missing"
    n = int(m.group(1))
    assert 1 <= n <= 15, f"read time says {n} minutes"


def test_jump_bar_names_come_from_items_json(page_text):
    items = json.loads((ROOT / "outputs" / "items.json").read_text(encoding="utf-8"))["items"]
    labels = ["Today"] + [f"{it['rank']} {it['center']}" for it in items] + ["Plans", "Hidden"]
    bar = " · ".join(labels)
    flat = re.sub(r"\s+", " ", page_text)
    assert bar in flat, bar


def test_page_tables_show_the_short_columns(page_text):
    flat = re.sub(r"\s+", " ", page_text)
    assert "Recheck, hours weighted" in flat and "Recheck, simple mean" not in flat.split("Appendix")[0]
    assert "Plan Center Metric Status Label 4 week actual vs target Action" in flat
    assert "Flags:" in flat and "Holds on 4 week baseline:" in flat, "the folded columns live in the row detail"


def test_last_week_strip_comes_from_the_csv(page_text):
    rows = pd.read_csv(ROOT / "outputs" / "last_week_items.csv", dtype=str).to_dict("records")
    assert 1 <= len(rows) <= 5, "last week carried at most five items above the line"
    flat = re.sub(r"\s+", " ", page_text)
    parts = [f"{r['rank']} {r['center']} {r['metric_label']}" + (" ✓" if r["still_above_line"].lower() == "false" else "") for r in rows]
    strip = f"Last week's items ({rows[0]['digest_date']}): " + " · ".join(parts) + ". A tick marks one that is no longer above the line."
    assert strip in flat, strip


def test_site_shows_the_plan_lead_and_the_mondays_label(page_text):
    S = EXPECTED["plan_summary"]
    assert f"{S['met_at_open']} of the {S['at_open_denominator']} plans with data at open" in page_text
    assert "Mondays before the latest complete week" in page_text
    assert "weeks before today" not in page_text
    assert "This page is static." in page_text


def test_read_time_line_has_a_print_only_twin(page_html):
    """The print stylesheet swaps the read time line for one that points at the live page. Each appears once, and the
    print twin carries the hidden attribute so a normal browser, a screen reader and a text extractor see one line."""
    assert "<p class='readtime print' hidden>" in page_html
    screen = re.findall(r"About \d+ minutes to read\. Sources and hidden items are one click away\.", page_html)
    printed = re.findall(r"About \d+ minutes to read\. Sources and hidden items are on the live page\.", page_html)
    assert len(screen) == 1, screen
    assert len(printed) == 1, printed
    n = json.loads((ROOT / "outputs" / "suppressed.json").read_text(encoding="utf-8"))["count"]
    counts = re.findall(r"(\d+) center and metric pairs were hidden this week\. The list and the reason for each are on the live page\.", page_html)
    assert counts == [str(n)], counts


class _FirstLoad(HTMLParser):
    """Text a reader sees on first load: skips script, style, elements with the hidden attribute, and the citation
    superscripts, which the stylesheet hides until a block is opened."""

    def __init__(self):
        super().__init__()
        self.parts, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.stack.append(tag in ("script", "style") or "hidden" in a or (tag == "sup" and a.get("class") == "cite"))

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not any(self.stack):
            self.parts.append(data)


def first_load_text(html: str) -> str:
    p = _FirstLoad()
    p.feed(html)
    return re.sub(r"\s+", " ", "".join(p.parts))


def test_first_load_text_has_no_space_before_punctuation(page_html):
    text = first_load_text(page_html)
    hits = re.findall(r".{0,30} [.,;].{0,10}", text)
    assert not hits, hits[:5]
