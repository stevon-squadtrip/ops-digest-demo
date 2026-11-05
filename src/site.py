"""Build site/index.html, one self contained page, from the accepted digest and the pipeline outputs.

Not an app. The repo is the source of truth; this page is the reader's view. Inputs: outputs/digest.md,
digest_facts.json, ranked.csv, suppressed.json, items.json, first_surfaced.csv, run_log.csv, and the
feedback rules. The page makes no network call of its own: no analytics, no API, no reader tracking.
Feedback clicks show the rule's effect line in the page and are not sent anywhere. Run: python -m src.site
"""
import csv
import html
import json
import math
import os
import re
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

from src import rank
from src.clean import ROOT
from src.facts import METRIC_SHORT
from src.feedback import EFFECT_LINES, VERDICT_LABELS
from src.write import APPENDIX_MARKER

OUT_DIR = ROOT / "outputs"
SITE_DIR = ROOT / "site"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/stevon-squadtrip/ops-digest-demo")
REPO_NAME = re.sub(r"^https?://", "", REPO_URL)
AUTHOR_URL = "https://stevonjudd.com"
BANNER = "Demo on synthetic data. Built by Stevon Judd."
DIGEST_DATE = "2026-05-04"
SECTION_IDS = {"At a glance": "at_a_glance", "One thing today": "one_thing_today", "Plan check": "plan_check",
               "Opportunity": "opportunity", "Watching and suppressed": "watching", "Feedback": "feedback", "Appendix": "appendix"}
# Interaction names the page would report if analytics were wired in. Today track() is a no-op; see JS below.
EVENTS = ["digest_page_opened", "digest_section_viewed", "digest_evidence_expanded", "digest_panel_opened",
          "digest_feedback", "digest_missed", "digest_scroll_depth", "digest_time_on_page"]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# Markdown subset used by the digest: headings, paragraphs, bold leads, tables, lists, citations.
def inline(text: str, cite) -> str:
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # No space before a citation: the superscript is hidden on first load, and the sentence must still end "peers."
    text = re.sub(r"\s*\[(E\d+)\]", lambda m: cite(m.group(1)), text)
    return text


def md_to_html(md: str, cite, section_id_for) -> list:
    """Returns a list of (section_id, heading_text, html) for each '## ' section, plus the preamble."""
    sections, cur_id, cur_head, buf = [], "preamble", "", []
    lines = md.splitlines()
    i = 0

    def flush_para(para):
        if para:
            buf.append(f"<p>{inline(' '.join(para), cite)}</p>")

    para = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            flush_para(para); para = []
            sections.append((cur_id, cur_head, "\n".join(buf)))
            cur_head = line[3:].strip(); cur_id = section_id_for(cur_head); buf = []
            i += 1; continue
        if line.startswith("# "):
            flush_para(para); para = []
            buf.append(f"<h1>{inline(line[2:], cite)}</h1>"); i += 1; continue
        if line.startswith("|"):
            flush_para(para); para = []
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|\s*-", r)]
            head, body = cells[0], cells[1:]
            t = ["<div class='tablewrap'><table><thead><tr>" + "".join(f"<th>{inline(c, cite)}</th>" for c in head) + "</tr></thead><tbody>"]
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c, cite)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            buf.append("".join(t)); continue
        if re.match(r"^\d+\. ", line) or line.startswith("- "):
            flush_para(para); para = []
            ordered = bool(re.match(r"^\d+\. ", line))
            items = []
            while i < len(lines) and (re.match(r"^\d+\. ", lines[i]) or lines[i].startswith("- ")):
                items.append(re.sub(r"^(\d+\. |- )", "", lines[i])); i += 1
            tag = "ol" if ordered else "ul"
            buf.append(f"<{tag}>" + "".join(f"<li>{inline(x, cite)}</li>" for x in items) + f"</{tag}>"); continue
        if line.startswith("**") and line.strip():
            flush_para(para); para = []
            cls = "lead todo" if line.startswith("**What to do.**") else "lead"
            buf.append(f"<p class='{cls}'>{inline(line, cite)}</p>"); i += 1; continue
        if line.startswith("Also moving here:") or re.match(r"^Plan AP_\d+ \(", line):
            # Typography only: supporting lines and plan lines inside an item sit lighter than What to do.
            flush_para(para); para = []
            buf.append(f"<p class='quiet'>{inline(line, cite)}</p>"); i += 1; continue
        if not line.strip():
            flush_para(para); para = []; i += 1; continue
        para.append(line); i += 1
    flush_para(para)
    sections.append((cur_id, cur_head, "\n".join(buf)))
    return sections


def _table_rows(table_html: str) -> list:
    """[(row_html, [cell_html, ...])] for a <table> string built by md_to_html. Cells never nest tables."""
    return [(m.group(0), re.findall(r"<t[dh]>(.*?)</t[dh]>", m.group(0), re.S)) for m in re.finditer(r"<tr>.*?</tr>", table_html, re.S)]


def drop_table_column(section_html: str, header: str) -> str:
    """Remove one column, named by its header cell, from every table in the section that has it. Page view only;
    the markdown and the appendix keep the column."""
    def fix(m):
        table = m.group(0)
        rows = _table_rows(table)
        if not rows or header not in rows[0][1]:
            return table
        idx = rows[0][1].index(header)
        for row_html, cells in rows:
            tag = "th" if row_html.startswith("<tr><th") else "td"
            kept = [c for j, c in enumerate(cells) if j != idx]
            table = table.replace(row_html, "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in kept) + "</tr>", 1)
        return table
    return re.sub(r"<table>.*?</table>", fix, section_html, flags=re.S)


PLAN_DETAIL_COLUMNS = ["Flags", "Holds on 4 week baseline"]


def fold_plan_columns(section_html: str) -> str:
    """Plan check table on the page: seven columns, with Flags and the 4 week baseline check folded into a
    per row detail that opens with the same control the first surfaced panel uses."""
    def fix(m):
        table = m.group(0)
        rows = _table_rows(table)
        if not rows or not all(h in rows[0][1] for h in PLAN_DETAIL_COLUMNS):
            return table
        head = rows[0][1]
        fold = [head.index(h) for h in PLAN_DETAIL_COLUMNS]
        keep = [j for j in range(len(head)) if j not in fold]
        out = "<tr>" + "".join(f"<th>{head[j]}</th>" for j in keep) + "</tr>"
        table = table.replace(rows[0][0], out, 1)
        for row_html, cells in rows[1:]:
            pid = re.sub(r"<[^>]+>", "", cells[0]).strip()
            detail = "; ".join(f"{head[j]}: {cells[j]}" for j in fold)
            first = (f"{cells[0]} <button class='btn small panelbtn' data-panel='plan_detail' data-item='{esc(pid)}' "
                     f"aria-expanded='false'>Details</button>")
            new = ("<tr><td>" + first + "</td>" + "".join(f"<td>{cells[j]}</td>" for j in keep[1:]) + "</tr>"
                   f"<tr class='detail' id='panel-plan_detail-{esc(pid)}' hidden><td colspan='{len(keep)}'>{detail}</td></tr>")
            table = table.replace(row_html, new, 1)
        return table
    return re.sub(r"<table>.*?</table>", fix, section_html, flags=re.S)


DETAIL_BLOCKS = ("<div class='tablewrap'>", "<p class='quiet'>")


def split_item_blocks(body_html: str) -> tuple:
    """(prose blocks, detail blocks). Tables, Also moving here and plan lines are detail; the three lead
    paragraphs stay on first load. Blocks are the newline separated units md_to_html emits."""
    prose, detail = [], []
    for block in body_html.split("\n"):
        (detail if block.startswith(DETAIL_BLOCKS) else prose).append(block)
    return prose, detail


PLAN_GROUPS = [("Close", ("achieved",)), ("Close and re-set", ("mistargeted",)),
               ("Challenge", ("contradicted", "stalled")), ("Too early to judge", ("early",))]


def grouped_plan_lines(plan_check_path: Path) -> str:
    """Close: Clinic 10 throughput (AP_019). ... built from plan_check.csv, in span, by plan id within each group."""
    df = pd.read_csv(plan_check_path, dtype=str)
    df = df[df.in_span.astype(str).str.lower() == "true"].sort_values("plan_id")
    parts = []
    for name, labels in PLAN_GROUPS:
        rows = df[df.label.isin(labels)]
        entries = [f"{r.location_name} {METRIC_SHORT.get(r.target_metric, r.metric_label.lower())} ({r.plan_id})" for r in rows.itertuples()]
        parts.append(f"{name}: {', '.join(entries) if entries else 'none'}.")
    return " ".join(parts)


NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _word(n: int) -> str:
    return NUMBER_WORDS.get(int(n), str(n))


def _join(xs: list) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _num(v) -> str:
    return f"{float(v):.0f}"


def _by_center(rows: list) -> list:
    """[(center, [label, ...])] in first-appearance order, metrics as display labels."""
    out, seen = [], {}
    for r in rows:
        if r["center"] not in seen:
            seen[r["center"]] = []
            out.append((r["center"], seen[r["center"]]))
        seen[r["center"]].append(METRIC_SHORT.get(r["metric"], r["metric"]))
    return out


def hidden_groups(suppressed: list) -> list:
    """The hidden list in plain words. Returns [(sentence, [center line, ...], count)] in the order the page shows
    them. Every number comes from suppressed.json: the row fields or the reason the ranking logged."""
    by = {}
    for r in suppressed:
        by.setdefault(r["reason"].split(":")[0].strip(), []).append(r)
    label = lambda r: METRIC_SHORT.get(r["metric"], r["metric"])
    groups = []

    rows = by.get("watch", [])
    if rows:
        bits = []
        for r in rows:
            need = re.search(r"need (\d+) a week", r["reason"])
            bits.append(f"{r['center']} {label(r)}. {r['latest']} on {_num(r['denominator'])} responses last week, "
                        f"{float(r['value_4wk_weighted']):.2f} on {_num(r['denominator_4wk'])} over four weeks. "
                        f"Needs {need.group(1) if need else '?'} responses a week to count.")
        groups.append(("Watching: " + " ".join(bits), [], len(rows)))

    rows = by.get("watching", [])
    if rows:
        bits = []
        for r in rows:
            run = re.search(r"run (\d+) week", r["reason"])
            need = re.search(r"need a run of (\d+) or (\d+) weeks", r["reason"])
            bits.append(f"{r['center']} {label(r)}. A {_word(run.group(1)) if run else '?'}-week spike with {r['weeks_available']} weeks of history. "
                        f"Needs {_word(need.group(1)) if need else '?'} weeks in a row, or {need.group(2) if need else '?'} weeks of history.")
        groups.append(("Watching at a new center: " + " ".join(bits), [], len(rows)))

    rows = by.get("cap", [])
    if rows:
        top = re.search(r"top (\d+) centers", rows[0]["reason"])
        n = _word(top.group(1)) if top else "?"
        scored = []
        for r in rows:
            s = re.search(r"scored ([\d.]+)", r["reason"])
            scored.append(f"{r['center']} {label(r)} scored {s.group(1) if s else '?'}")
        groups.append((f"At the line but past the {n}-item cap: {_join(scored)}. Only {n} centers get an item.", [], len(rows)))

    rows = by.get("other movements cap", [])
    if rows:
        shown = re.search(r"(\d+) lines shown", rows[0]["reason"])
        n = _word(shown.group(1)) if shown else "?"
        centers = "; ".join(f"{c} {_join(ms)}" for c, ms in _by_center(rows))
        groups.append((f"Moved, but past the {n}-line cap on Other movements ({len(rows)}): {centers}.", [], len(rows)))

    rows = by.get("definition change", [])
    if rows:
        date = re.search(r"changed on (\d{4}-\d{2}-\d{2})", rows[0]["reason"])
        metrics = _join(sorted({label(r) for r in rows}))
        groups.append((f"{metrics[0].upper() + metrics[1:]} at all {len(rows)} centers: the definition changed on "
                       f"{date.group(1) if date else '?'}, so it is not ranked.", [], len(rows)))

    rows = by.get("ramp", [])
    if rows:
        need = re.search(r"need (\d+)", rows[0]["reason"])
        metrics = []
        for r in rows:
            if label(r) not in metrics:
                metrics.append(label(r))
        centers = _join(f"{c} ({rows[[x['center'] for x in rows].index(c)]['weeks_available']} weeks)" for c, _ in _by_center(rows))
        groups.append((f"Too new to judge volume and money ({len(rows)}): {_join(metrics)} at {centers}. Needs {need.group(1) if need else '?'}.", [], len(rows)))

    for key, head in [("below threshold", "Moved, but under the line"),
                      ("favorable", "Moving in the good direction, not the opportunity pick"),
                      ("bundled", "At a center that already has an item above, and scored too low to join it")]:
        rows = by.get(key, [])
        if rows:
            groups.append((f"{head} ({len(rows)}):", [f"{c}: {', '.join(ms)}" for c, ms in _by_center(rows)], len(rows)))
    return groups


def how_this_date_was_found(fsr) -> str:
    """The panel under an item's first-surfaced date. Every number is a column of first_surfaced.csv: the count is
    mondays_above_line_before_latest_week, which counts every prior Monday above the line including the current
    run, so the sentence says "in all". It is dropped when the count is 0, and when every Monday counted belongs
    to the current run (first_monday_ever equals first_monday), because then no earlier run exists to have broken."""
    ever = str(fsr.get("first_monday_ever", fsr["first_monday"]))
    count = int(float(fsr.get("mondays_above_line_before_latest_week", 0) or 0))
    earlier = (f"It had cleared the line on {count} Mondays in all, the first on {esc(ever)}, but the earlier runs broke. "
               if count > 0 and ever != str(fsr["first_monday"]) else "")
    return ("How this date was found: the ranking was run again as of each earlier Monday, using only the data up to that Monday. "
            "Long run averages, baselines, peer medians and doctor history all stop there. The date above is the first Monday of the "
            "unbroken run of weeks, ending last Monday, on which this center and metric scored at or above the "
            f"{rank.ITEM_THRESHOLD:.1f} line. {earlier}"
            "One caveat: plan status is a single snapshot, so earlier Mondays use the status recorded today.")


class _VisibleText(HTMLParser):
    """Text a reader sees on first load: skips script, style, and every element carrying the hidden attribute."""
    SKIP = {"script", "style"}

    def __init__(self):
        super().__init__()
        self.words, self.stack = 0, []

    def handle_starttag(self, tag, attrs):
        hidden = tag in self.SKIP or any(k == "hidden" for k, _ in attrs)
        self.stack.append(hidden)

    def handle_endtag(self, tag):
        if self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not any(self.stack):
            self.words += len(data.split())


def visible_words(page_html: str) -> int:
    p = _VisibleText()
    p.feed(page_html)
    return p.words


WORDS_PER_MINUTE = 200


def read_minutes(page_html: str) -> int:
    return max(1, math.ceil(visible_words(page_html) / WORDS_PER_MINUTE))


def jump_bar(items: list) -> str:
    """In page anchors: Today, the five items by center from items.json, Plans, Hidden."""
    links = [("Today", "one_thing_today")] + [(f"{it['rank']} {it['center']}", f"item{it['rank']}") for it in items] \
            + [("Plans", "plan_check"), ("Hidden", "show_hidden")]
    return "<nav class='jump' aria-label='Jump to'>" + " <span class='sep'>·</span> ".join(
        f"<a href='#{sid}'>{esc(text)}</a>" for text, sid in links) + "</nav>"


def plain_terms(row: pd.Series) -> list:
    """Score terms in plain words, from the formula in src/rank.py: score = weight x confidence x (size + persistence + context)."""
    terms = [
        ("Size of the move", f"{row['size']:.1f} standard deviations (the largest of last week against the 8 week average, "
                             f"the 8 week average against the long run average, and the gap to the peer median; capped at 4)"),
        ("How many weeks it has persisted", f"{int(row['run_weeks_uncapped'])} weeks on the wrong side of its norm, counted as {row['persistence']:.1f} (10 weeks is the most that counts)"),
        ("Clinical weight", f"{row['weight']:.1f} ({'quality metric' if row['weight'] == 1.5 else 'staffing' if row['weight'] == 1.2 else 'wait time or satisfaction' if row['weight'] == 1.0 else 'volume or commercial'})"),
        ("Confidence from the denominator", f"{row['confidence']:.2f} (share of the last 10 weeks with a value; lower when a pooled spread had to be used)"),
    ]
    def text(v):
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
    if row["context"]:
        bits = []
        if text(row["plan_context"]):
            bits.append(f"a plan on this metric is off track ({text(row['plan_context'])})")
        if text(row["provider_context"]):
            bits.append(f"one doctor drives it ({text(row['provider_context'])})")
        terms.append(("Context", f"{row['context']:.0f} added: " + "; ".join(bits)))
    if row["integrity"]:
        terms.append(("Integrity rule", "at or above 99.5 in 6 of the last 10 weeks, so the size is set to 4 and persistence to 0"))
    terms.append(("Score", f"{row['score']:.1f} = {row['weight']:.1f} x {row['confidence']:.2f} x ({row['size']:.1f} + {row['persistence']:.1f} + {row['context']:.0f})"))
    terms.append(("Line it had to clear", f"{rank.ITEM_THRESHOLD:.1f}"))
    return terms


def build(out_path: Path = SITE_DIR / "index.html") -> Path:
    md = (OUT_DIR / "digest.md").read_text(encoding="utf-8")
    fx = json.loads((OUT_DIR / "digest_facts.json").read_text(encoding="utf-8"))
    F, D = fx["facts"], fx["digest"]
    ranked = pd.read_csv(OUT_DIR / "ranked.csv")
    suppressed = json.loads((OUT_DIR / "suppressed.json").read_text(encoding="utf-8"))["suppressed"]
    items = json.loads((OUT_DIR / "items.json").read_text(encoding="utf-8"))["items"]
    fs = pd.read_csv(OUT_DIR / "first_surfaced.csv", dtype=str).set_index("item")
    last_week = []
    if (OUT_DIR / "last_week_items.csv").exists():
        last_week = pd.read_csv(OUT_DIR / "last_week_items.csv", dtype=str).to_dict("records")
    cost = None
    # Leaders across the network, for the yearly cost line: distinct rmp_name in locations.csv.
    n_leaders = int(pd.read_csv(ROOT / "data" / "raw" / "locations.csv").rmp_name.nunique())
    if (OUT_DIR / "run_log.csv").exists():
        log = pd.read_csv(OUT_DIR / "run_log.csv")
        ok = log[(log.ok == True) & (log.tag.fillna("") == "")]
        if len(ok):
            cost = float(ok.iloc[-1].cost_usd)

    e_to_fact = {e: fid for fid, e in D["evidence_index"].items()}
    evidence = {e: {"fact": fid, "value": F[fid]["display"], "source": F[fid]["source"].replace(" | ", " / ")} for e, fid in e_to_fact.items()}
    counter = {"n": 0}

    def cite(e):
        counter["n"] += 1
        return f"<sup class='cite' data-e='{e}' data-fact='{esc(e_to_fact.get(e, ''))}' role='button' tabindex='0'>{e}</sup>"

    item_by_id = {f"item{it['rank']}": it for it in items}

    def section_id_for(head):
        m = re.match(r"Item (\d+):", head)
        if m:
            return f"item{m.group(1)}"
        return SECTION_IDS.get(head, re.sub(r"[^a-z0-9]+", "_", head.lower()).strip("_"))

    body_md, appendix_md = md.split(APPENDIX_MARKER, 1)
    sections = md_to_html(body_md, cite, section_id_for)
    appendix_sections = md_to_html("## Appendix\n" + appendix_md, cite, section_id_for)

    # Hidden pairs, grouped by reason prefix.
    groups = {}
    for s in suppressed:
        key_ = s["reason"].split(":")[0].strip()
        groups.setdefault(key_, []).append(s)
    hidden_count = len(suppressed)

    def feedback_block(item_id):
        btns = "".join(f"<button class='btn fb' data-item='{item_id}' data-verdict='{v}'>{VERDICT_LABELS[v]}</button>" for v in VERDICT_LABELS)
        return f"<div class='feedback' data-item='{item_id}'>{btns}</div>"

    parts = []
    for sid, head, body_html in sections:
        if sid == "preamble":
            parts.append(f"<header id='top'>{body_html}</header>")
            continue
        extra = ""
        if sid == "at_a_glance":
            rows = "".join(
                f"<li><strong>{esc(g)}</strong><ul>" + "".join(f"<li>{esc(s['center'])} {esc(s['metric'])}: {esc(s['reason'])}</li>" for s in lst) + "</ul></li>"
                for g, lst in sorted(groups.items(), key=lambda kv: -len(kv[1]))
            )
            plain = "".join(
                f"<li class='hgroup' data-count='{n}'>{esc(sentence)}" + (("<ul>" + "".join(f"<li>{esc(l)}</li>" for l in lines) + "</ul>") if lines else "") + "</li>"
                for sentence, lines, n in hidden_groups(suppressed))
            extra = (f"<span id='show_hidden'><button class='btn panelbtn' data-panel='show_hidden' data-item='all' aria-expanded='false'>Show what I hid ({hidden_count})</button></span>"
                     f"<p class='hiddencount' hidden>{hidden_count} center and metric pairs were hidden this week. The list and the reason for each are on the live page.</p>"
                     f"<div class='panel' id='panel-show_hidden' hidden><p>Every center and metric considered and not shown this week, with the reason the ranking logged.</p>"
                     f"<ul class='hgroups'>{plain}</ul>"
                     f"<button class='btn small panelbtn' data-panel='hidden_log' data-item='all' aria-expanded='false'>Show the full log with scores</button>"
                     f"<div class='panel' id='panel-hidden_log' hidden><ul>{rows}</ul></div></div>")
            if last_week:
                # Last week's items with a tick on any pair that is no longer above the item line this week.
                strip = " <span class='sep'>·</span> ".join(
                    f"{esc(r['rank'])} {esc(r['center'])} {esc(r['metric_label'])}{' &#10003;' if str(r['still_above_line']).lower() == 'false' else ''}"
                    for r in last_week)
                body_html = body_html + (f"<p class='lastweek'>Last week's items ({esc(last_week[0]['digest_date'])}): {strip}. "
                                         "A tick marks one that is no longer above the line.</p>")
        if sid in item_by_id:
            it = item_by_id[sid]
            r = ranked[(ranked.center == it["center"]) & (ranked.metric == it["lead"]["metric"])].iloc[0]
            terms = "".join(f"<li><strong>{esc(k)}:</strong> {esc(v)}</li>" for k, v in plain_terms(r))
            n = it["rank"]
            fsr = fs.loc[str(n)] if str(n) in fs.index else fs.loc[n]
            if fsr["first_monday"] == "this week":
                surfaced = "First surfaced this week."
            else:
                surfaced = (f"This would have surfaced on {esc(fsr['first_monday'])}, "
                            f"{int(fsr['mondays_before_latest_week'])} Mondays before the latest complete week.")
            how = how_this_date_was_found(fsr)
            extra = (f"<div class='itemtools'><button class='btn panelbtn' data-panel='why_here' data-item='{sid}' aria-expanded='false'>Why is this here</button>"
                     f"<div class='panel' id='panel-why_here-{sid}' hidden><ul>{terms}</ul></div>"
                     f"<p class='surfaced'>{surfaced} <button class='btn small panelbtn' data-panel='first_surfaced' data-item='{sid}' aria-expanded='false'>How this was computed</button></p>"
                     f"<div class='panel' id='panel-first_surfaced-{sid}' hidden><p>{how}</p></div></div>")
            # Page view of the doctor table: the simple mean column stays in the markdown and the appendix only.
            body_html = drop_table_column(body_html, "Recheck, simple mean")
            # Reader first: the three lead paragraphs show on first load. The tools, every table, Also moving here
            # and the plan lines sit in one collapsed block that opens with the same control as the other panels.
            prose, detail = split_item_blocks(body_html)
            numbers = (f"<button class='btn panelbtn' data-panel='numbers' data-item='{sid}' aria-expanded='false'>Show the numbers</button>"
                       f"<div class='panel numbers' id='panel-numbers-{sid}' hidden>{extra}{''.join(detail)}</div>")
            body_html = "".join(prose) + numbers + feedback_block(sid)
            heading = f"<h2>{inline(head, cite)}</h2>"
            parts.append(f"<section id='{sid}' data-section='{sid}'>{heading}{body_html}</section>")
            continue
        if sid == "plan_check":
            body_html = fold_plan_columns(body_html)
            blocks = body_html.split("\n")
            table = next(b for b in blocks if b.startswith("<div class='tablewrap'>"))
            grouped = next(b for b in blocks if b.startswith("<p>Close: "))
            rest = [b for b in blocks if b not in (table, grouped)]
            body_html = ("".join(rest) + f"<p>{esc(grouped_plan_lines(OUT_DIR / 'plan_check.csv'))}</p>"
                         "<button class='btn panelbtn' data-panel='plan_table' data-item='all' aria-expanded='false'>Show the plan table</button>"
                         f"<div class='panel plantable' id='panel-plan_table' hidden>{table}</div>")
        if sid in ("one_thing_today", "opportunity"):
            body_html = body_html + feedback_block(sid)
        if sid == "feedback":
            body_html = (f"<p>Reply {', '.join(VERDICT_LABELS.values())} under each item above. In this demo a reply shows the rule it would "
                         "trigger and is not sent anywhere. In the real digest each reply lands in feedback/feedback.csv and moves the line "
                         "that item has to clear next week.</p>"
                         f"<div class='missed'><label for='missed'>Anything we missed?</label><textarea id='missed' rows='3'></textarea>"
                         f"<button class='btn' id='missed-send'>Send</button><p id='missed-done' hidden>Thanks. In the real digest this is read before next week's run.</p></div>")
        heading = f"<h2>{inline(head, cite)}</h2>"
        parts.append(f"<section id='{sid}' data-section='{sid}'>{heading}{extra}{body_html}</section>")

    # Action first: One thing today sits above At a glance on the page. The markdown keeps its own order.
    order = {"preamble": 0, "one_thing_today": 1, "at_a_glance": 2}
    ids = [re.search(r"id='([^']+)'", p).group(1) if "id='" in p else "" for p in parts]
    ids = ["preamble" if i == "top" else i for i in ids]
    parts = [p for _, p in sorted(zip(ids, parts), key=lambda t: (order.get(t[0], 3), 0))]

    # How it works (optional section, above the footer)
    cost_line = f"What a Monday costs: ${cost:.2f} in model calls for this run (${cost:.4f} exact), at list price." if cost is not None else "What a Monday costs: see outputs/run_log.csv."
    # Yearly cost: this run's model cost, times 52 Mondays, times the leaders, rounded to the dollar.
    annual = round(cost * 52 * n_leaders) if cost is not None else None
    check_lines = ("<li>To check any number: open Show all the detail and click its E number. The appendix names the file, the column, the center and the week.</li>"
                   f"<li>To rerun the numbers: python -m src.run --stage digest --cached, from the repo, no API key. To rewrite the words, drop --cached; "
                   f"this run cost {'$' + format(cost, '.2f') if cost is not None else 'see outputs/run_log.csv'}. A test suite runs before every commit.</li>"
                   f"<li>What it needs each week: the four CSVs. {_word(n_leaders).capitalize()} leaders for a year at this run's rate: "
                   f"{'about $' + str(annual) if annual is not None else 'see outputs/run_log.csv'} in model calls.</li>")
    howitworks = ("<section id='how_it_works' data-section='how_it_works'><h2>How it works</h2><ol>"
                  "<li>Code computes every number, ranks every center and metric, and labels every plan.</li>"
                  "<li>The model writes 16 short passages from placeholders. It never sees a digit.</li>"
                  "<li>A validator rejects any digit not in the facts file, any banned word, and any restated line.</li>"
                  "<li>A person approves the run. The approved run is what you are reading.</li>"
                  f"<li>{esc(cost_line)}</li>{check_lines}</ol>"
                  f"<p>The repository holds the pipeline, the prompts, the tests, the data generator and every output: "
                  f"<a href='{esc(REPO_URL)}'>{esc(REPO_NAME)}</a>.</p>"
                  "<p>This page is static. It makes no network call, keeps no cookie, and records nothing about you.</p></section>")

    # Appendix: evidence tables collapsed under one toggle, other movements and data notes open.
    app_html = "".join(b for _, _, b in appendix_sections)
    app_html = app_html.replace("<h2>Appendix</h2>", "")
    # Split the appendix into the evidence part (bold titles with tables) and the notes part.
    split_at = app_html.find("<p class='lead'><strong>Other movements</strong></p>")
    evidence_html, notes_html = (app_html[:split_at], app_html[split_at:]) if split_at >= 0 else (app_html, "")
    appendix = (f"<section id='appendix' data-section='appendix'><h2>Appendix</h2>"
                f"<button class='btn panelbtn' data-panel='source_all' data-item='all' aria-expanded='false'>Show the source for everything</button>"
                f"<div class='panel evidence-all' id='panel-source_all' hidden>"
                "<p class='legend'>How to read a row: the fact, its value, then where it came from as file / column / center (id) / week. "
                "The CSVs store centers by id. A value computed by the pipeline names the output file and the step that made it.</p>"
                f"{evidence_html}</div>{notes_html}</section>")

    footer = (f"<footer><p>{esc(BANNER)} Every clinic, person and number on this page is generated by "
              f"scripts/make_synthetic_data.py in <a href='{esc(REPO_URL)}'>the repository</a>. No analytics, no cookies.</p></footer>")
    banner = (f"<p class='banner'>Demo on synthetic data. Built by <a href='{esc(AUTHOR_URL)}'>Stevon Judd</a>. "
              f"<a href='{esc(REPO_URL)}'>Source</a>.</p>")

    ev_json = json.dumps(evidence)
    effects_json = json.dumps(EFFECT_LINES)
    css = CSS
    js = JS.replace("__EVIDENCE__", ev_json).replace("__EFFECTS__", effects_json).replace("__DIGEST_DATE__", DIGEST_DATE)
    main_html = "".join(parts) + howitworks + appendix + footer
    # Read time from the words a reader sees on first load, at WORDS_PER_MINUTE, rounded up. Collapsed panels do not count.
    minutes = read_minutes(main_html)
    readtime = (f"<p class='readtime screen'>About {minutes} minutes to read. Sources and hidden items are one click away. "
                f"<button class='btn small' id='detail-all' aria-expanded='false'>Show all the detail</button></p>"
                f"<p class='readtime print' hidden>About {minutes} minutes to read. Sources and hidden items are on the live page.</p>")
    main_html = main_html.replace("</header>", readtime + "</header>" + jump_bar(items), 1)
    main_html = main_html.replace("<header id='top'>", "<header id='top'>" + banner, 1)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<meta name='description' content='{esc(BANNER)} A Monday digest for the medical director of a clinic network.'>"
            f"<title>Monday digest, {DIGEST_DATE} (demo)</title><style>{css}</style></head><body><main>"
            + main_html + "</main>"
            f"<script>{js}</script></body></html>")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(page, encoding="utf-8", newline="\n")
    return out_path


CSS = """
:root{--ink:#1a1a1a;--muted:#555;--line:#ddd;--accent:#1f5f8b;--bg:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.55 Georgia,'Times New Roman',serif}
main{max-width:780px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:30px;line-height:1.2;margin:0 0 12px}
h2{font-size:23px;line-height:1.25;margin:40px 0 10px;border-top:1px solid var(--line);padding-top:18px}
p{margin:10px 0}
p.lead strong{color:var(--ink)}
ol,ul{padding-left:22px}
li{margin:4px 0}
.tablewrap{overflow-x:auto;margin:12px 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-size:15px;min-width:100%}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f4f4f4;font-weight:600}
sup.cite{display:none;font-size:11px;line-height:1;color:var(--accent);cursor:pointer;margin-left:6px;padding:1px 5px;border:1px solid var(--accent);border-radius:9px;font-family:Arial,sans-serif;vertical-align:super;white-space:nowrap}
section.numbers-open sup.cite,body.sources-open sup.cite,body.detail-open sup.cite{display:inline-block}
.panel.numbers,.panel.plantable{background:none;border-color:var(--line)}
.panel.numbers .itemtools{margin-bottom:8px}
#detail-all{margin-left:6px;vertical-align:middle}
sup.cite:hover,sup.cite:focus{background:#e6f0f7;outline:none}
.ev{display:block;margin:6px 0 10px 12px;padding:8px 10px;border-left:3px solid var(--accent);background:#f7fafc;font-size:15px;font-family:Arial,sans-serif;color:var(--muted)}
.btn{font:15px Arial,sans-serif;color:var(--accent);background:#fff;border:1px solid var(--accent);border-radius:6px;padding:6px 12px;margin:6px 8px 6px 0;cursor:pointer}
.btn.small{font-size:14px;padding:3px 8px}
.btn:hover{background:#e6f0f7}
.panel{margin:8px 0 14px;padding:10px 14px;border:1px solid var(--line);border-radius:6px;background:#fafafa;font-size:16px;font-family:Arial,sans-serif}
.panel ul{padding-left:18px}
.hgroups>li{margin:8px 0}.hgroups ul{margin:4px 0 0}
.surfaced{font-family:Arial,sans-serif;font-size:16px;color:var(--muted)}
.feedback{margin:14px 0 6px;padding-top:8px;border-top:1px dashed var(--line);font-family:Arial,sans-serif}
.feedback .effect{font-size:16px;color:var(--muted)}
.missed textarea{width:100%;font:16px Arial,sans-serif;padding:8px;border:1px solid var(--line);border-radius:6px;margin:6px 0}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);font:14px Arial,sans-serif;color:var(--muted)}
#how_it_works{font-size:17px}
.readtime{font-family:Arial,sans-serif;font-size:15px;color:var(--muted);margin:6px 0 0}
.banner{font:14px Arial,sans-serif;color:var(--muted);background:#f4f4f4;border:1px solid var(--line);border-radius:6px;padding:6px 10px;margin:0 0 14px}
.banner a{color:var(--accent)}
.jump{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;margin:12px 0 0;font:14px Arial,sans-serif;white-space:nowrap;overflow-x:auto}
.jump a{color:var(--accent);text-decoration:none}
.jump a:hover{text-decoration:underline}
.jump .sep,.lastweek .sep{color:var(--line)}
.lastweek{font-family:Arial,sans-serif;font-size:15px;color:var(--muted)}
section[id^='item']{margin-top:16px}
section[id^='item'] h2{margin-top:56px;padding-top:24px}
p.quiet{color:var(--muted);font-size:16px}
p.lead.todo{font-weight:600}
p.lead.todo strong{font-weight:700}
tr.detail td{background:#fafafa;font-family:Arial,sans-serif;font-size:14px;color:var(--muted)}
@media (max-width:480px){body{font-size:17px}h1{font-size:26px}h2{font-size:21px}table{font-size:14px}th,td{padding:5px 6px}
 .jump{position:static;white-space:normal;overflow:visible}section[id^='item'] h2{margin-top:44px}}
.readtime.print{display:none}
@media print{
 @page{size:letter;margin:0.6in}
 *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
 body{font-size:12pt}main{max-width:none;padding:0}
 .btn,.feedback,.missed,.surfaced button,.jump,.readtime.screen{display:none !important}
 .banner{background:none;border:none;padding:0}
 .readtime.print[hidden]{display:block !important}
 .panel{display:none !important}
 .panel.numbers[hidden],.panel.plantable[hidden]{display:block !important;border:none;padding:0}
 sup.cite{display:none !important}
 .hiddencount[hidden]{display:block !important}
 .ev{display:none !important}
 sup.cite{color:#000}
 .tablewrap{overflow:visible}table{font-size:9pt;width:100%;table-layout:fixed;word-break:break-word}
 section[id^='item'],.tablewrap,table,tr{break-inside:avoid;page-break-inside:avoid}
 #plan_check{page-break-before:always}
 h2{page-break-after:avoid}
 a[href]:after{content:""}
}
"""

JS = """
(function(){
  var DIGEST_DATE='__DIGEST_DATE__';
  var EVIDENCE=__EVIDENCE__; var EFFECTS=__EFFECTS__;
  // Demo build: no analytics. track() is the one place an analytics call would go; it is a no-op here.
  function track(name,props){}
  track('digest_page_opened',{digest_date:DIGEST_DATE});
  // Citations: tap to open the evidence row inline under the sentence.
  document.querySelectorAll('sup.cite').forEach(function(s){
    function toggle(){var e=s.getAttribute('data-e'); var row=EVIDENCE[e]; var block=s.parentNode.querySelector('.ev[data-e="'+e+'"]');
      if(block){block.remove();return;} if(!row)return;
      var d=document.createElement('span'); d.className='ev'; d.setAttribute('data-e',e);
      d.textContent=e+' '+row.fact+': '+row.value+'. Source: '+row.source; s.parentNode.appendChild(d);
      track('digest_evidence_expanded',{fact_id:row.fact});}
    s.addEventListener('click',toggle); s.addEventListener('keydown',function(ev){if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();toggle();}});
  });
  // Panels: numbers per item, why here, first surfaced, show hidden, plan table, source for everything.
  var PANEL_EVENT='digest_panel_opened';
  function setPanel(b,open){var panel=b.getAttribute('data-panel'), item=b.getAttribute('data-item');
    var id='panel-'+panel+(item&&item!=='all'?'-'+item:''); var p=document.getElementById(id); if(!p)return;
    if(open){p.removeAttribute('hidden');}else{p.setAttribute('hidden','');} b.setAttribute('aria-expanded',open?'true':'false');
    if(panel==='numbers'){var sec=document.getElementById(item); if(sec){sec.classList.toggle('numbers-open',open);}}
    if(panel==='source_all'){document.body.classList.toggle('sources-open',open);}}
  document.querySelectorAll('.panelbtn').forEach(function(b){
    b.addEventListener('click',function(){var open=b.getAttribute('aria-expanded')!=='true'; setPanel(b,open);
      if(open){track(PANEL_EVENT,{panel:b.getAttribute('data-panel'),item:b.getAttribute('data-item')});}});
  });
  // Show all the detail: every numbers block, the plan table and the citations at once, and back.
  var all=document.getElementById('detail-all');
  function setAll(open){document.querySelectorAll(".panelbtn[data-panel='numbers'],.panelbtn[data-panel='plan_table']").forEach(function(b){setPanel(b,open);});
    document.body.classList.toggle('detail-open',open); if(all){all.setAttribute('aria-expanded',open?'true':'false'); all.textContent=open?'Hide the detail':'Show all the detail';}}
  if(all){all.addEventListener('click',function(){var open=all.getAttribute('aria-expanded')!=='true'; setAll(open);
    if(open){track(PANEL_EVENT,{panel:'detail_all',item:'all'});}});}
  // Feedback: one click posts the event and shows the effect from the rule.
  document.querySelectorAll('.feedback').forEach(function(box){
    box.querySelectorAll('.fb').forEach(function(b){b.addEventListener('click',function(){
      var item=box.getAttribute('data-item'), verdict=b.getAttribute('data-verdict');
      track('digest_feedback',{item_id:item,verdict:verdict});
      box.innerHTML='<p class="effect">'+(EFFECTS[verdict]||'Noted.')+'</p>';});});
  });
  var send=document.getElementById('missed-send');
  if(send){send.addEventListener('click',function(){var t=document.getElementById('missed'); var text=(t.value||'').trim().slice(0,2000); if(!text)return;
    track('digest_missed',{text:text}); t.value=''; document.getElementById('missed-done').removeAttribute('hidden');});}
  // Sections in view: half on screen for one second, once each.
  var seen={}, timers={};
  if('IntersectionObserver' in window){var io=new IntersectionObserver(function(entries){entries.forEach(function(en){var id=en.target.getAttribute('data-section');
      if(en.isIntersecting&&en.intersectionRatio>=0.5){if(!seen[id]&&!timers[id]){timers[id]=setTimeout(function(){seen[id]=true;track('digest_section_viewed',{section:id});},1000);}}
      else if(timers[id]){clearTimeout(timers[id]);timers[id]=null;}});},{threshold:[0,0.5,1]});
    document.querySelectorAll('section[data-section]').forEach(function(s){io.observe(s);});}
  // Scroll depth at 25, 50, 75, 100.
  var marks=[25,50,75,100], hit={};
  function depth(){var h=document.documentElement; var pct=Math.round((h.scrollTop+window.innerHeight)/h.scrollHeight*100);
    marks.forEach(function(m){if(pct>=m&&!hit[m]){hit[m]=true;track('digest_scroll_depth',{pct:m});}});}
  window.addEventListener('scroll',depth,{passive:true}); depth();
  // Time on page, reported on pagehide.
  var t0=Date.now();
  window.addEventListener('pagehide',function(){var s=Math.round((Date.now()-t0)/1000); track('digest_time_on_page',{seconds:s});});
})();
"""


if __name__ == "__main__":
    p = build()
    print(f"Wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
