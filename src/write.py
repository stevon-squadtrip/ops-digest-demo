"""Render the digest from digest_facts.json. Every number is a fact display value.

Milestone 4 part one: the skeleton, with named slots rendered as placeholders.
Part two passes the model's {"slots": {...}} JSON into render(); code inserts each slot,
turns {fact_id} into the display value and [fact_id] into its E number. The model never
touches headings, tables or evidence.
"""
import re
from pathlib import Path

from src.clean import ROOT

OUT_DIR = ROOT / "outputs"
APPENDIX_MARKER = "## Appendix"
PLACEHOLDER = re.compile(r"\[([a-z0-9_]+): up to (\d+) words\]")


def code_sentence(fx: dict, template: str, fids: list, audit: bool = False) -> str:
    """A sentence code writes from facts, with every fact cited by its E number."""
    F = fx["facts"]
    idx = fx["digest"]["evidence_index"]
    text = re.sub(r"\{([a-z0-9_]+)\}", lambda m: (f"{F[m.group(1)]['display']} {{{m.group(1)}}}" if audit else F[m.group(1)]["display"]), template)
    cites = " ".join(f"[{m} {idx[m]}]" if audit else f"[{idx[m]}]" for m in fids if m in idx)
    return f"{text} {cites}".strip()


def already_stated(fx: dict) -> dict:
    """Per slot, the code written lines in its section. The model must not restate these."""
    F = fx["facts"]
    D = fx["digest"]
    v = lambda fid: F[fid]["display"]
    out = {}
    o = D["one_thing_today"]
    out["one_thing_today_why"] = [f"Call {v(o['who_to_call'])} about {v(o['center'])} staffing.",
                                  f"See item {v(o["item_rank"])}. Check back {v(o['check_back'])}."]
    for it in D["items"]:
        n = it["rank"]
        lines = [f"{v(it['headline'])}."]
        if it["also_moving"]:
            lines.append(_also_moving_line(fx, it))
        lines += [_plan_line(fx, pid) for pid in it["plans"]]
        if it.get("code_what"):
            lines.append(code_sentence(fx, it["code_what"]["template"], it["code_what"]["facts"]))
        if it["code_why"]:
            lines.append(code_sentence(fx, it["code_why"]["template"], it["code_why"]["facts"]))
        lines.append(_tail(fx, it))
        for sid in it["slots"]:
            out[sid] = lines
    out["opportunity"] = [_opportunity_lead(fx)]
    return out


def _also_moving_line(fx: dict, it: dict) -> str:
    v = lambda fid: fx["facts"][fid]["display"]
    parts = []
    for s in it["also_moving"]:
        if s["driver"] == "spike":
            parts.append(f"{s['metric_label']} {v(s['last'])} last week, {v('window_8wk')} week average {v(s['avg8'])}")
        elif s["driver"] == "drift":
            parts.append(f"{s['metric_label']} {v(s['avg8'])} on the {v('window_8wk')} week average, long run average {v(s['long_mean'])}")
        else:
            parts.append(f"{s['metric_label']} {v(s['avg8'])} on the {v('window_8wk')} week average, peer median {v(s['peer_median'])}")
    return "Also moving here: " + "; ".join(parts) + "."


def _plan_line(fx: dict, pid: str) -> str:
    F = fx["facts"]
    v = lambda fid: F[fid]["display"]
    p = pid.lower()
    status, label, code = v(f"{p}_status"), v(f"{p}_label"), F[f"{p}_label"]["value"]
    if code == "contradicted":
        reason = f"data shows {v(f'{p}_actual')} against a target of {v(f'{p}_target')}, and {v(f'{p}_trend')}"
    elif code == "mistargeted":
        reason = f"the target of {v(f'{p}_target')} was already met when it opened ({v(f'{p}_baseline_panel')})"
    elif code == "stalled":
        progress = F[f"{p}_progress_pct"]["value"]
        weeks_past_due = F[f"{p}_weeks_past_due"]["value"]
        if weeks_past_due is not None and progress is not None:
            # Past due: say when it was due and how the gap to target compares with the gap at open.
            ago = f"{v(f'{p}_weeks_past_due')} week{'s' if weeks_past_due != 1 else ''} ago"
            if progress == 0:
                gap = "The gap to target is unchanged since the plan opened."
            else:
                word = "wider" if progress < 0 else "narrower"
                gap = f"The gap to target is {v(f'{p}_progress_abs_pct')} percent {word} than when the plan opened."
            return f"Plan {pid} ({label}): marked {status}. Due {v(f'{p}_due')}, {ago}. {gap}"
        if progress is not None and progress < 0:
            reason = f"the gap has widened {v(f'{p}_progress_abs_pct')} percent with {v(f'{p}_window_elapsed_pct')} percent of the time gone"
        else:
            reason = f"{v(f'{p}_progress_pct')} percent of the gap closed with {v(f'{p}_window_elapsed_pct')} percent of the time gone"
    elif code == "achieved":
        reason = f"actual {v(f'{p}_actual')} is past the target of {v(f'{p}_target')}"
    elif code == "early":
        reason = f"opened {v(f'{p}_weeks_open')} weeks ago"
    else:
        reason = f"actual {v(f'{p}_actual')} against a target of {v(f'{p}_target')}"
    return f"Plan {pid} ({label}): marked {status}; {reason}."


def _tail(fx: dict, it: dict) -> str:
    """Whose item it is and when to check back: 'This one is yours (medical). Check back Monday May 11.'
    C65: when the lane and the owner disagree (a medical metric owned by the ROP, or an ops metric owned by the
    RMP), the line names both instead of the lane in parentheses, so a medical tag never reads as a doctor's task."""
    v = lambda fid: fx["facts"][fid]["display"]
    owner = v(it["owner"])
    whose = "yours" if owner == "You" else f"{owner}'s"
    check = f"Check back {v(it['check_back'])}."
    if it["lane"] == "medical" and owner == v("rop_name"):
        return f"This one is {whose}. The metric is medical; the task is operations. {check}"
    if it["lane"] == "ops" and owner == "You":
        return f"This one is {whose}. The metric is operations; the task is medical. {check}"
    return f"This one is {whose} ({it['lane']}). {check}"


def _opportunity_lead(fx: dict) -> str:
    v = lambda fid: fx["facts"][fid]["display"]
    return (f"{v('opp_center')} {v('opp_metric')}: {v('opp_last')} last week, {v('opp_avg8')} over {v('window_8wk')} weeks, "
            f"{v('opp_gap_to_clinic_05')} points above Clinic 05's {v('c05_recheck_avg8')}. Held {v('opp_held_weeks')} of {v('window_6wk')} weeks "
            f"above its own long run average, and above the median of the other {v('n_other_centers')} centers. Needed by {v('opp_needed_by')}.")


def render(fx: dict, slots: dict = None, audit: bool = False) -> str:
    """audit=True keeps fact ids visible next to each value and citation (digest_audit.md)."""
    F = fx["facts"]
    D = fx["digest"]
    idx = D["evidence_index"]
    v = lambda fid: F[fid]["display"]
    src = lambda fid: F[fid]["source"].replace(" | ", " / ")

    def slot(sid):
        spec = D["slots"][sid]
        if slots is None or sid not in slots:
            return f"[{sid}: up to {spec['cap']} words]"
        text = slots[sid]
        if audit:
            text = re.sub(r"\{([a-z0-9_]+)\}", lambda m: f"{v(m.group(1))} {{{m.group(1)}}}" if m.group(1) in F else m.group(0), text)
            text = re.sub(r"\[([a-z0-9_]+)\]", lambda m: f"[{m.group(1)} {idx[m.group(1)]}]" if m.group(1) in idx else m.group(0), text)
        else:
            text = re.sub(r"\{([a-z0-9_]+)\}", lambda m: v(m.group(1)) if m.group(1) in F else m.group(0), text)
            text = re.sub(r"\[([a-z0-9_]+)\]", lambda m: f"[{idx[m.group(1)]}]" if m.group(1) in idx else m.group(0), text)
        return text

    plan_line = lambda pid: _plan_line(fx, pid)

    L = []
    h = D["header"]
    L += [f"# Monday digest: {v(h['leader'])}", "",
          f"Monday {v(h['date'])}. {v(h['centers'])} clinics. Latest complete week starts {v(h['latest_week'])}. "
          f"Ops partner: {v(h['rop'])}.", ""]

    L += ["## At a glance", ""]
    for g in D["at_a_glance"]:
        whose = "Yours" if v(g["owner"]) == "You" else "Ops partner" if v(g["owner"]) == v("rop_name") else v(g["owner"])
        L.append(f"{g['rank']}. {v(g['headline'])}. {whose}. Check back {v(g['check_back_short'])}.")
    L.append("")

    o = D["one_thing_today"]
    # One paragraph: the code line, the accepted slot, then where the detail is and when to check back.
    L += ["## One thing today", "",
          f"Call {v(o['who_to_call'])} about {v(o['center'])} staffing. {slot(o['slot'])} "
          f"See item {v(o["item_rank"])}. Check back {v(o['check_back'])}.", ""]

    for it in D["items"]:
        n = it["rank"]
        L += [f"## Item {n}: {v(it['headline'])}", ""]
        what = slot(f"item{n}_what_happened")
        if it.get("code_what"):
            what += " " + code_sentence(fx, it["code_what"]["template"], it["code_what"]["facts"], audit)
        L.append(f"**What happened.** {what}")
        L.append("")
        if it["tables"].get("weeks"):
            L += ["| Week | Record completion within 24h |", "|---|---|"]
            for wk in ["20260209", "20260223", "20260316", "20260406", "20260427"]:
                L.append(f"| {wk[:4]}-{wk[4:6]}-{wk[6:]} | {v(f'c11_rc_{wk}')} |")
            L.append("")
        if it["tables"].get("doctors"):
            L += ["| Doctor | Record completion, first 8 weeks | Last 8 weeks |", "|---|---|---|"]
            for d in it["tables"]["doctors"]:
                L.append(f"| {d['provider_id']} | {v(d['early'])} | {v(d['late'])} |")
            L.append("")
        if it["tables"].get("peers"):
            L += ["| Doctor | Type | Tenure months | Hours, 8 weeks | Recheck, hours weighted | Recheck, simple mean |", "|---|---|---|---|---|---|"]
            for p in it["tables"]["peers"]:
                L.append(f"| {p['provider_id']} | {v(p['employment'])} | {v(p['tenure'])} | {v(p['hours'])} | {v(p['recheck_hw'])} | {v(p['recheck_mean'])} |")
            L.append("")
        if it["also_moving"]:
            L.append(_also_moving_line(fx, it))
            L.append("")
        for pid in it["plans"]:
            L.append(plan_line(pid))
        if it["plans"]:
            L.append("")
        if it["code_why"]:
            L.append(f"**Why it matters.** {code_sentence(fx, it['code_why']['template'], it['code_why']['facts'], audit)}")
        else:
            L.append(f"**Why it matters.** {slot(f'item{n}_why_it_matters')}")
        L.append("")
        L.append(f"**What to do.** {slot(f'item{n}_what_to_do')} {_tail(fx, it)}")
        L.append("")

    pc = D["plan_check"]
    L += ["## Plan check", "",
          f"Across Northside Clinic Network, {v('plans_met_at_open')} of the {v('plans_at_open_denominator')} plans with data at open had already met "
          f"their target when they opened, and {v('plans_tracking_word')} of the {v('plans_total')} plans track as reported without a flag.", "",
          "| Plan | Center | Metric | Status | Label | Flags | 4 week actual vs target | Holds on 4 week baseline | Action |",
          "|---|---|---|---|---|---|---|---|---|"]
    for pid in pc["plans"]:
        p = pid.lower()
        L.append(f"| {pid} | {v(f'{p}_center')} | {v(f'{p}_metric')} | {v(f'{p}_status')} | {v(f'{p}_label')} | {v(f'{p}_flags')} | "
                 f"{v(f'{p}_actual')} vs {v(f'{p}_target')} | {v(f'{p}_robust')} | {v(f'{p}_action')} |")
    L += ["", f"Close: {v('plans_group_close')}. Close and re-set: {v('plans_group_close_and_reset')}. Challenge: {v('plans_group_challenge')}. "
          f"Too early to judge: {v('plans_group_too_early')}.", ""]

    op = D["opportunity"]
    L += ["## Opportunity", "", _opportunity_lead(fx), "", slot(op["slot"]), ""]

    L += ["## Watching and suppressed", ""]
    for b in D["suppressed"]:
        L.append(f"- {b['lead']}")
    L.append("")

    fb = D["feedback"]
    L += ["## Feedback", "",
          f"Reply {', '.join(fb['verdicts'][:-1])} or {fb['verdicts'][-1]} per line; saved to {fb['file']}.", "",
          "| Item | useful | not_useful | already_handled |", "|---|---|---|---|"]
    for item_id in fb["items"]:
        L.append(f"| {item_id} | [ ] | [ ] | [ ] |")
    L.append(f"| {fb['free_text_row']} | | | |")
    L.append("")

    L += [APPENDIX_MARKER, "", "Evidence rows are numbered. Citations in the text point at these rows.", ""]
    for t in D["evidence_tables"]:
        L += [f"**{t['title']}**", "", "| Row | Fact | Value | Source |", "|---|---|---|---|"]
        for fid in t["facts"]:
            L.append(f"| {idx[fid]} | {fid} | {v(fid)} | {src(fid)} |")
        L.append("")
    L += ["**Other movements**", ""]
    if D["other_movements"]:
        for m in D["other_movements"]:
            L.append(f"- {m['center']} {m['metric_label']}: {v(m['last'])} last week, {v('window_8wk')} week average {v(m['avg8'])}, "
                     f"long run average {v(m['long_mean'])} ({m['direction']}, {m['driver']}).")
    else:
        L.append("- None at or above the supporting line this week.")
    L += ["", "**Data notes**", "",
          f"- {v('plans_baseline_mismatch')} of {v('plans_baseline_denominator')} recorded plan baselines are off by more than that center's normal week to week swing.",
          "- No center level owner exists in the data. Every plan in this span lists the Medical Director. A clinic lead field would let actions route below them.",
          f"- {v('dn_duplicates')} duplicate rows removed from clinic_weekly.",
          f"- {v('dn_negative_waits')} negative wait times treated as missing.",
          f"- Maturity tier recomputed from opened_date: {v('dn_tier_mismatches')} centers differ from the recorded tier ({v('dn_tier_mismatch_centers')}).",
          f"- {v('dn_state_code_errors')} centers carry a state code that disagrees with their metro. Not used by any metric.",
          f"- Doctor FTE target oddities: {v('dn_fte_oddities')}.",
          f"- Clinic 09 callbacks at or above {v('integrity_level')} in {v('c09_cb_weeks_at_ceiling_last10')} of the last {v('window_10wk')} weeks: logged as a data integrity question, not as performance.",
          ""]
    return "\n".join(L)


def body(text: str) -> str:
    return text.split(APPENDIX_MARKER)[0]


def top_block(text: str) -> str:
    """Header, at a glance, one thing today: everything before the first item."""
    return re.split(r"^## Item 1:", text, maxsplit=1, flags=re.M)[0]


def top_words_empty(text: str) -> int:
    return word_count_excluding_tables(PLACEHOLDER.sub("", top_block(text)))


def top_words_at_cap(text: str, fx: dict) -> int:
    return top_words_empty(text) + fx["digest"]["slots"]["one_thing_today_why"]["cap"]


def word_count_excluding_tables(text: str) -> int:
    return sum(len(line.split()) for line in text.splitlines() if not line.lstrip().startswith("|"))


def body_words_empty(text: str) -> int:
    """Body words with every slot placeholder removed."""
    return word_count_excluding_tables(PLACEHOLDER.sub("", body(text)))


def body_words_at_cap(text: str, fx: dict) -> int:
    return body_words_empty(text) + sum(s["cap"] for s in fx["digest"]["slots"].values())


def write_text(text: str, path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path
