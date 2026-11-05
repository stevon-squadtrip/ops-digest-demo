"""Digest validator. Hard fail on any violation; the digest is never rendered with a failing validation.

Rules:
1. every number in the text is a fact display value (ids, ISO dates and E numbers are checked separately)
2. every item cites at least one evidence row once prose exists
3. no hard suppressed metric appears in an item or in One thing today; membership conversion only in Watching and suppressed or the appendix
4. banned words anywhere: diagnose, diagnosis, treatment, prescribe, medication, symptom, dose
5. banned in item 1: because, caused, due to
6. no em dashes anywhere
7. slots: each slot exists, is within its word cap, and references only its allowed facts
8. body cap: the body (above the appendix) stays under the cap with every slot at its cap
"""
import re

from src.write import APPENDIX_MARKER, PLACEHOLDER, already_stated, body, top_block, word_count_excluding_tables

MIN_RESTATE_WORDS = 4


def _norm(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _sentences(s: str) -> list:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]

BANNED_ANYWHERE = ["diagnose", "diagnosis", "treatment", "prescribe", "medication", "symptom", "dose"]
BANNED_ITEM1 = ["because", "caused", "due to"]
EM_DASH = chr(0x2014)
ID_PATTERN = re.compile(r"\b(?:AP|DVM|CL)_\d+\b|\bE\d+\b")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])")
LABEL_LITERALS = ["within 24h", "24h"]
REF_PATTERN = re.compile(r"[\{\[]([a-z0-9_]+)[\}\]]")


class ValidationError(Exception):
    pass


def _center_names(facts: dict) -> list:
    """Center names carry digits here (Clinic 05). They are names, not numbers, so they are removed before any
    digit check. Longest first so Clinic 11 is removed before Clinic 1 could match."""
    names = {e["display"] for fid, e in facts["facts"].items() if fid.endswith("_center") and isinstance(e["display"], str)}
    return sorted(names, key=len, reverse=True)


def _strip_centers(text: str, names: list) -> str:
    for n in names:
        text = text.replace(n, " ")
    return text


def _allowed_numbers(facts: dict) -> set:
    allowed = set()
    for entry in facts["facts"].values():
        for m in NUMBER_PATTERN.findall(entry["display"]):
            allowed.add(m)
    return allowed


def _allowed_dates(facts: dict) -> set:
    dates = set()
    for entry in facts["facts"].values():
        dates |= set(DATE_PATTERN.findall(entry["display"])) | set(DATE_PATTERN.findall(entry["source"]))
    return dates


def _sections(text: str) -> list:
    parts = re.split(r"^## ", text, flags=re.M)
    out = [("preamble", parts[0])]
    for p in parts[1:]:
        heading, _, body_ = p.partition("\n")
        out.append((heading.strip(), body_))
    return out


def _strip_known(text: str) -> str:
    text = ID_PATTERN.sub(" ", text)
    text = DATE_PATTERN.sub(" ", text)
    for lit in LABEL_LITERALS:
        text = text.replace(lit, " ")
    return text


def validate_slots(slots: dict, facts: dict) -> list:
    """Rule 7, plus banned words and the item 1 causal rule inside each slot."""
    violations = []
    spec = facts["digest"]["slots"]
    F = facts["facts"]
    stated = already_stated(facts)
    centers = _center_names(facts)
    for sid, s in spec.items():
        if sid not in slots:
            violations.append(f"slot {sid}: missing")
            continue
        text = slots[sid]
        # Rule 16: never restate a code written line from the same section.
        rendered = re.sub(r"\{([a-z0-9_]+)\}", lambda m: F[m.group(1)]["display"] if m.group(1) in F else m.group(0), text)
        stated_norm = [_norm(line) for line in stated.get(sid, [])]
        for sent in _sentences(rendered):
            ns = _norm(sent)
            if len(ns.split()) < MIN_RESTATE_WORDS:
                continue
            if any(ns == st or ns in st for st in stated_norm):
                violations.append(f"slot {sid}: restates an already stated line: '{sent[:60]}'")
        # Placeholders and citations do not count as words (prompt rule 14).
        words = len(re.sub(r"[\{\[][a-z0-9_]+[\}\]]", "", text).split())
        if words > s["cap"]:
            violations.append(f"slot {sid}: {words} words, cap {s['cap']}")
        for ref in REF_PATTERN.findall(text):
            if ref not in s["allowed_facts"]:
                violations.append(f"slot {sid}: reference {ref} is not in allowed_facts")
        if re.search(r"(?<![\{\[\w])\d", _strip_centers(text, centers)):
            violations.append(f"slot {sid}: a digit outside a placeholder")
        if EM_DASH in text:
            violations.append(f"slot {sid}: em dash")
        low = text.lower()
        for w in BANNED_ANYWHERE:
            if re.search(rf"\b{w}\w*", low):
                violations.append(f"slot {sid}: banned word '{w}'")
        if sid.startswith("item1_"):
            for w in BANNED_ITEM1:
                if re.search(rf"\b{w}\b", low):
                    violations.append(f"slot {sid}: causal word '{w}'")
    for sid in slots:
        if sid not in spec:
            violations.append(f"slot {sid}: unknown slot")
    return violations


def validate(text: str, facts: dict, slots: dict = None) -> list:
    """Return the list of violations. Empty list means the digest may be rendered."""
    violations = []
    allowed = _allowed_numbers(facts)
    allowed_dates = _allowed_dates(facts)
    centers = _center_names(facts)
    D = facts["digest"]
    item_center = {f"Item {it['rank']}": it["center"] for it in D["items"]}

    for i, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            violations.append(f"line {i}: em dash")
    lower = text.lower()
    for w in BANNED_ANYWHERE:
        for m in re.finditer(rf"\b{w}\w*", lower):
            violations.append(f"banned word '{m.group(0)}' at char {m.start()}")

    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("|---"):
            continue
        check = line
        if re.match(r"^\| E\d+ \|", line):
            # Appendix evidence row: only the Value cell is reader text. The Source cell names windows and formulas.
            cells = line.split("|")
            check = cells[3] if len(cells) > 3 else line
        for d in DATE_PATTERN.findall(check):
            if d not in allowed_dates:
                violations.append(f"line {i}: date {d} not in facts")
        stripped = _strip_known(_strip_centers(PLACEHOLDER.sub(" ", check), centers))
        for num in NUMBER_PATTERN.findall(stripped):
            if num not in allowed:
                violations.append(f"line {i}: number {num} not a fact display value")

    suppressed_pairs = facts["facts"].get("suppressed_pairs", {}).get("value", [])
    in_appendix = False
    for heading, body_ in _sections(text):
        if heading.startswith(APPENDIX_MARKER.replace("## ", "")):
            in_appendix = True
        is_item = heading.startswith("Item ")
        is_one_thing = heading.startswith("One thing")
        if is_item and slots is not None:
            if not re.search(r"\[E\d+\]", body_):
                violations.append(f"section '{heading}': no evidence citation")
        if is_item and heading.startswith("Item 1:"):
            b = body_.lower()
            for w in BANNED_ITEM1:
                if re.search(rf"\b{w}\b", b):
                    violations.append(f"item 1: causal word '{w}'")
        if is_item or is_one_thing:
            key = heading.split(":")[0].strip()
            center = item_center.get(key) or facts["facts"]["one_thing_center"]["display"]
            for pair in suppressed_pairs:
                if pair["center"] == center and pair["metric_label"].lower() in body_.lower():
                    violations.append(f"section '{heading}': suppressed metric '{pair['metric_label']}' for {center}")
        if not in_appendix and not heading.startswith("Watching and suppressed") and "membership conversion" in body_.lower():
            violations.append(f"section '{heading}': membership conversion outside the suppressed section")

    top_cap = D.get("top_cap", 10**9)
    if slots is not None:
        violations += validate_slots(slots, facts)
        words = word_count_excluding_tables(body(text))
        if words > D["body_cap"]:
            violations.append(f"body {words} words, cap {D['body_cap']}")
        top = word_count_excluding_tables(top_block(text))
        if top > top_cap:
            violations.append(f"top block {top} words, cap {top_cap}")
    else:
        empty = word_count_excluding_tables(PLACEHOLDER.sub("", body(text)))
        at_cap = empty + sum(s["cap"] for s in D["slots"].values())
        if at_cap > D["body_cap"]:
            violations.append(f"body would be {at_cap} words with every slot at its cap, cap {D['body_cap']}")
        top = word_count_excluding_tables(PLACEHOLDER.sub("", top_block(text))) + D["slots"]["one_thing_today_why"]["cap"]
        if top > top_cap:
            violations.append(f"top block would be {top} words with its slot at cap, cap {top_cap}")
    return violations


def assert_valid(text: str, facts: dict, slots: dict = None) -> None:
    v = validate(text, facts, slots)
    if v:
        raise ValidationError("\n".join(v))
