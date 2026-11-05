"""Validator and skeleton tests: the skeleton passes, each injected violation is caught, display rules hold."""
import json
import re
from pathlib import Path

import pytest

from src import facts as facts_mod, rank, validate, write
from src.clean import ROOT, load_and_clean

EXPECTED = json.loads((Path(__file__).parent / "fixtures" / "expected.json").read_text(encoding="utf-8"))
DISPLAY = EXPECTED["display"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    panel = load_and_clean()
    result = rank.run(panel, write=True)
    fx = facts_mod.build_facts(panel, result)
    text = write.render(fx)
    return fx, text, result


def _filled(fx):
    """A minimal valid fill: one sentence per slot citing its first allowed fact."""
    out = {}
    for sid, s in fx["digest"]["slots"].items():
        fid = s["allowed_facts"][0]
        out[sid] = f"At the same time the value is {{{fid}}} [{fid}]."
    return out


def _item(fx, center):
    return [it for it in fx["digest"]["items"] if it["center"] == center][0]


def test_every_fact_has_source_display_kind(built):
    fx, _, _ = built
    assert fx["meta"]["fact_count"] == len(fx["facts"]) > 150
    for fid, entry in fx["facts"].items():
        assert entry["source"] and "|" in entry["source"], fid
        assert "display" in entry and "kind" in entry, fid


def test_display_formatting(built):
    fx, _, _ = built
    F = fx["facts"]
    for fid, want in DISPLAY.items():
        assert F[fid]["display"] == want, fid
    assert re.fullmatch(r"\d+\.\d minutes (shorter|longer)|unchanged", F["c11_duration_change"]["display"])
    assert F["c11_visits_shorter"]["display"] == "no"


def test_skeleton_passes_and_fits_budget(built):
    fx, text, _ = built
    assert validate.validate(text, fx) == []
    assert write.body_words_at_cap(text, fx) <= fx["digest"]["body_cap"]
    assert "[CONNECT" not in text
    assert "(score " not in write.body(text)


def test_filled_digest_passes(built):
    fx, _, _ = built
    slots = _filled(fx)
    text = write.render(fx, slots)
    assert validate.validate(text, fx, slots) == []
    assert "[E" in text and "{" not in write.body(text)
    assert "[item" not in write.body(text) and "[one_thing" not in write.body(text) and "[opportunity" not in write.body(text)


def test_slot_allowed_facts_enforced(built):
    fx, _, _ = built
    slots = _filled(fx)
    slots["item1_what_happened"] = "Scores were {item1_score} [item1_score]."
    v = validate.validate_slots(slots, fx)
    assert any("item1_what_happened" in x and "item1_score" in x for x in v)
    slots = _filled(fx)
    slots["item1_why_it_matters"] = "Clinic 07 call outs were {c07_callouts_last}."
    assert any("c07_callouts_last" in x for x in validate.validate_slots(slots, fx))


def test_audit_only_facts_never_offered(built):
    fx, _, _ = built
    for sid, s in fx["digest"]["slots"].items():
        for fid in s["allowed_facts"]:
            assert not facts_mod.AUDIT_ONLY.search(fid), f"{sid} offers {fid}"


def test_slot_caps_enforced(built):
    fx, _, _ = built
    slots = _filled(fx)
    slots["item1_what_to_do"] = " ".join(["word"] * 16) + " {c11_doctors} [c11_doctors]"
    v = validate.validate_slots(slots, fx)
    assert any("item1_what_to_do" in x and "cap 15" in x for x in v)
    caps = {sid: s["cap"] for sid, s in fx["digest"]["slots"].items()}
    # 16 slots. Plan check and the watching lines are code only; the Clinic 05 item's why it matters is code.
    assert len(caps) == 16
    assert "plan_check" not in caps and not any(s.startswith("suppressed_") for s in caps)
    code_rank = _item(fx, "Clinic 05")["rank"]
    assert f"item{code_rank}_why_it_matters" not in caps
    assert caps["one_thing_today_why"] == 20 and caps["opportunity"] == 25
    for n in range(1, len(fx["digest"]["items"]) + 1):
        assert caps[f"item{n}_what_happened"] == 40 and caps[f"item{n}_what_to_do"] == 15
        if n != code_rank:
            assert caps[f"item{n}_why_it_matters"] == 20


def test_word_counter_excludes_placeholders_and_citations(built):
    # Ten words plus twenty placeholders and citations passes a 15 cap.
    fx, _, _ = built
    slots = _filled(fx)
    fid = fx["digest"]["slots"]["item1_what_to_do"]["allowed_facts"][0]
    slots["item1_what_to_do"] = " ".join(["word"] * 10) + " " + " ".join([f"{{{fid}}} [{fid}]"] * 10)
    v = validate.validate_slots(slots, fx)
    assert not any("item1_what_to_do" in x and "cap" in x for x in v)


def test_already_stated_check(built):
    # A slot may not restate a code written line from its section.
    fx, _, _ = built
    stated = write.already_stated(fx)
    assert stated["item1_what_happened"][0] == DISPLAY["item1_headline"] + "."
    assert any(line.startswith("Plan AP_001") for line in stated["item1_what_to_do"])
    slots = _filled(fx)
    slots["item1_what_happened"] = "Clinic 11: record completion below its norm {c11_rc_run_weeks} weeks running. [c11_rc_run_weeks]"
    v = validate.validate_slots(slots, fx)
    assert any("item1_what_happened" in x and "restates" in x for x in v)
    inp = facts_mod.slot_inputs(fx)
    assert inp["item1_what_happened"]["already_stated"] == stated["item1_what_happened"]


def test_clinic_05_why_is_code(built):
    # Arithmetic restated is never a slot: the counterfactual sentence is written by code.
    fx, text, _ = built
    b = write.body(text)
    F = fx["facts"]
    want = (f"If DVM_0024 matched the other Clinic 05 doctors, the clinic would reach {F['c05_clinic_recheck_if_dvm0024_at_peers']['display']}, "
            f"still {F['c05_gap_to_target_after']['display']} points short of the {F['ap_002_target']['display']} target.")
    assert want in b
    it = _item(fx, "Clinic 05")
    assert it["code_why"] and f"item{it['rank']}_why_it_matters" not in it["slots"]
    section = b.split(f"## Item {it['rank']}")[1].split(f"## Item {it['rank'] + 1}")[0]
    assert "[E" in section
    # AP_019 has no panel row, so the robustness column says so instead of a bool.
    assert re.search(r"\| AP_019 \|.*\| no panel row \|", b)


def test_clinic_07_consequence_facts_and_duration_units(built):
    fx, _, _ = built
    n = _item(fx, "Clinic 07")["rank"]
    allowed = fx["digest"]["slots"][f"item{n}_why_it_matters"]["allowed_facts"]
    for fid in ["c07_appts_last", "c07_appts_avg8", "c07_wait_last", "c07_wait_avg8"]:
        assert fid in allowed
    F = fx["facts"]
    assert F["c11_duration_before"]["display"].endswith(" minutes") and F["c07_wait_last"]["display"].endswith(" minutes")
    p = (ROOT / "prompts" / "digest_prompt.txt").read_text(encoding="utf-8")
    assert "nothing downstream has moved yet" in p and "Never restate a line in already_stated" in p
    assert "long mean" not in write.body(write.render(fx))


def test_budget_caps(built):
    # Body 1,100 words (about 5 minutes at 220 a minute), top block under 150 (under 1 minute).
    fx, text, _ = built
    assert fx["digest"]["body_cap"] == 1100 and fx["digest"]["top_cap"] == 150
    assert write.body_words_at_cap(text, fx) <= 1100
    assert write.top_words_at_cap(text, fx) <= 150


def test_also_moving_outside_noise_band(built):
    # A supporting line shows only when its driving term is at least 1 sd, with the comparison that term used.
    fx, text, result = built
    scores = result["scores"]
    for it in fx["digest"]["items"]:
        for s in it["also_moving"]:
            metric = s["last"].split("_also_")[1].rsplit("_last", 1)[0]
            r = scores[(scores.center == it["center"]) & (scores.metric == metric)].iloc[0]
            term = {"spike": r.spike_sd, "drift": r.drift_sd, "level": r.level_sd}[s["driver"]]
            assert term >= facts_mod.ALSO_MOVING_MIN_SD


def test_plain_word_labels(built):
    # Readers see plain words, outputs keep the code labels.
    fx, text, result = built
    F = fx["facts"]
    assert F["ap_001_label"]["value"] == "contradicted" and F["ap_001_label"]["display"] == "data disagrees"
    assert F["ap_009_label"]["display"] == "target met at open"
    assert F["ap_019_label"]["display"] == "done" and F["ap_027_label"]["display"] == "too soon to judge"
    assert "| contradicted |" not in write.body(text) and "| mistargeted |" not in write.body(text)
    assert set(result["plans"].label) <= {"mistargeted", "achieved", "contradicted", "stalled", "early", "confirmed"}


def test_mistargeted_past_target_closes(built):
    # A mistargeted plan already past target closes; it reads close, not re-set.
    fx, text, result = built
    plans = result["plans"].set_index("plan_id")
    for pid in ["AP_004", "AP_007", "AP_009", "AP_024"]:
        assert plans.loc[pid, "label"] == "mistargeted" and plans.loc[pid, "target_met"]
        assert plans.loc[pid, "action"].startswith("Close it. If the metric still needs to move, re-open with a target set from the panel.")
    assert plans.loc["AP_009", "action"].endswith("Check the linked metric first.")
    assert "Re-set the target" not in write.body(text)
    assert "Close and re-set: AP_004, AP_007, AP_009, AP_024." in text


def test_plan_lines_carry_reasons(built):
    # Templated by label, from facts; progress and elapsed only in these code built lines.
    fx, text, _ = built
    F = fx["facts"]
    b = write.body(text)
    v = lambda k: F[k]["display"]
    assert f"Plan AP_001 (data disagrees): marked on track; data shows {v('ap_001_actual')} against a target of {v('ap_001_target')}, and {v('ap_001_trend')}." in b
    assert f"Plan AP_009 (target met at open): marked complete; the target of {v('ap_009_target')} was already met when it opened ({v('ap_009_baseline_panel')})." in b
    assert f"Plan AP_002 (stalled): marked on track; {v('ap_002_progress_pct')} percent of the gap closed with {v('ap_002_window_elapsed_pct')} percent of the time gone." in b
    assert re.search(r"Plan AP_003 \(stalled\): marked at risk\. Due 2026-04-19, \d+ weeks ago\. The gap to target is \d+ percent (wider|narrower) than when the plan opened\.", b)
    for sid, s in fx["digest"]["slots"].items():
        assert not any(f.endswith("_progress_pct") or f.endswith("_window_elapsed_pct") for f in s["allowed_facts"]), sid


def test_plan_check_lead_holds_at_every_tolerance(built):
    # The lead carries the two numbers that do not move with the band; the mismatch count goes to the appendix.
    fx, text, _ = built
    F = fx["facts"]
    S = EXPECTED["plan_summary"]
    b = write.body(text)
    tracking = "none" if S["plans_tracking_as_reported"] == 0 else str(S["plans_tracking_as_reported"])
    lead = (f"Across Northside, {S['met_at_open']} of the {S['at_open_denominator']} plans with data at open had already met their "
            f"target when they opened, and {tracking} of the {S['plans']} tracks as reported without a flag.")
    assert lead in b
    mismatch = f"{S['baseline_mismatch_flags']} of {S['baseline_mismatch_denominator']}"
    assert mismatch not in b
    appendix = text.split(write.APPENDIX_MARKER)[1]
    assert f"{mismatch} recorded plan baselines are off by more than that center's normal week to week swing." in appendix
    assert "No center level owner exists in the data." in appendix


def test_headlines(built):
    fx, text, _ = built
    F = fx["facts"]
    for n in range(1, 6):
        assert F[f"item{n}_headline"]["display"] == DISPLAY[f"item{n}_headline"]
    assert "weeks running" in DISPLAY["item1_headline"] and "below its norm" in DISPLAY["item1_headline"]
    assert "points below peers; one doctor at" in DISPLAY["item3_headline"]
    assert DISPLAY["item4_headline"].endswith("at the ceiling, verify the logging")
    for it in fx["digest"]["items"]:
        h = F[it["headline"]]["display"]
        assert f"## Item {it['rank']}: {h}" in text
        assert f"{it['rank']}. {h}." in text


def test_clinic_09_required_count_fact(built):
    # The count of required callbacks is not in the data, and the Clinic 09 item may say so.
    fx, _, _ = built
    F = fx["facts"]
    assert F["c09_cb_required_count"]["display"] == "not in the data"
    assert "data/README" in F["c09_cb_required_count"]["source"]
    n = _item(fx, "Clinic 09")["rank"]
    for part in ["what_happened", "why_it_matters", "what_to_do"]:
        assert "c09_cb_required_count" in fx["digest"]["slots"][f"item{n}_{part}"]["allowed_facts"]
    p = (ROOT / "prompts" / "digest_prompt.txt").read_text(encoding="utf-8")
    assert "{c09_cb_required_count}" in p and "pull the callback log for one week" in p


def test_slot_digit_and_causal_rules(built):
    fx, _, _ = built
    slots = _filled(fx)
    slots["item1_why_it_matters"] = "Records fell because 9 doctors slid [c11_doctors]."
    v = validate.validate_slots(slots, fx)
    assert any("causal" in x for x in v) and any("digit" in x for x in v)


def test_wrong_number_is_caught(built):
    fx, text, _ = built
    bad = text.replace("## One thing today", "## One thing today\n\nCall outs were 9999 last week.")
    assert any("9999" in x for x in validate.validate(bad, fx))


def test_banned_words_and_em_dash_are_caught(built):
    fx, text, _ = built
    for word in ["treatment", "diagnosis", "medication", "prescribe", "diagnose", "symptom", "dose"]:
        assert any(word in x for x in validate.validate(text + f"\n\nThe {word} was fine.\n", fx)), word
    assert any("em dash" in x for x in validate.validate(text + "\n\nA line " + chr(0x2014) + " here.\n", fx))


def test_suppressed_metric_outside_its_section_is_caught(built):
    fx, text, _ = built
    n = _item(fx, "Clinic 07")["rank"]
    bad = text.replace(f"## Item {n}:", f"## Item {n}: Clinic 07 X\n\nClient satisfaction fell last week.\n\n## Item {n}b:", 1)
    assert any("suppressed metric" in x for x in validate.validate(bad, fx))
    bad2 = text.replace("## Item 2:", "## Item 2: X\n\nMembership conversion fell.\n\n## Item 2b:", 1)
    assert any("membership conversion" in x for x in validate.validate(bad2, fx))


def test_second_person_owner(built):
    fx, text, _ = built
    F = fx["facts"]
    assert F["ap_001_owner"]["display"] == "You" and F["ap_001_owner"]["value"] == fx["meta"]["leader"]
    assert F["item1_owner"]["display"] == "You"
    assert F[f"item{_item(fx, 'Clinic 05')['rank']}_owner"]["display"] == "You"
    assert F[f"item{_item(fx, 'Clinic 06')['rank']}_owner"]["display"] == "Regional Ops Partner, North"
    b = write.body(text)
    assert "This one is yours (medical). Check back Monday May 11." in b and "This one is yours (medical). Check back Monday May 18." in b
    assert "Update the status to match the data, or flag the data as wrong." in text


def test_at_a_glance_headlines(built):
    fx, text, _ = built
    F = fx["facts"]
    assert F["item1_headline"]["display"] == DISPLAY["item1_headline"]
    for it in fx["digest"]["items"]:
        h = F[it["headline"]]["display"]
        assert f"## Item {it['rank']}: {h}" in text
        assert f"{it['rank']}. {h}." in text
    assert "# Monday digest: Medical Director, North" in text
    assert "Ops partner: Regional Ops Partner, North." in text


def test_one_thing_today_rule(built):
    fx, text, _ = built
    assert "same day call" in facts_mod.ONE_THING_RULE
    assert fx["facts"]["one_thing_center"]["display"] == DISPLAY["one_thing_center"]
    n = DISPLAY["one_thing_item_rank"]
    assert f"See item {n}." in text
    assert "Call Regional Ops Partner, North about Clinic 06 staffing." in text


def test_plans_appear_once_in_body(built):
    fx, text, _ = built
    b = write.body(text)
    assert b.count("| AP_001 |") == 1
    assert "Holds on 4 week baseline" in b and "4 week actual vs target" in b


def test_appendix_and_evidence_numbering(built):
    fx, text, _ = built
    idx = fx["digest"]["evidence_index"]
    assert set(idx.values()) == {f"E{i}" for i in range(1, len(idx) + 1)}
    for fid, e in idx.items():
        assert f"| {e} | {fid} |" in text
    appendix = text.split(write.APPENDIX_MARKER)[1]
    assert "**Other movements**" in appendix and "**Data notes**" in appendix
    assert "**Data notes**" not in write.body(text)


def test_feedback_block(built):
    fx, text, _ = built
    assert "Anything we missed? (free text)" in text
    assert "feedback/feedback.csv" in write.body(text)
    assert fx["digest"]["feedback"]["columns"] == ["digest_date", "item_id", "verdict", "note"]


def test_prompt_shape():
    p = (ROOT / "prompts" / "digest_prompt.txt").read_text(encoding="utf-8")
    assert '{"slots": {"<slot_id>": "<prose>", ...}}' in p
    assert "JSON only" in p
    assert chr(0x2014) not in p


def test_slot_inputs_only_carry_allowed_facts(built):
    fx, _, _ = built
    inp = facts_mod.slot_inputs(fx)
    for sid, s in fx["digest"]["slots"].items():
        assert set(inp[sid]["facts"]) == set(s["allowed_facts"])
        assert inp[sid]["cap_words"] == s["cap"]


def test_item_tails_end_with_check_back_weekday_month_day():
    """Every item tail code writes ends 'Check back <weekday> <month> <day>.' with the lane in parentheses before it,
    or, when lane and owner disagree, the two-lane sentence instead."""
    text = (ROOT / "outputs" / "digest.md").read_text(encoding="utf-8")
    tails = re.findall(r"^\*\*What to do\.\*\* .*$", text, re.M)
    assert len(tails) == 5
    days = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for tail in tails:
        assert re.search(rf"This one is (yours|.+?'s)( \((medical|ops)\)|\. The metric is (medical|operations); the task is (operations|medical))\. "
                         rf"Check back ({days}) ({months}) \d{{1,2}}\.$", tail), tail
    # Clinic 09 is a medical metric owned by the ops partner, so its tail names both lanes.
    assert "This one is Regional Ops Partner, North's. The metric is medical; the task is operations. Check back Monday May 11." in text
    assert "Regional Ops Partner, North's (medical)" not in text
