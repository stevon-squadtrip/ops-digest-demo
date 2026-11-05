"""Milestone 4 part two and three: turn the slot inputs into prose with one model call, validate, render.

The model receives prompts/digest_prompt.txt as the system message and the slot JSON as the user
message. It returns {"slots": {...}}. Code validates the slots and the rendered digest. On a failed
validation it retries with the violation list appended. Three failures stop the run.
Every generation appends a row to outputs/run_log.csv with tokens, attempts, wall time and cost.
"""
import csv
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from src import facts as facts_mod, validate, write
from src.clean import ROOT

OUT_DIR = ROOT / "outputs"
PROMPT_PATH = ROOT / "prompts" / "digest_prompt.txt"
RUN_LOG = OUT_DIR / "run_log.csv"
# Model config. Override with ANTHROPIC_MODEL in .env. claude-opus-5-5 rejects sampling parameters
# (temperature, top_p, top_k), so none are sent; effort is pinned low for a short fill.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("ANTHROPIC_EFFORT", "low")
# --smoke: the standing rule's yardstick run. Cheap model, never accepted. The requested id is tried first;
# if the API does not know it, the current Haiku from the API reference is used and the log says so.
SMOKE_MODELS = ["claude-sonnet-5-5", "claude-haiku-4-5"]
MAX_TOKENS = 16000
MAX_ATTEMPTS = 3
# List price, US dollars per million tokens (Anthropic API, cached 2026-09-25 in the API reference).
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
                  "claude-haiku-5-5": (1.0, 5.0)}


def parse_slots(text: str) -> dict:
    """Accept bare JSON or JSON inside a fence. Anything else is a validation failure."""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.S)
    if m:
        s = m.group(1)
    data = json.loads(s)
    if not isinstance(data, dict) or "slots" not in data or not isinstance(data["slots"], dict):
        raise ValueError('reply is not {"slots": {...}}')
    return data["slots"]


def call_model(client, system: str, user: str, model: str = None) -> tuple:
    """One request. Returns (reply_text, usage dict). Fallback-on-refusal is enabled per the API guide."""
    model = model or MODEL
    kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}])
    if model != "claude-haiku-4-5":  # Haiku 4.5 rejects effort; every current model, Haiku 5.5 included, takes it
        kwargs["output_config"] = {"effort": EFFORT}
    # Streamed, because a long non streaming request can be dropped by the connection before the reply arrives.
    path = "beta with fallback, streamed"
    try:
        with client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs) as s:
            resp = s.get_final_message()
    except (TypeError, anthropic.BadRequestError):
        path = "plain, streamed"
        with client.messages.stream(**kwargs) as s:
            resp = s.get_final_message()
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason == "max_tokens" and not text.strip():
        text = ""  # the validator reports this as invalid JSON; the log keeps stop_reason
    usage = {"model": resp.model, "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "stop_reason": resp.stop_reason, "request_id": getattr(resp, "_request_id", None), "path": path}
    return text, usage


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["claude-opus-5-5"])
    return round(tokens_in / 1e6 * pin + tokens_out / 1e6 * pout, 4)


def log_run(row: dict) -> None:
    new = not RUN_LOG.exists()
    with open(RUN_LOG, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp_utc", "tag", "model", "effort", "attempts", "tokens_in", "tokens_out",
                                          "wall_seconds", "cost_usd", "ok"])
        if new:
            w.writeheader()
        w.writerow(row)


def generate(fx: dict, client=None, tag: str = "", smoke: bool = False) -> dict:
    """Run the generation loop. Returns a dict with slots, texts, violations per attempt, and paths."""
    client = client or anthropic.Anthropic()
    system = PROMPT_PATH.read_text(encoding="utf-8")
    inputs = facts_mod.slot_inputs(fx)
    user = json.dumps(inputs, indent=1)
    model = MODEL
    if smoke:
        tag = tag or "smoke"
        model = SMOKE_MODELS[0]
        try:
            client.models.retrieve(model)
        except anthropic.NotFoundError:
            model = SMOKE_MODELS[1]
    log = {"model": model, "effort": EFFORT, "smoke": smoke, "attempts": []}
    slots = None
    t0 = time.time()
    tokens_in = tokens_out = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply, usage = call_model(client, system, user, model=model)
        tokens_in += usage["input_tokens"]
        tokens_out += usage["output_tokens"]
        entry = {"attempt": attempt, "usage": usage, "raw": reply}
        try:
            candidate = parse_slots(reply)
            text = write.render(fx, candidate)
            problems = validate.validate(text, fx, candidate)
        except (ValueError, json.JSONDecodeError) as e:
            candidate, problems = None, [f"reply is not valid JSON: {e}"]
        entry["violations"] = problems
        log["attempts"].append(entry)
        if not problems:
            slots = candidate
            break
        user = json.dumps(inputs, indent=1) + "\n\nYour previous reply was rejected. Fix every violation below and return the full JSON again.\n" \
            + "\n".join(f"- {p}" for p in problems)
    wall = round(time.time() - t0, 1)
    cost = cost_usd(model, tokens_in, tokens_out)
    log.update({"tokens_in": tokens_in, "tokens_out": tokens_out, "wall_seconds": wall, "cost_usd": cost})
    log_run({"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "model": model,
             "effort": EFFORT, "attempts": len(log["attempts"]), "tokens_in": tokens_in, "tokens_out": tokens_out,
             "wall_seconds": wall, "cost_usd": cost, "ok": slots is not None})
    paths = {}
    suffix = f"_{tag}" if tag else ""
    paths["log"] = OUT_DIR / f"digest_generation_log{suffix}.json"
    with open(paths["log"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(log, f, indent=1)
    if slots is None:
        return {"ok": False, "log": log, "paths": paths}
    paths["response"] = OUT_DIR / f"digest_model_response{suffix}.json"
    with open(paths["response"], "w", encoding="utf-8", newline="\n") as f:
        json.dump({"slots": slots}, f, indent=1)
    reader = write.render(fx, slots)
    audit = write.render(fx, slots, audit=True)
    paths["digest"] = write.write_text(reader, OUT_DIR / f"digest{suffix}.md")
    paths["audit"] = write.write_text(audit, OUT_DIR / f"digest_audit{suffix}.md")
    final_problems = validate.validate(reader, fx, slots)
    return {"ok": not final_problems, "slots": slots, "digest": reader, "audit": audit, "violations": final_problems,
            "log": log, "paths": paths,
            "words": {"body": write.word_count_excluding_tables(write.body(reader)),
                      "top": write.word_count_excluding_tables(write.top_block(reader))}}


def accept(paths: dict) -> dict:
    """Store the accepted run so --cached can serve it without an API call."""
    out = {"digest": OUT_DIR / "digest_accepted.md", "slots": OUT_DIR / "digest_accepted_slots.json"}
    shutil.copyfile(paths["digest"], out["digest"])
    shutil.copyfile(paths["response"], out["slots"])
    return out


def render_cached(fx: dict) -> dict:
    """Render digest.md and digest_audit.md from the accepted slots against the current facts. No API call."""
    slots = json.loads((OUT_DIR / "digest_accepted_slots.json").read_text(encoding="utf-8"))["slots"]
    reader = write.render(fx, slots)
    audit = write.render(fx, slots, audit=True)
    problems = validate.validate(reader, fx, slots)
    paths = {"digest": write.write_text(reader, OUT_DIR / "digest.md"), "audit": write.write_text(audit, OUT_DIR / "digest_audit.md")}
    if not problems:
        # The accepted slots are the stored artifact; the accepted markdown is always their latest valid render.
        paths["accepted"] = write.write_text(reader, OUT_DIR / "digest_accepted.md")
    return {"ok": not problems, "violations": problems, "paths": paths, "digest": reader,
            "words": {"body": write.word_count_excluding_tables(write.body(reader)),
                      "top": write.word_count_excluding_tables(write.top_block(reader))}}
