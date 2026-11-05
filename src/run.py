"""One command runs the pipeline: python -m src.run --date 2026-05-04

Stages build on each other. --stage clean stops after cleaning and prints what it did.
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.clean import DEFAULT_DATE, DEFAULT_LEADER, ROOT, load_and_clean
from src.metrics import long_panel

OUT_DIR = ROOT / "outputs"
STAGES = ["clean", "rank", "plans", "digest", "all"]


def load_secrets() -> int:
    """Load .env from the repo root. Returns the key length, never the key. 0 means not set."""
    load_dotenv(ROOT / ".env")
    return len(os.environ.get("ANTHROPIC_API_KEY", ""))


def print_clean_summary(panel) -> None:
    print(f"Today {panel.today.date()}. Latest complete week starts {panel.latest_week.date()}.")
    print(f"Leader {panel.leader}: {len(panel.my_ids)} centers.")
    print()
    print("Row counts after cleaning")
    print(f"  locations        {len(panel.locations):>6}")
    print(f"  clinic_weekly    {len(panel.clinic):>6}  (leader's centers: {len(panel.my_clinic)})")
    print(f"  provider_weekly  {len(panel.providers):>6}  (leader's centers: {len(panel.my_providers)})")
    print(f"  action_plans     {len(panel.plans):>6}  (leader's centers: {len(panel.my_plans)})")
    print()
    print("Cleaning notes")
    for n in panel.notes:
        print(f"  {n['note']}: {n['count']}. {n['detail']}")
    print()
    print("Maturity tier mismatches (recomputed against today, reported value kept)")
    if panel.tier_mismatches.empty:
        print("  none")
    for r in panel.tier_mismatches.itertuples():
        print(f"  {r.location_name:<15} opened {r.opened_date.date()}  {r.months_open:>5} months  "
              f"reported {r.maturity_tier_reported:<8} recomputed {r.maturity_tier}")


def write_data_notes(panel, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "data_notes.json"
    payload = {
        "today": str(panel.today.date()),
        "latest_week": str(panel.latest_week.date()),
        "leader": panel.leader,
        "notes": panel.notes,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Northside Monday digest pipeline")
    ap.add_argument("--date", default=DEFAULT_DATE, help="the Monday the digest is for")
    ap.add_argument("--leader", default=DEFAULT_LEADER, help="rmp_name whose span to run")
    ap.add_argument("--stage", default="all", choices=STAGES)
    ap.add_argument("--generate", action="store_true", help="call the model to fill the slots (Milestone 4 part two)")
    ap.add_argument("--cached", action="store_true", help="render digest.md from the accepted slots, no API call")
    ap.add_argument("--smoke", action="store_true", help="one cheap generation to test the yardstick; never accepted")
    ap.add_argument("--tag", default="", help="suffix for generated files, for repeat runs")
    args = ap.parse_args(argv)

    key_len = load_secrets()
    print(f"ANTHROPIC_API_KEY loaded from .env: {'yes' if key_len else 'no'} (length {key_len}).")

    panel = load_and_clean(today=args.date, leader=args.leader)
    print_clean_summary(panel)
    path = write_data_notes(panel)
    print(f"\nWrote {path.relative_to(ROOT)}")
    if args.stage == "clean":
        return

    metrics = long_panel(panel)
    print(f"Metric panel: {len(metrics)} rows, {metrics.metric.nunique()} metrics, "
          f"{metrics.location_id.nunique()} centers.")

    from src import rank
    result = rank.run(panel)
    print_rank_summary(result)
    for name, path in result["paths"].items():
        print(f"Wrote {path.relative_to(ROOT)}")
    if args.stage == "rank":
        return

    from src import facts as facts_mod, validate, write
    fx = facts_mod.build_facts(panel, result)
    fpath = facts_mod.write_facts(fx)
    text = write.render(fx)
    problems = validate.validate(text, fx)
    print(f"\nDigest facts: {fx['meta']['fact_count']} facts, {fx['meta']['slot_count']} slots. Wrote {fpath.relative_to(ROOT)}")
    if problems:
        print(f"Skeleton failed validation with {len(problems)} problems.")
        for p in problems:
            print(f"  {p}")
        if all(p.startswith("body would be") for p in problems):
            # Budget only: write the skeleton so the structure can be reviewed, but fail the run.
            write.write_text(text, OUT_DIR / "digest_skeleton.md")
            print("Written for review only (budget violation). Not valid for rendering.")
        raise SystemExit(1)
    spath = write.write_text(text, OUT_DIR / "digest_skeleton.md")
    print(f"Skeleton validated: 0 problems. Body words with slots empty {write.body_words_empty(text)}, "
          f"with every slot at its cap {write.body_words_at_cap(text, fx)}, cap {fx['digest']['body_cap']}. "
          f"Top block {write.top_words_empty(text)} empty, {write.top_words_at_cap(text, fx)} at cap, cap {fx['digest']['top_cap']}. "
          f"Wrote {spath.relative_to(ROOT)}")
    from src import generate as gen
    if args.cached:
        out = gen.render_cached(fx)
        if not out["ok"]:
            print("Cached digest no longer validates against the current facts:")
            for p in out["violations"]:
                print(f"  {p}")
            raise SystemExit(1)
        print(f"Rendered {out['paths']['digest'].relative_to(ROOT)} from the accepted run. Body {out['words']['body']} words, "
              f"top block {out['words']['top']} words. Validator: 0 problems. No API call.")
        return
    if not (args.generate or args.smoke):
        return
    if not key_len:
        print("ANTHROPIC_API_KEY is not set. Stopping before generation.")
        raise SystemExit(2)
    out = gen.generate(fx, tag=args.tag, smoke=args.smoke)
    if args.smoke:
        print(f"Smoke model: {out['log']['model']}")
    for a in out["log"]["attempts"]:
        print(f"Attempt {a['attempt']}: {a['usage']['model']}, in {a['usage']['input_tokens']} out {a['usage']['output_tokens']}, "
              f"{len(a['violations'])} violations")
        for p in a["violations"]:
            print(f"  {p}")
    if not out["ok"]:
        print("Generation failed validation twice. Nothing rendered.")
        raise SystemExit(1)
    print(f"Rendered {out['paths']['digest'].relative_to(ROOT)} and {out['paths']['audit'].relative_to(ROOT)}. "
          f"Body {out['words']['body']} words, top block {out['words']['top']} words. Validator: 0 problems. "
          f"Tokens in {out['log']['tokens_in']}, out {out['log']['tokens_out']}, wall {out['log']['wall_seconds']}s, "
          f"cost ${out['log']['cost_usd']:.4f}.")
    if not args.tag and not args.smoke:
        acc = gen.accept(out["paths"])
        print(f"Accepted: {acc['digest'].relative_to(ROOT)} and {acc['slots'].relative_to(ROOT)}. Serve with --cached.")


def print_rank_summary(result) -> None:
    print()
    print("Top five (one item per center)")
    for it in result["items"]:
        lead = it["lead"]
        flags = []
        if it["integrity"]:
            flags.append("integrity")
        if lead["plan_context"]:
            flags.append(lead["plan_context"])
        if lead["provider_context"]:
            flags.append(lead["provider_context"])
        print(f"  {it['rank']}. {it['center']:<15} {lead['label']:<30} score {it['score']:>5.2f}  "
              f"[{it['lane']}]  size {lead['size']:.1f} persistence {lead['persistence']:.1f} "
              f"context {lead['context']:.0f}  {'; '.join(flags)}")
        for s in it["supporting"]:
            print(f"       supporting: {s['label']} {s['latest']} (score {s['score']:.1f})")
        for f in it["favorable_context"]:
            print(f"       favorable:  {f['label']} {f['latest']} vs 8 week average {f['avg8']:.2f}")
        for c in it["unscored_context"]:
            print(f"       context:    {c['label']} {c['start']} ({c['start_week']}) to {c['latest']} ({c['latest_week']}), {c['direction']}")
    opp = result["opportunity"]
    if opp:
        print()
        print(f"Opportunity ({opp['rule']}): {opp['center']} {opp['label']} {opp['latest']} vs 8 week average {opp['avg8']:.2f}, "
              f"held {opp['held_weeks_of_6']} of 6 weeks")
        for n in opp["needed_by"]:
            print(f"  needed by {n['center']}: {n['why']}")
    print()
    print("Other movements")
    for m in result["other_movements"]:
        print(f"  {m['center']:<15} {m['label']:<30} {m['latest']} vs 8 week average {m['avg8']:.1f} ({m['direction']})")
    print()
    print("Why each center did or did not make the top five")
    for w in result["why_not"]:
        print(f"  {w['center']:<15} {w['why']}")


if __name__ == "__main__":
    main()
