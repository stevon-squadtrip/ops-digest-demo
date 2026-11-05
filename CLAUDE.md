# Ops digest demo: rules for Claude Code

Read this file before changing anything. This is a public portfolio demo on synthetic data. There is no client, no
real clinic and no real person anywhere in the repo, and none may be added.

## What it is
A Monday digest for the medical director of a 24 clinic network. Code ranks every clinic and metric, holds back
weak signals, checks every action plan against the data, and writes every number. A model writes 16 short passages
around placeholders. A validator rejects any draft that breaks a rule. Output: outputs/digest.md, a static page in
site/index.html, and the CSV and JSON files in outputs/.

## Hard rules
1. Code computes every number. The model only writes sentences around placeholders such as c11_rc_avg8 in curly braces.
2. Every claim is traceable: each fact carries a source (file, column, scope, window) and an evidence row.
3. Nothing medical. The validator blocks: diagnose, diagnosis, symptom, treatment, prescribe, medication, dose.
4. Suppression is logged, never silent: every (center, metric) considered and not shown goes to
   outputs/suppressed.json with the reason.
5. Plain words, short sentences, no em dashes anywhere (code, docs, digest, commit messages).
6. No secrets. ANTHROPIC_API_KEY lives in .env, which is ignored. The site never calls an API.
7. Generic names only: "Northside Clinic Network", "Clinic 01" to "Clinic 24", roles instead of people.
8. Thresholds are never bent toward inclusion. If a planted case does not surface, fix the data generator or
   report it; do not move a line after seeing a result.

## Data
data/raw is written by scripts/make_synthetic_data.py from a fixed seed. Change the generator, never the CSVs.
After any generator change: rerun it, rerun the pipeline, rebuild tests/fixtures/expected.json from the outputs
(the tests are the spec for behaviour, not for exact values), and run the tests.

## Run
```
python scripts/make_synthetic_data.py
python -m src.run --stage digest --cached
python -m src.site
pytest -q
```
`--generate` rewrites the prose with one model call and needs ANTHROPIC_API_KEY. The accepted run is committed
and served with `--cached`, so the public page and the tests never need a key.

## Working agreement
Build one change at a time. Run the tests after each. Report every place the implementation differs from what was
asked, one line each, before saying a step is done.
