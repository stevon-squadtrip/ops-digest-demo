# Ops digest demo

A Monday digest for the medical director of a multi site clinic network. It decides which of their eleven clinics
earn attention this week, says what to do about each one, and checks whether the action plans their teams
committed to are actually on track. Built for thirty minutes on a Monday.

Demo on synthetic data. Built by [Stevon Judd](https://stevonjudd.com). Live page: see the repository description.

![The digest at desktop width](outputs/site_screens/1200.png)

## How it works

1. **Code ranks every clinic and metric** with a visible score: size of the move in standard deviations,
   weeks persisted, clinical weight, and confidence from the denominator. Five items, one per clinic.
2. **Weak signals are held back**, never dropped silently: a satisfaction score on seven responses, a metric whose
   definition changed mid period, a clinic too new to judge. Every hidden pair is logged with its reason.
3. **Every action plan is checked against the data.** A plan marked on track while its metric gets worse is called
   out. A target the panel had already met when the plan opened is called out. A perfect score that points at a
   logging gap is an integrity question, not praise.
4. **Claude writes the narrative with placeholders**, so it cannot invent a number. Code fills every value.
5. **A validator rejects any draft that breaks a rule**: a digit outside a placeholder, a banned word, a restated
   line, a word cap. Three failures stop the run.
6. **Reader feedback adjusts thresholds** and every change is written to a decision log.

## Run it locally

```
pip install -r requirements.txt
python -m src.run --stage digest --cached
python -m src.site
```

The first command needs no API key: it renders the accepted prose against freshly computed numbers. The data in
data/raw comes from scripts/make_synthetic_data.py with a fixed seed. `pytest -q` runs the suite.
Run `python scripts/prove_planted_cases.py` to see every planted edge case get caught.
