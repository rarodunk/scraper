# Health Lab Analysis

An agent that decodes blood panels and functional lab results, finds the
root-cause pattern behind the numbers, and builds a food-first protocol from
Traditional Chinese Medicine, Ayurveda, Western herbalism, and ancestral
nutrition — sequenced onto Dr. Josh Axe's four-month cellular arc.

## Use it

```
> decode the lab results in health/results/  ...or just share a lab PDF
```

The agent (`.claude/agents/ancient-remedies-lab-analyst.md`) triggers on lab
work, bloodwork, biomarker panels, "what do my results mean," cellular reset,
or The Health Institute protocol.

Parse a panel PDF directly:

```bash
pip install pypdf
python3 health/tools/parse_panel.py path/to/panel.pdf -o health/results/panel.json
```

It emits structured JSON plus a triage table — **RED** (outside every range the
lab gives), **AMBER** (inside a wider tier but below optimal), **GREEN** (optimal).

## Layout

```
.claude/agents/ancient-remedies-lab-analyst.md   the agent
health/reference/
  marker-playbook.md            per-marker meaning, optimal ranges, doses
  absorption-and-root-cause.md  the six root-cause patterns
  ancient-materia-medica.md     TCM / Ayurvedic / Western herbs + cautions
  axe-method.md                 Axe's frameworks + the 4-month protocol
  report-template.md            report structure
health/tools/parse_panel.py     PDF -> structured markers
health/results/                 parsed panels
health/reports/                 written analyses
```

## How it works

1. **Parse and triage** every marker against each range tier the lab provides.
2. **Find the pattern, not the list** — malabsorption, under-fuelling, iron
   deficiency, HPA down-regulation, methylation strain, subclinical thyroid.
   State the causal chain so one fix visibly explains several markers.
3. **Overlay the traditional systems** — the named TCM pattern and classical
   formula, the Ayurvedic constitution, the folk-herbal tonic.
4. **Prescribe food first**, then pairing and timing, then preparations, then
   herbs, then supplements, then practices.
5. **Sequence onto the four-month arc** — Reset → Repair → Detoxify →
   Regenerate — and adapt it to the person, stating every deviation out loud.
6. **Close the loop** — retest schedule, the tests to ask a doctor for, stop
   conditions, and a first-week starting point.

## Safety

The agent is built with hard rails: escalate anything outside every range to a
physician rather than treating it; never change a prescribed medication; carry
every herb's caution in the same breath as its dose; and never recommend iron,
iodine, vitamin A, vitamin D, or selenium open-endedly — always with a retest
date and a stop condition.

**Health education, not medical advice.** The Health Institute's own materials
and a person's own practitioner outrank anything here.
