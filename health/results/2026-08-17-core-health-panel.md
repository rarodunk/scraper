# Core Health Panel — 17 Aug 2026

**Patient:** Samantha Reeves · Female · ID SILA42B532
**Panel:** Ranges Core Health Test · Exported 26 Aug 2026
**Parsed by:** `health/tools/parse_panel.py` (23/23 markers, values verified against source PDF)

## RED — outside every range the lab provides

| Marker | Value | Optimal | Direction |
|---|---|---|---|
| Cystatin C | 1.17 mg/L | 0.55–0.9 (fair caps at 1.01) | **HIGH** |
| Ferritin | 21.1 ng/mL | 70–90 | **LOW** |
| 25-(OH) Vitamin D | 42.85 ng/mL | 50–80 | **LOW** |
| Morning Cortisol | 6.6 µg/dL | 10–15 (good floor 6.7) | **LOW** |
| LDL-C:ApoB Ratio | 0.85 | 1.5–2.0 (fair floor 1.2) | **LOW** |

## AMBER — inside a wider tier, outside optimal

| Marker | Value | Optimal | Note |
|---|---|---|---|
| Cholesterol, Total | 111.2 mg/dL | 160–200 | Low; flagged by the lab's own insight |
| HDL Cholesterol | 49 mg/dL | 80–100 | In "fair" only |
| Homocysteine | 8.13 µmol/L | 5–7 | Methylation strain |
| TSH | 2.78 µIU/mL | 1–2 | Early thyroid under-function |
| Triglycerides:HDL | 1.24 | 0–1.1 | Marginal, driven by low HDL |
| Estradiol | 66.8 pg/mL | good 0–517 | Cycle-day dependent |
| FSH | 3.07 mIU/mL | good 1.79–113.59 | Premenopausal; rules out perimenopause |

## GREEN — at optimal

hs-CRP 0.24 · HbA1c 4.6% · eAG 84 mg/dL · ApoB 59 mg/dL · ApoA1 134 mg/dL ·
LDL-C 50 mg/dL · LDL-C:HDL-C 1.0 · ApoB:ApoA1 0.44 · Total Chol:HDL 2.27 ·
VLDL 12.2 mg/dL · Triglycerides 61 mg/dL

## Internal consistency check
Friedewald holds exactly: TC 111.2 = HDL 49 + LDL 50 + VLDL 12.2, and
VLDL 12.2 = Trig 61 ÷ 5. **LDL-C is calculated, not directly measured** — which
matters for reading the LDL-C:ApoB ratio at this low a cholesterol.

## Draw context (supplied 27 Aug 2026)
- **Cortisol drawn 9:00am** — inside the conventional 7–9am window, so the 6.6
  should be read as probably real rather than a timing artifact. Wake time still
  unknown, and it is the variable that would refine this.
- **Cycle day −1.** Period began Tue 18 Aug 2026, the day after the draw — the
  final day of the luteal phase.
  - FSH 3.07 is physiologically suppressed in the luteal phase and **cannot** be
    used to assess ovarian reserve. Needs a day-3 draw.
  - Estradiol 66.8 is unremarkable for day −1.
  - Ferritin 21.1 is a pre-menstrual value; the post-period nadir is likely lower.
  - Total cholesterol and HDL sit near their cycle low (TC ~3% / LDL-C ~5% lower
    mid-luteal than mid-follicular) — a 3–5% correction on a ~50 mg/dL gap.
  - Ferritin, cystatin C, vitamin D, TSH, homocysteine, HbA1c: unaffected.

## Still missing
- **Wake time** on the day of the draw
- **Progesterone** — never measured; needs a mid-luteal draw (~7–8 Sept)
- Fasting status; recent illness, training load, NSAID or steroid use
- Current medications and supplements
