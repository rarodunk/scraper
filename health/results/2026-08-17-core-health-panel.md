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

## Missing context needed for full interpretation
- **Cycle day** at draw (estradiol and FSH are uninterpretable without it)
- **Draw time** for the morning cortisol (a 10am draw is not a 7:30am draw)
- Fasting status; recent illness, training load, NSAID or steroid use
- Current medications and supplements
