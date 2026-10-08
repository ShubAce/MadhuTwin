# Clinician feedback session kit (15–20 minutes per clinician)

Purpose: get structured feedback from 3–5 practising clinicians (diabetologists, physicians, diabetes educators) on the MadhuTwin dashboard. The resulting usability score and quotes go into the presentation and README. Sessions use synthetic and de-identified data only, so no patient data is involved.

## Before the session

- Run the dashboard (`docker compose up`, or API + `npm run dev`) on a laptop, or share the public demo link.
- Read the information note below to the clinician and ask whether they are happy to continue and to be quoted, either by name and role or anonymously. Record their answer.
- Open the panel at **Day 1 · 06:30**. Keep a timer and a notes sheet.

> **Information note.** "This is a student research prototype for the Happiest Health Digital Twin Challenge. All patients are synthetic or de-identified research recordings. We would like 15–20 minutes of your feedback on whether the tool would help in clinical practice. Your answers are voluntary and you can stop at any time. With your permission we may quote you, by name or anonymously as you prefer."

## Tasks (think aloud; do not help unless they are stuck for more than 2 minutes)

| # | Task | Link / start point | What to note |
|---|---|---|---|
| 1 | "Which patient on this panel would you call first, and why?" | Panel, Day 1 · 06:30 | Did the ranking match their judgement? |
| 2 | "This patient is on premixed insulin. What is the twin telling you, and what would you do?" | `#/patient/MT-0066?clock=480` | Did they understand the forecast cone and the reasons? |
| 3 | "She is about to have lunch. Compare rice-dal with jowar roti and a 15-minute walk." | What-if tab | Was the what-if useful for counselling? |
| 4 | "What has changed for this patient over the last few days?" | Illness patient (`#/patient/MT-0990?clock=2040`) | Did they notice the insulin-resistance insight? |
| 5 | "Is this patient meeting time-in-range targets?" | Festival patient → AGP report tab | Correct reading of TIR, TAR and TBR |
| 6 | "Her eGFR is 51. Would you stop the SGLT2 inhibitor? Try it on the twin first." | SGLT2 patient → Therapy simulator tab (`?tab=therapy`) | Did they read the trust check? Would they use the dose-response table? |
| 7 | "Share this patient's forecast with the hospital record system." | Interoperability (FHIR) tab | Is the RiskAssessment meaningful to them? |

## Questionnaire 1: System Usability Scale (Brooke, 1986)

Score each item from 1 (strongly disagree) to 5 (strongly agree).

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

**Scoring:** for odd items subtract 1 from the score; for even items subtract the score from 5. Add the ten results and multiply by 2.5, giving a score from 0 to 100. The average across all products is about 68; above 80 is excellent.

## Questionnaire 2: clinical usefulness (1 strongly disagree to 5 strongly agree)

1. A 1–2 hour warning before a glucose spike or low would change what I do for this patient.
2. The reasons shown with each alert are clinically sensible.
3. I would trust the forecast cone enough to act on it, alongside my own judgement.
4. The what-if simulator would help me counsel patients about meals and activity.
5. The organ-level "virtual patient" summary is useful.
6. The number of alerts would be acceptable in my practice.
7. I would use a tool like this for remote monitoring of my patients with diabetes.
8. The therapy simulator, with its trust check, would help me discuss medication changes with patients and colleagues.

**Open questions**

- What is the single most useful thing you saw?
- What is missing, or what would stop you using it?
- Is anything shown potentially unsafe or misleading?
- Which patients in your practice would benefit most?

## Results template (paste into `docs/clinician_feedback.md`)

| Clinician (role, setting) | Quotable? | SUS | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Key quote |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e.g. Consultant diabetologist, private clinic, Bengaluru | Name | | | | | | | | | | |

**Summary line for the presentation:** "N clinicians · mean SUS X/100 · Y of N would use it for remote monitoring", followed by the best quote and one concrete improvement you made in response.
