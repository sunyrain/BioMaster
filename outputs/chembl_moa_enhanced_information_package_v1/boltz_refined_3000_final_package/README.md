# Boltz-2 Refined Top3000 Final Package

Generated: 2026-07-05T19:00:23Z

## Scope

Merged the full refined Boltz-2 Top3000 run with singleton reruns for rows that failed inside 5-row batches.

## Headline

- Rows: 3000
- Completed rows: 2988 (99.60%)
- A/B rows: 910 (30.33% of all rows; 30.46% of completed rows)
- Tier counts: {'C_boltz_partial_signal_review': 2078, 'B_boltz_review_supported': 800, 'A_boltz_second_model_supported': 110, 'U_boltz_not_completed': 12}
- A/B unique drugs: 297
- A/B unique targets: 180
- Recommended final rows: 1000
- Recommended A/B rows: 547

## Interpretation

- Boltz refined A/B is structural second-model support, not wet-lab binding proof.
- Carbonic anhydrase, same-family kinase, and ion-channel rows are explicitly flagged for control/risk review.
- The refined final 1000 keeps the existing physics-first scoring framework and replaces fast Boltz evidence with refined Boltz evidence.
