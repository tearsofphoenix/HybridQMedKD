# Verification

- Date: 2026-03-29
- Executor: Codex

## Verified evidence used for paper updates

- `outputs/tables/teacher_summary.json`
  - AUC: `0.9939 ± 0.0050`
  - F1: `0.9416 ± 0.0149`
  - MCC: `0.9089 ± 0.0244`
  - Train time: `0.29 ± 0.02 s/fold`
- `outputs/tables/teacher_entropy.json`
  - Train mean entropy: `0.0968 bits`
  - Validation mean entropy: `0.1003 bits`
  - High-confidence fraction `|p-0.5| > 0.45`: `89.7%` (train), `89.6%` (val)

## Verified actions completed

1. Updated `paper/main.tex` to replace stale teacher-entropy claims with the real values above.
2. Added teacher MCC to the main results table.
3. Clarified that the main hybrid baseline uses middle placement and that the dedicated `alpha` sweep is evaluated under tail placement.
4. Verified real alpha-ablation outputs in `outputs/tables/ablation_alpha.json`:
   - `alpha=0.3`: AUC `0.9920 ± 0.0082`, F1 `0.9358 ± 0.0303`, MCC `0.9013 ± 0.0440`
   - `alpha=0.5`: AUC `0.9918 ± 0.0067`, F1 `0.9510 ± 0.0186`, MCC `0.9231 ± 0.0302`
   - `alpha=0.7`: AUC `0.9903 ± 0.0071`, F1 `0.9504 ± 0.0187`, MCC `0.9240 ± 0.0277`
5. Fixed `src/analysis/plot_results.py` so `plot_ablation_alpha` accepts JSON-loaded string keys and generated `outputs/figures/ablation_alpha.pdf/png`.
6. Updated the additional-dataset appendix note to reflect that Pima dataset preparation and a smoke run exist, but the full multi-dataset pipeline has not yet been executed.
7. Compiled `paper/main.pdf` successfully with local TeX packages installed under `.texmf`.

## Remaining limitations

- The paper still reports only a single full benchmark dataset (WDBC); Pima remains smoke-test-only.
- LaTeX compilation still emits non-fatal overfull `\hbox` warnings.
