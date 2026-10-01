# Lotto generation review

## Conclusion

The user-approved workflow restores lean ML generation, not the previous heavy
ensemble pipeline. It uses a single logistic model and keeps uniform generation
selectable. No predictive advantage is guaranteed: for fair independent draws,
historical frequency and gaps do not change the probability of a valid ticket.
This document does not claim a live historical evaluation was run or establish
an out-of-sample improvement over a uniform baseline.

## Earlier simplification — heavy workflow remains removed

- Automatic XGBoost/Random Forest/LightGBM training and stacked-model selection.
- SHAP, PageRank, calibration and conformal-coverage panels on the generation page.
- Ranking generated tickets by structural quality and locking their repeated
  numbers into an AI "consensus core". Repetition within generated samples is
  evidence about the generator's weighting, not about future lottery draws.
- Prompt feedback that treated numerical near misses as evidence or penalized
  the latest winning numbers. Numerical closeness has no predictive meaning here.
- Automatic AI requests when opening charts or the generation page.
- ML selection of wheel keys and ML-run logging for ordinary random tickets.
- Torch, Optuna, XGBoost, LightGBM, SHAP and NetworkX from default requirements.

## Current workflow

1. Keep historical statistics separate from generation. The default ML path uses
  `lean_ml`: a single logistic regression with three frequency features plus
  position indicators, rather than ensembles or automatic model selection.
2. Use separate production and evaluation fits of the same model specification.
  Evaluation uses an 80/20 chronological split, a 30-draw burn-in, and at least
  100 valid draws per slot. Production fitting is separate from the held-out
  evaluation; its fit must not be presented as held-out evidence.
3. Compare Brier scores with a uniform baseline and report an approximate
  draw-level confidence interval. These diagnostics do not guarantee an
  advantage or provide a confidence interval for winning a ticket.
4. Keep uniform generation explicitly selectable and use it as a fallback when
  lean ML cannot be fitted. Retain the 30/90-day window options. Both methods
  preserve game semantics: distinct sorted numbers for unordered games; ordered
  positions with repeats permitted for digit games. Exclude duplicate tickets.
5. Preserve reproducibility through an explicit seed or a data-snapshot seed.
  A seed changes reproducibility, not winning odds.
6. On request, ask AI to explain the same first displayed combination using the
  actual scores and coefficients supplied by the local model. The ticket is
  immutable: AI cannot replace, regenerate, or reorder it, and cannot replace
  the application disclaimer. Uniform/fallback explanations must not imply a
  fitted model or invent coefficients.
7. Let the user select wheel keys manually. Coverage depends on winning numbers
   falling within that key set, and extra tickets incur extra cost.

The generation API defaults to `method=ml`. GET supports an explicit
`method=uniform` and returns `by_draw_time` for multi-draw games when draw time is
unspecified. POST accepts `method` and `n_suggestions`. Public API paths remain
available; the earlier statement that ML arguments are ignored no longer
describes the current generation workflow. Saved bets and draw data were not
migrated or deleted.

Default dependencies include scikit-learn and SciPy for lean ML, without restoring
Torch, Optuna, XGBoost, LightGBM, SHAP, NetworkX, or the heavy research stack.

## Additional correctness fixes

- Chronological preparation includes intraday draw time.
- Ordered API explanations use position-specific counts, including zero digits,
  rather than sorting a digit sequence as an unordered ticket.
- The named probability route now precedes the generic analysis route.
- Test fixtures now sample pick-six draws without replacement and preserve order
  and repeats for digit games.
- AI prompts use the correct interpretation of a p-value.
- Explanation caches track the dataset, not the unrelated ticket-count slider.

## Retained experimental code — not validated for prediction

`src/lotto/scorer.py` and `src/lotto/lstm_forecast.py` remain available to avoid
destroying existing experimental work. They are no longer called by the dashboard
generation or wheel workflows. Their additional packages are isolated in
`requirements-research.txt`; installed packages were not uninstalled.

Static review identified unresolved research limitations:

- Calibration diagnostics reuse data seen during fitting/calibration and do not
  evaluate the same stacked predictor used at inference.
- The conformal routine reuses training data and conflates individual-number
  coverage with coverage of all numbers in a draw. Its guarantee is not established.
- The LSTM range is a heuristic, not a validated confidence interval.
- Legacy backtests can discard ordered positions/repeated digits. Payout assumptions
  and comparisons with other weighted strategies are not proof of an advantage.
- Small training partitions, unused scoring options and tuning on calibration
  labels need correction before these modules can support research conclusions.

These algorithms were isolated, not repaired or certified. Any future experiment
should treat the lean ML restoration as separate work: the old research modules
and their unresolved limitations are unaffected. Research evaluation
should use chronological training, separate calibration and untouched test periods,
correct game semantics, a uniform baseline, and draw-level uncertainty estimates.
Do not infer predictive skill from training accuracy, SHAP, near misses, or
ticket structure.

## Validation

Regression coverage addresses ticket validity and reproducibility, ordered repeats
and zeros, finite ticket spaces, uniform generation and fallback, lean ML features
and chronological evaluation boundaries, mocked AI handling of immutable tickets,
and API defaults, methods, suggestion counts, windows and draw-time payloads.
Test-suite validation is reported separately by the primary implementation report;
no test totals are asserted here. Unit tests do not establish live historical
evaluation results, predictive advantage, live AI output quality, or browser
interaction. Mocked explanation tests do not require paid inference requests.
