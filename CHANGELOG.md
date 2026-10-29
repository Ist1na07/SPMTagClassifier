# Changelog

## 1.1.0 ¡ª 2026-10-30

- Extracted the 7K classifier from plugin tag v1.2.0 (not release v1.0.0).
- Preserved algorithm, embedded model, public functions, browser global and rate semantics.
- Added standalone Node package, CLI, unit tests, full-corpus parity runner and provenance.

## 1.2.0 â€” 2026-10-30

- Added modular parser, head/tail-separated geometry, union LN occupancy, current timing-phase features, sequence descriptors, normalized evidence and Viterbi segmentation.
- Added tuned 4K five-class models and 6K projected-domain transfer; documented missing human 6K labels and the negative coarse-fusion result.
- Preserved stable 7K decisions because the newly trained head did not beat the mature baseline; exposed the modern head explicitly for research, plus corrected interval diagnostics and leading-pickup coverage.
- Added browser IIFE, TypeScript declarations, CLI mode selection and preserved the original cache entry point.
- Published grouped splits, full independently retuned map/segment/row-merge ablations, projection controls, per-chart predictions, class metrics, numerical parity and corpus audit.
- Reject invalid backwards hold tails; ignore nonfinite timing points with a diagnostic.
