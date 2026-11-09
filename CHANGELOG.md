# Changelog

## 1.2.2 — 2026-11-09

- Correct 7K displayed equivalent BPM for rows uniform to source-millisecond resolution; avoid coarse-grid aliases such as 300 → 200 BPM. Keep archived model features and raw `legacy` results unchanged.
- Expose the direct measurement and original display value in `segment.evidence.cadence`.
- Preserve Jack's chord-row/repetition clock when single rows interleave; do not substitute the all-row cadence.
- Restrict the default 7K regular-small-chord Break disjunct using jointly selected relative NPS and actual row rate. Preserve expert coefficients and the original whole-map aggregation policy; expose `modelInfo.overlayPolicy`.
- Complete matched 57-gate × 245-aggregation comparisons and original-aggregation controls with source-grouped player supervision. Fix the two curated Speed→Break regressions; whole-map macro-F1 is essentially unchanged, and original six-chart OOF accuracy decreases by 0.17 percentage points. Tech→Dense CS remains unresolved.
- Add focused overlay cadence/hand-mechanics regressions and a test-only driver using the actual upstream renderer, without connecting to or modifying the user's live tosu/plugin.

## 1.2.1 — 2026-11-01

- Replace 4K/6K inference with native-key coarse models: the same local support drives display, jointly fitted evidence weights and whole-map composition.
- Use real six-column features and shared 4K/7K supervision rather than projected output mixtures; remove unsupported six-key fine-label claims.
- Keep mature default 7K decisions and add anchor/release geometry. Replace the opt-in flat research head with the jointly selected anchored LN candidate, without claiming a default accuracy gain.
- Retune every declared expert/consumer configuration in nested song groups, retain negative results and distinguish fit identities from equivalent training problems.
- Publish full-data inner-selection profiles, actual six-key runtime/mirror checks and final Python/Node/browser comparisons.
- Preserve the legacy cache and 40-feature diagnostic-cache contracts; add static CSP-compatible builds, checked TypeScript declarations and explicit reproduction/model-export commands.
- Keep published v1.1.0/v1.2.0 tags immutable. See `research/rework/CONCLUSIONS.md` for corrected early-study claims and current evidence.

## 1.1.0 — 2026-10-30

- Extracted the 7K classifier from plugin tag v1.2.0 (not release v1.0.0).
- Preserved algorithm, embedded model, public functions, browser global and rate semantics.
- Added standalone Node package, CLI, unit tests, full-corpus parity runner and provenance.

## 1.2.0 — 2026-10-30

- Added modular parser, head/tail-separated geometry, union LN occupancy, current timing-phase features, sequence descriptors, normalized evidence and Viterbi segmentation.
- Added tuned 4K five-class models and 6K projected-domain transfer; documented missing human 6K labels and the negative coarse-fusion result.
- Preserved stable 7K decisions because the newly trained head did not beat the mature baseline; exposed the modern head explicitly for research, plus corrected interval diagnostics and leading-pickup coverage.
- Added browser IIFE, TypeScript declarations, CLI mode selection and preserved the original cache entry point.
- Published grouped splits, full independently retuned map/segment/row-merge ablations, projection controls, per-chart predictions, class metrics, numerical parity and corpus audit.
- Reject invalid backwards hold tails; ignore nonfinite timing points with a diagnostic.
