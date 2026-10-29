# Algorithm audit

## Confirmed defects in the extracted baseline

1. **LN area double counting.** `computeWindowFeatures` first collects holds overlapping a window from heads, then adds tails of holds starting before that window again. A long hold crossing the start boundary can therefore contribute twice. The new engine integrates the union of clipped intervals once per column. It also unions overlapping malformed holds so physical occupancy is bounded by one finger per key.
2. **Tail rows break RC adjacency.** `attachTails` inserts tail-only rows into the head row sequence; some window features then inspect only the immediately preceding row and miss a repeated head when a tail is between them. The new engine maintains separate head and tail streams.
3. **Head/tail ordering and lock semantics.** The original held-state loop processes starts before ends at a shared timestamp, which can clear a newly begun hold. The new features measure pre-existing holds at head times and use explicit half-open intervals. Touching tail/head intervals are not merged for pre-existing-held support.
4. **Timing phase.** A window's off-beat calculation uses the first timing point as the phase anchor even after BPM/phase changes. The new grid residual is measured against the current uninherited timing point.
5. **Aggregation units.** The baseline divides confidence-weighted tag durations by unweighted total duration. Shares therefore do not sum to one, and lowering confidence changes the meaning of the mix thresholds. New evidence normalizes by the actual summed weights and reports model support separately from normalized composition.
6. **Confidence meaning.** The baseline multiplies softmax values by 1.4, doubles vibro rule scores, then clips only the winner. Those outputs are neither a probability distribution nor a calibrated confidence. New local heads emit a normalized distribution, entropy and margin. Temperature is searched in segment validation. Whole-map multi-label supports remain separate independent classifier scores.
7. **Sparse LN sections.** An early return when no heads occur in a window discards sustained LN occupancy. New windows can represent hold-only activity through the tail.

The parser additionally validates supported modes/keys, finite positive rate, and invalid hold tails. It reads comma-separated break events and metadata independent of section ordering. Original public behavior remains available through the untouched legacy module.

## New measurements

Head geometry is normalized by key count; 7K's flexible center column contributes half to each hand. The 6K geometry is 3+3, and 4K is 2+2. Features cover chord/overlap/cut-return, within-hand substitutions, interval variation, timing phase, LN union occupancy, independent releases, inverse gaps, repetitive masks, trills, rolls and sustained burden. The dimensional definitions are in `src/features.cjs` and the generated model's feature schema.

The `no_geometry` and `no_holds` ablations also remove their derived `effort` feature and effort weighting. `no_burden` removes effort weighting too. Other group deletions use the same confidence-and-effort aggregation as the full candidate. Explicit hard/duration/effort/confidence evidence ablations are separately retuned. Additional trills/rolls/anchor are continuous descriptors; they are not promoted to new supervised headline classes without corresponding labels.

## Experimental assumptions

- A 2-second output grid is a temporal resolution/computation contract, not a learned musical phrase duration. Context windows 2/4/8 seconds are searched. Viterbi switch penalties are searched; adjacent equal labels are merged with time-weighted evidence.
- Row merge tolerance stays at the original 10 ms acquisition convention. The complete-feature 0/5 ms controls were independently retuned on the same splits; validation did not select them over the main candidates. This is a one-factor sensitivity study, not every feature-subset × merge-tolerance interaction.
- Equivalent BPM is withheld when interval CV exceeds 0.5. No human BPM-per-segment labels were supplied: this is an explicit extrapolated display-uncertainty convention (interval spread no greater than half the mean), not a learned key-pattern boundary.
- Feature constants such as 15,000 ms for sixteenth-note equivalent BPM, the 1/2 interval ratio, unit occupancy and entropy normalization are dimensional definitions. They are not fitted difficulty thresholds.
- Weak map labels do not become precise segment ground truth. The local learner weights each weak chart as one observation and each manually annotated interval as one observation. Alternatives divide that interval's training weight.
- A 6K hold ending before it starts is rejected as malformed and logged with its chart ID. Difficulty-only 6K labels are never used as pattern supervision.

## Production boundary

The corrected feature engine and newly trained 7K head did not improve measured 7K accuracy. Default 7K therefore retains the inherited decision path and its known limitations, while exposing corrected interval measurements and normalized affinities. `algorithm: 'modern'` opts into the replacement feature/model pipeline. `algorithm: 'legacy'` is the exact raw extraction. The stable wrapper also extends an omitted short leading pickup to the first note without changing its tag.

Full-corpus QA found an infinite timing-point beat length in 6K `bm6k_1687`; the new parser ignores that invalid point and records a warning. A scan confirmed that no 4K/7K training timing point was affected, so the fitted-model evidence remains applicable.
