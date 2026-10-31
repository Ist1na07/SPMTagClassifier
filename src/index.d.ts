export interface Options { speedRate?: number; algorithm?: 'modern' | 'stable' | 'legacy' }
export interface Note { t: number; col: number; lnEnd: number | null }
export interface Beatmap {
  keys: number; notes: Note[]; firstNote: number; lastNote: number;
  timingPoints: { t: number; beatLength: number }[];
  breaks: [number, number][]; meta: Record<string, string | number | undefined>;
  warnings?: string[];
}
export interface Segment {
  start: number; end: number; tag: string; family: 'rc' | 'ln' | 'break';
  secondary?: string; equivBpm?: number;
  /** Support of the displayed class; after smoothing this need not be the largest score. */
  confidence: number; scores?: Record<string, number>;
  /** Stable 7K: normalized legacy affinities, not calibrated probabilities. */
  distribution?: Record<string, number>;
  probability?: number[];
  /** Effective contribution to the current aggregation; not empirical correctness. */
  aggregationWeight?: number;
  evidence?: { normalizedEntropy: number; margin: number; effort: number; occupancy: number;
    featureWindows: number; contextNoteObservations: number; legacyConfidence?: number;
    concentration?: number; structuralWindows?: number; clippedFeatureFraction?: number;
    sourceStart?: number; sourceEnd?: number; geometry?: GeometryEvidence };
}
export interface GeometryEvidence {
  start: number; end: number;
  anchor: { peakShare: number; peakExcess: number; repeatConcentration: number;
    runAsymmetry: number; runCoverage: number; regularity: number; regularAnchor: number };
  release: { tailRatePerKey: number; headIsolation: number; sameFingerRecovery: number;
    tailAsynchrony: number; tailTimingVariability: number; heldOtherAtTail: number;
    sameHandOrder: number; releaseChordFraction: number; independentHeldRelease: number };
}
export interface Overall {
  tags: string[]; sort: string; weights: Record<string, number>;
  scores?: Record<string, number>; scoreKind?: string; compositionKind?: string;
  composition?: Record<string, number>;
  familyMass?: { ln: number; rc: number };
  taxonomy?: string;
  evidence?: { activeWeight: number; normalizedEntropy: number; meanLocalEntropy: number; diversity: number };
  transfer?: { lambda: number; segmentLambda: number; evidenceLevel: string };
}
export interface FeatureWindow {
  start: number; end: number; values: number[]; metrics: Record<string, number>;
  notes: number; rows: number; equivBpm?: number;
  contextStart?: number; contextEnd?: number;
}
export interface Result {
  map: Beatmap; segments: Segment[]; overall: Overall;
  debug: { windows: FeatureWindow[] | Record<string, number>[]; norm?: unknown; ctx?: unknown };
  modelInfo?: { version: string; keyCount: number; variant: string; segmentEvidence: string; temporalResolutionMs: number;
    taxonomy?: string; classes?: string[]; featureKeys?: string[]; featureCount?: number;
    physicalSupport?: 'none' | 'jack' | 'ln' | 'both'; defaultReplacementAccepted?: boolean;
    effortPower?: number; concentrationPower?: number; extraGate?: string;
    localScoreKind?: string; confidenceMeaning?: string; coverage?: string;
    sortMeaning?: string; overallDecision?: string; supervision?: string;
    aggregationPolicy?: { mode: 'legacy' | 'normalized' | 'soft'; effortPower: number;
      confidencePower: number; burdenGate: number; lnRatio: number; scorer: string;
      [parameter: string]: unknown } };
  /** Canonical native coarse taxonomy; public stamina is its chordstream alias. */
  coarse?: { dominant: string | null; composition: Record<string, number>;
    familyMass: {ln: number; rc: number}; scoreKind: string;
    evidence: { activeWeight: number; normalizedEntropy: number; meanLocalEntropy: number; diversity: number } };
  descriptors?: { start: number; end: number; anchor: number; trill: number; roll: number;
    repetition: number; occupancy: number; releaseIndependence: number;
    releaseHeldInteraction?: number; geometry?: GeometryEvidence }[];
}
export function analyzeMap(text: string, options?: Options): Result;
export function extractIntermediate(text: string, options?: Options): {
  map: Beatmap; windows: Result['debug']['windows']; segments: Segment[]; overall: Overall;
};
/** Original v1.1.0 cache schema, retained unchanged. This cache is for the legacy algorithm. */
export function buildCachedMap(text: string, options?: Options): {
  cutDemandRef: number; rawSegs: unknown[]; extras: unknown[]; windows: Record<string, number>[];
};
export function buildEvidenceMap(text: string, options?: Options): {
  map: Beatmap; windows: FeatureWindow[]; global: FeatureWindow; featureKeys: string[];
};
export const legacy: {
  analyzeMap: typeof analyzeMap; extractIntermediate: typeof extractIntermediate;
  buildCachedMap: typeof buildCachedMap;
};
