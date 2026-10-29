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
  evidence?: { normalizedEntropy: number; margin: number; effort: number; occupancy: number;
    featureWindows: number; contextNoteObservations: number };
}
export interface Overall {
  tags: string[]; sort: string; weights: Record<string, number>;
  scores?: Record<string, number>; scoreKind?: string; compositionKind?: string;
  composition?: Record<string, number>;
  transfer?: { lambda: number; segmentLambda: number; evidenceLevel: string };
}
export interface FeatureWindow {
  start: number; end: number; values: number[]; metrics: Record<string, number>;
  notes: number; rows: number; equivBpm?: number;
}
export interface Result {
  map: Beatmap; segments: Segment[]; overall: Overall;
  debug: { windows: FeatureWindow[] | Record<string, number>[]; norm?: unknown; ctx?: unknown };
  modelInfo?: { version: string; keyCount: number; variant: string; segmentEvidence: string; temporalResolutionMs: number };
  descriptors?: { start: number; end: number; anchor: number; trill: number; roll: number;
    repetition: number; occupancy: number; releaseIndependence: number }[];
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
