'use strict';

/** Milliseconds after rate conversion; tail intervals are half-open [head, tail). */
function parseOsu(text, { speedRate = 1 } = {}) {
  if (!Number.isFinite(speedRate) || speedRate <= 0) throw new RangeError('speedRate must be finite and positive');
  let section = '', mode = null, keys = null;
  const rawNotes = [], timingPoints = [], breaks = [], meta = {}, warnings=[];
  for (const raw of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('//')) continue;
    if (line.startsWith('[')) { section = line.slice(1, -1).toLowerCase(); continue; }
    if (['general', 'difficulty', 'metadata'].includes(section)) {
      const at = line.indexOf(':'); if (at < 0) continue;
      const k = line.slice(0, at).trim(), v = line.slice(at + 1).trim();
      if (k === 'Mode') mode = Number(v);
      if (k === 'CircleSize') keys = Number(v);
      if (section === 'metadata') meta[k] = v;
      if (k === 'OverallDifficulty') meta.od = Number(v);
    } else if (section === 'timingpoints') {
      const p = line.split(','), t = Number(p[0]), beatLength = Number(p[1]);
      if(!Number.isFinite(t)||!Number.isFinite(beatLength)){warnings.push('IgnoredNonfiniteTimingPoint');continue;}
      if (beatLength > 0 && (p[6] === undefined || p[6].trim() !== '0'))
        timingPoints.push({ t: t / speedRate, beatLength: beatLength / speedRate });
    } else if (section === 'events') {
      const p = line.split(/\s*,\s*|\s+/);
      if (p[0] === '2' || p[0] === 'Break') {
        const a = Number(p[1]), b = Number(p[2]);
        if (Number.isFinite(a) && b > a) breaks.push([a / speedRate, b / speedRate]);
      }
    } else if (section === 'hitobjects') rawNotes.push(line.split(','));
  }
  if (mode !== 3) throw new Error('NotMania');
  if (![4, 6, 7].includes(keys)) throw new Error('UnsupportedKeys: expected 4, 6 or 7');
  const notes = [];
  for (const p of rawNotes) {
    const x = Number(p[0]), t = Number(p[2]), type = Number(p[3]);
    if (p.length < 5 || !Number.isFinite(x) || !Number.isFinite(t) || !Number.isInteger(type))
      throw new Error('InvalidHitObject');
    if (!(type & 129)) continue;
    const end = type & 128 ? Number((p[5] ?? '').split(':')[0]) : null;
    if (end !== null && (!Number.isFinite(end) || end < t)) throw new Error('InvalidHoldTail');
    notes.push({ t: t / speedRate, col: Math.min(keys - 1, Math.max(0, Math.floor(x * keys / 512))),
      lnEnd: end !== null && end > t ? end / speedRate : null });
  }
  notes.sort((a, b) => a.t - b.t || a.col - b.col);
  if (!notes.length) throw new Error('EmptyBeatmap');
  Object.assign(meta,{title:meta.Title,artist:meta.Artist,version:meta.Version,creator:meta.Creator});
  timingPoints.sort((a, b) => a.t - b.t);
  // Duplicate objects on one key at the same time are one physical action.
  const unique = [];
  for (const n of notes) {
    const last = unique.at(-1);
    if (last && last.t === n.t && last.col === n.col) {
      const end = Math.max(last.lnEnd ?? n.t, n.lnEnd ?? n.t); last.lnEnd = end > n.t ? end : null;
    }
    else unique.push(n);
  }
  return { keys, notes: unique, timingPoints, breaks, meta, warnings,
    firstNote: unique[0].t, lastNote: unique.reduce((end,n)=>Math.max(end,n.lnEnd??n.t),unique[0].t) };
}

function popcount(mask) { let n = 0; while (mask) { mask &= mask - 1; n++; } return n; }
function lowerBound(a, t, key = x => x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (key(a[mid]) < t) lo = mid + 1; else hi = mid; }
  return lo;
}
function prepare(map, rowMergeMs = 10) {
  const rows = [], cols = Array.from({ length: map.keys }, () => []), tails = [];
  for (const n of map.notes) {
    cols[n.col].push(n);
    let r = rows.at(-1);
    if (!r || n.t - r.t > rowMergeMs) rows.push(r = { t: n.t, mask: 0, lnMask: 0 });
    r.mask |= 1 << n.col; if (n.lnEnd !== null) { r.lnMask |= 1 << n.col; tails.push({ t: n.lnEnd, col: n.col, start: n.t }); }
  }
  // Union overlapping holds per column: occupancy never exceeds one finger per key.
  const holds = cols.map(notes => {
    const out = [];
    for (const n of notes) if (n.lnEnd !== null) {
      const last = out.at(-1);
      if (last && n.t < last[1]) last[1] = Math.max(last[1], n.lnEnd);
      else out.push([n.t, n.lnEnd]);
    }
    return out;
  });
  tails.sort((a, b) => a.t - b.t || a.col - b.col);
  return { map, rows, cols, tails, holds, rowMergeMs };
}
module.exports = { parseOsu, prepare, popcount, lowerBound };
