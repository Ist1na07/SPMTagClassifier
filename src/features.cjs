'use strict';
const { popcount: pc, lowerBound: lb } = require('./parser.cjs');
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
const quantile = (a, q) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.floor((s.length - 1) * q)]; };
const sd = a => { const m = mean(a); return Math.sqrt(mean(a.map(x => (x - m) ** 2))); };
const frac = (a, f) => a.length ? a.filter(f).length / a.length : 0;
const GROUPS = {
  base: ['npsPerKey','rowRate','chordFraction','multiFraction','singleFraction','lnHeadFraction'],
  geometry: ['overlap','repeatFraction','cutReturn','chordChange','columnEntropy','handImbalance','sameHand','crossCut'],
  rhythm: ['logIoiSd','ioiChange','graceFraction','gridResidual','rowBpm','ioiCv'],
  holds: ['occupancy','minOccupancy','occupancySd','heldAtHead','lockFraction','releaseIndependence','releaseScatter','tailRate','shortHoldFraction','inverseGap','lnHeadOverlap'],
  sequence: ['anchorShare','repeatRun','maskEntropy','trillFraction','rollFraction'],
  burden: ['effort','sustain','burstiness','activeFraction']
};
const FEATURE_KEYS = Object.values(GROUPS).flat();

/** All hold area uses interval unions; all RC adjacency uses heads, never tails. */
function features(ctx, start, end) {
  const { map, rows, cols, holds, tails } = ctx, K = map.keys;
  const dur = Math.max(end - start, 1), seconds = dur / 1000;
  const rs = rows.slice(lb(rows, start, x => x.t), lb(rows, end, x => x.t));
  const notes = map.notes.slice(lb(map.notes, start, x => x.t), lb(map.notes, end, x => x.t));
  const ts = tails.slice(lb(tails, start, x => x.t), lb(tails, end, x => x.t));
  const hits = rs.map(r => pc(r.mask)), iv = rs.slice(1).map((r, i) => r.t - rs[i].t).filter(x => x > 0);
  const med = quantile(iv, .5), overlaps = [], changes = [], returns = [], sameHands = [], crossCuts = [];
  const counts = Array(K).fill(0), maskCount = new Map();
  let currentRun = 0, maxRun = 0, trills = 0, rolls = 0, previousDir = 0;
  const hand = mask => {
    let l = 0, r = 0;
    for (let c = 0; c < K; c++) if (mask & (1 << c)) {
      if (K === 7 && c === 3) { l += .5; r += .5; } else if (c < K / 2) l++; else r++;
    }
    return [l, r];
  };
  for (let i = 0; i < rs.length; i++) {
    const r = rs[i]; maskCount.set(r.mask, (maskCount.get(r.mask) ?? 0) + 1);
    for (let c = 0; c < K; c++) if (r.mask & (1 << c)) counts[c]++;
    if (!i) continue;
    const p = rs[i - 1], inter = pc(r.mask & p.mask);
    overlaps.push(inter / Math.max(1, Math.min(hits[i], hits[i - 1])));
    changes.push(pc(r.mask ^ p.mask) / K);
    currentRun = inter ? currentRun + 1 : 0; maxRun = Math.max(maxRun, currentRun);
    const [l, rr] = hand(r.mask), [pl, pr] = hand(p.mask);
    sameHands.push((l * pl + rr * pr) / Math.max(1, (l + rr) * (pl + pr)));
    // Within-hand finger substitutions; ignore the flexible 7K center finger.
    let cut = 0, pairs = 0;
    for (const range of [Array.from({length: Math.floor(K/2)}, (_, c) => c),
      Array.from({length: Math.floor(K/2)}, (_, c) => K - 1 - c)]) {
      for (let a = 0; a < range.length; a++) for (let b = a + 1; b < range.length; b++) {
        const x = 1 << range[a], y = 1 << range[b]; pairs++;
        if (((p.mask & x) && !(p.mask & y) && !(r.mask & x) && (r.mask & y)) ||
          ((p.mask & y) && !(p.mask & x) && !(r.mask & y) && (r.mask & x))) cut++;
      }
    }
    crossCuts.push(cut / Math.max(1, pairs));
    if (i >= 2) {
      const a = rs[i - 2]; returns.push(pc(a.mask & r.mask & ~p.mask) / Math.max(1, pc(a.mask)));
      if (hits[i] === 1 && hits[i-1] === 1 && hits[i-2] === 1 && r.mask === a.mask && r.mask !== p.mask) trills++;
    }
    if (hits[i] === 1 && hits[i-1] === 1) {
      const d = Math.sign(Math.log2(r.mask) - Math.log2(p.mask));
      if (d && d === previousDir) rolls++; previousDir = d;
    } else previousDir = 0;
  }
  const area = holds.map(hs => {
    let a = 0; const hi = lb(hs, end, h => h[0]);
    for (let i = hi - 1; i >= 0 && hs[i][1] > start; i--) a += Math.max(0, Math.min(end, hs[i][1]) - Math.max(start, hs[i][0]));
    return a / dur;
  });
  let heldSum = 0, locks = 0;
  for (const r of rs) {
    let heldMask = 0;
    for (let c = 0; c < K; c++) {
      const hs = holds[c], i = lb(hs, r.t, h => h[0]) - 1;
      if (i >= 0 && hs[i][0] < r.t && hs[i][1] > r.t) heldMask |= 1 << c;
    }
    heldSum += pc(heldMask) / K;
    if (heldMask && (r.mask & ~heldMask)) locks++;
  }
  let independent = 0, inverse = 0; const gaps = [];
  for (const t of ts) {
    const i = lb(rows, t.t, x => x.t), candidates = [rows[i-1], rows[i]].filter(Boolean);
    const nearest = candidates.length ? Math.min(...candidates.map(r => Math.abs(r.t - t.t))) : Infinity;
    if (nearest > ctx.rowMergeMs) independent++;
    const a = cols[t.col], next = a[lb(a, t.t, x => x.t)];
    if (next) { const gap = Math.max(0, next.t - t.t); gaps.push(gap / Math.max(1, med)); inverse += Math.exp(-gap / Math.max(1, med)); }
  }
  const total = notes.length, ln = notes.filter(n => n.lnEnd !== null), leftRight = counts.reduce((a, n, c) => {
    const [l, r] = hand(1 << c); a[0] += n*l; a[1] += n*r; return a;
  }, [0,0]);
  const entropy = (counts, total, max) => total && max > 1 ? -counts.reduce((s,n) => n ? s + n/total * Math.log(n/total) : s,0) / Math.log(max) : 0;
  const residuals = [];
  for (const r of rs) {
    const i = Math.max(0, lb(map.timingPoints, r.t + 1e-8, p => p.t) - 1), tp = map.timingPoints[i];
    if (tp) { const unit = tp.beatLength / 4; const q = (r.t - tp.t) / unit; residuals.push(Math.abs(q - Math.round(q))); }
  }
  const bins = Array(Math.max(1, Math.ceil(seconds))).fill(0);
  for (const n of notes) bins[Math.min(bins.length - 1, Math.floor((n.t - start) / 1000))]++;
  const nps = total / seconds, cv = mean(iv) ? sd(iv)/mean(iv) : 0;
  const f = {
    npsPerKey: nps/K, rowRate: rs.length/seconds, chordFraction: mean(hits)/K,
    multiFraction: frac(hits,x=>x>=2), singleFraction: frac(hits,x=>x===1), lnHeadFraction: total ? ln.length/total : 0,
    overlap: mean(overlaps), repeatFraction: frac(overlaps,x=>x>0), cutReturn: mean(returns), chordChange: mean(changes),
    columnEntropy: entropy(counts,total,K), handImbalance: total ? Math.abs(leftRight[0]-leftRight[1])/total : 0,
    sameHand: mean(sameHands), crossCut: mean(crossCuts), logIoiSd: sd(iv.map(Math.log)),
    ioiChange: mean(iv.slice(1).map((v,i)=>Math.abs(Math.log(v/iv[i])))), graceFraction: med ? frac(iv,x=>x<med/2) : 0,
    gridResidual: mean(residuals), rowBpm: med ? 15000/med : 0, ioiCv: cv,
    occupancy: mean(area), minOccupancy: Math.min(...area), occupancySd: sd(area),
    heldAtHead: rs.length ? heldSum/rs.length : 0, lockFraction: rs.length ? locks/rs.length : 0,
    releaseIndependence: ts.length ? independent/ts.length : 0, releaseScatter: sd(gaps.map(x=>Math.log1p(x))),
    tailRate: ts.length/seconds/K, shortHoldFraction: ln.length ? frac(ln,n=>n.lnEnd-n.t<=2*Math.max(1,med)) : 0,
    inverseGap: ts.length ? inverse/ts.length : 0,
    lnHeadOverlap: mean(rs.slice(1).map((r,i)=>pc(r.lnMask&rs[i].lnMask)/Math.max(1,pc(r.lnMask)))),
    anchorShare: total ? Math.max(...counts)/total : 0, repeatRun: rs.length>1 ? maxRun/(rs.length-1) : 0,
    maskEntropy: entropy([...maskCount.values()],rs.length,Math.min(2**K-1,rs.length)),
    trillFraction: rs.length>2 ? trills/(rs.length-2) : 0, rollFraction: rs.length>2 ? rolls/(rs.length-2) : 0,
    effort: nps/K * (1+mean(overlaps)) * (1+mean(area)),
    sustain: frac(bins,x=>x>=mean(bins)), burstiness: mean(bins) ? sd(bins)/mean(bins) : 0,
    activeFraction: frac(bins,x=>x>0)
  };
  return {start,end,values: FEATURE_KEYS.map(k=>f[k]),metrics:f,notes:total,rows:rs.length,
    equivBpm: iv.length>=3 && cv<=.5 ? Math.round(15000/med) : undefined};
}
function windows(ctx, windowMs = 4000, stepMs = 2000) {
  const out = [], {firstNote, lastNote} = ctx.map;
  const end = Math.max(firstNote+1,lastNote+1);
  for (let t = firstNote; t < end; t += stepMs) {
    const center = Math.min(end,t+stepMs/2);
    const a = Math.max(firstNote,center-windowMs/2), b = Math.min(end,center+windowMs/2);
    out.push({...features(ctx,a,b),start:t,end:Math.min(t+stepMs,end)});
  }
  return out;
}
module.exports = { GROUPS, FEATURE_KEYS, features, windows, mean, quantile };
