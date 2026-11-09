'use strict';
const fs = require('node:fs'), path = require('node:path');
const { records7 } = require('../rework/data.cjs');
const { parseOsu, prepare } = require('../../src/parser.cjs');
const { alternatingFixedGrips } = require('./fixed-grips.cjs');
const output = path.join(__dirname, 'fixed-grips-inventory-r1.json');
if (fs.existsSync(output)) throw new Error('Preserve fixed-grip inventory');
const old = new Map(JSON.parse(fs.readFileSync('.cache/rework-legacy-full.json', 'utf8')).map(r => [r.uid, r]));
const matches = [];
for (const record of records7()) {
  const ctx = prepare(parseOsu(fs.readFileSync(record.path, 'utf8')));
  for (const s of old.get(record.uid).segments) {
    const evidence = alternatingFixedGrips(ctx, s.start, s.end);
    if (!evidence) continue;
    matches.push({ uid: record.uid, source: record.path, ...s, evidence,
      annotations: record.annotations?.filter(a => a.start < s.end && a.end > s.start) });
  }
}
const constructs = ['H01', 'H02', 'H03', 'H04'].map(id => {
  const ctx = prepare(parseOsu(fs.readFileSync(`artifacts/frontier-annotation-r7/${id}.osu`, 'utf8')));
  return { id, evidence: alternatingFixedGrips(ctx, 9000, 17000) };
});
const report = { scope: 'Exact alternating fixed-grip support only; no fitting, class replacement or automatic Tech labels.',
  matches, constructs, csEligible: matches.filter(s => ['dense chordstream', 'fast chordstream'].includes(s.tag)) };
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
