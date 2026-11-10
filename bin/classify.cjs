#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const { analyzeMap } = require('..');
const [file, rate = '1', algorithm, aggregation, evidenceFile] = process.argv.slice(2);
if (!file) { console.error('Usage: spm-tag-classifier map.osu [speedRate] [stable|modern|legacy] [perceptual|classic] [rating-evidence.json]'); process.exit(2); }
try {
  const result = analyzeMap(fs.readFileSync(file, 'utf8'), {speedRate: Number(rate),algorithm,aggregation,
    ratingEvidence:evidenceFile?JSON.parse(fs.readFileSync(evidenceFile,'utf8')):undefined});
  console.log(JSON.stringify({modelInfo:result.modelInfo,segments: result.segments, overall: result.overall}, null, 2));
} catch (err) { console.error(err.message); process.exit(1); }
