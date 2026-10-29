#!/usr/bin/env node
'use strict';
const fs = require('node:fs');
const { analyzeMap } = require('..');
const [file, rate = '1', algorithm] = process.argv.slice(2);
if (!file) { console.error('Usage: spm-tag-classifier map.osu [speedRate] [stable|modern|legacy]'); process.exit(2); }
try {
  const result = analyzeMap(fs.readFileSync(file, 'utf8'), {speedRate: Number(rate),algorithm});
  console.log(JSON.stringify({modelInfo:result.modelInfo,segments: result.segments, overall: result.overall}, null, 2));
} catch (err) { console.error(err.message); process.exit(1); }
