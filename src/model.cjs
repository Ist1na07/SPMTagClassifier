'use strict';
function scores(model, vector) {
  const x = vector.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  const z = model.W.map((w, k) => (w.reduce((s, a, i) => s + a * x[i], model.b[k])) / (model.temperature || 1));
  if (model.multilabel) return z.map(v => 1 / (1 + Math.exp(-Math.max(-700, Math.min(700, v)))));
  const max = Math.max(...z), ex = z.map(v => Math.exp(v - max)), total = ex.reduce((a, b) => a + b, 0);
  return ex.map(v => v / total);
}
function named(classes, vector) { return Object.fromEntries(classes.map((c,i)=>[c,vector[i]])); }
function choose(classes, p, threshold) {
  const tags = classes.filter((_, i) => p[i] >= threshold);
  return tags.length ? tags : [classes[p.indexOf(Math.max(...p))]];
}
function linearQuantile(a, q) {
  if (!a.length) return 0;
  const s = [...a].sort((a,b)=>a-b), t=(s.length-1)*q, i=Math.floor(t);
  return s[i] + (s[Math.min(i+1,s.length-1)]-s[i])*(t-i);
}
module.exports = {scores,named,choose,linearQuantile};
