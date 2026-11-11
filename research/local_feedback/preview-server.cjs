'use strict';
// Test-only transport driver for the ACTUAL upstream renderer. It never
// connects to the user's running tosu and never edits deployed plugin files.
const http = require('node:http'), fs = require('node:fs'), path = require('node:path');
const { replaceOnce } = require('../rework/legacy-lab.cjs');
const root = path.resolve(__dirname, '../..');
const patches=require('../../integration/patch-overlay.cjs');
const fixtures = {
  uniform50: null,
  tempoChange: '.cache/local-feedback-cadence-change.osu',
  H01: 'artifacts/frontier-annotation-r7/H01.osu',
  H02: 'artifacts/frontier-annotation-r7/H02.osu',
  F02: 'artifacts/frontier-annotation-r5/F02.osu',
  G01: 'artifacts/frontier-annotation-r6/G01.osu',
  E02: 'artifacts/frontier-annotation-r4/E02.osu',
  real0088: 'dataset/spm_dataset-1.1.1/beatmaps/bm_0088.osu',
  hybrid: 'dataset/spm_dataset-1.1.1/beatmaps/bm_0157.osu',
  B02: 'artifacts/frontier-annotation-r1/B02.osu',
};
function uniform() {
  return '[General]\nMode:3\n[Metadata]\nTitle:Cadence regression\nArtist:SPM\nCreator:Research\nVersion:Uniform 50 ms\n' +
    '[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n' +
    Array.from({length:200}, (_, i) => [0, 2, 4].map(c => `${(c+.5)*512/7},192,${1000+i*50},1,0,0:0:0:0:`).join('\n')).join('\n');
}
function html(engine) {
  let text = patches.patchHtml(fs.readFileSync(path.join(root, '.upstream/index.html'), 'utf8'));
  text = replaceOnce(text, '<script src="tag_engine.js"></script>', `<script src="tag_engine.js?engine=${engine}"></script>`);
  text = replaceOnce(text, 'fetch(`http://localhost:${TOSU_PORT}/files/beatmap/file`)',
    'fetch(`/fixture.osu?case=${encodeURIComponent(document.getElementById("qa-fixture").value)}`)');
  const controls = `<aside style="position:fixed;right:16px;top:16px;z-index:999;background:#17202e;color:#fff;padding:16px;width:270px;border-radius:8px;font:14px system-ui;line-height:1.7">
    <strong>Offline overlay regression</strong><br><small>Actual plugin renderer · simulated transport<br>Not connected to a game session</small>
    <label style="display:block">Fixture <select id="qa-fixture">${Object.keys(fixtures).map(k=>`<option>${k}</option>`).join('')}</select></label>
    <label style="display:block">Rate <select id="qa-rate"><option value="1">1×</option><option value="1.5">1.5× DT</option><option value="0.75">0.75× HT</option></select></label>
    <label style="display:block">Source time (s) <input id="qa-time" type="number" min="0" step="0.1" value="5" style="width:70px"></label>
    <button id="qa-load">Load / recalculate</button><br>
    <a href="/?engine=old" style="color:#9dd2ff">Archived engine</a> · <a href="/?engine=new" style="color:#9dd2ff">Working engine</a>
    <output id="qa-status" style="display:block"></output></aside>`;
  text = replaceOnce(text, '<body>', '<body>' + controls);
  const startup = `    showLoading('Connecting to tosu...');
    connectSettings();
    startSettingsPolling();
    loadDanConstants();
    connect();`;
  const driver = `    // Test-only driver: no websocket or background tosu polling.
    const qaRate=document.getElementById('qa-rate'),qaTime=document.getElementById('qa-time');
    const qaStatus=document.getElementById('qa-status');
    function qaCursor(){
        isPaused=true;songTimeMs=Number(qaTime.value)*1000/currentSpeedRate;
        prevSongTimeMs=songTimeMs;songTimeReceiveTs=performance.now();
        lastSegIndex=-2;if(cachedResult)showResult(cachedResult);updateSegmentLive(songTimeMs);
    }
    async function qaLoad(){
        qaStatus.textContent='Calculating…';currentSpeedRate=Number(qaRate.value);
        await fetchAndRate(currentSpeedRate);qaCursor();
        qaStatus.textContent='${engine === 'old' ? 'Archived' : 'Working'} engine · '+currentSpeedRate+'×';
    }
    qaTime.addEventListener('input',qaCursor);
    qaRate.addEventListener('change',qaLoad);
    document.getElementById('qa-load').addEventListener('click',qaLoad);
    window.perceptionQA={state:()=>({tags:cachedTagResult?.overall,segments:cachedTagResult?.segments,profile:cachedTagResult?.modelInfo?.perception,rating:cachedResult?.rating}),model:(name)=>{settings.algorithm=name;return renderRating(currentSpeedRate);},smooth:(value)=>{settings.curveSmoothing=value;rebuildCurves();showResult(cachedResult);}};
    qaLoad();`;
  return replaceOnce(text, startup, driver);
}
const server = http.createServer((request, response) => {
  const url = new URL(request.url, 'http://127.0.0.1');
  response.setHeader('Cache-Control', 'no-store');
  if (url.pathname === '/') {
    response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(html(url.searchParams.get('engine') === 'old' ? 'old' : 'new')); return;
  }
  if (url.pathname === '/fixture.osu') {
    const name = url.searchParams.get('case') ?? 'uniform50';
    if (!Object.hasOwn(fixtures, name)) { response.writeHead(404); response.end(); return; }
    response.setHeader('Content-Type', 'text/plain; charset=utf-8');
    response.end(fixtures[name] ? fs.readFileSync(path.join(root, fixtures[name])) : uniform()); return;
  }
  const files = {'/spm_algorithm.js':'.upstream/spm_algorithm.js', '/dan_constants.json':'.upstream/dan_constants.json',
    '/tag_engine.js':url.searchParams.get('engine') === 'old' ? '.upstream/tag_engine.js' : 'dist/tag-engine.js'};
  if (Object.hasOwn(files, url.pathname)) {
    response.setHeader('Content-Type', url.pathname.endsWith('.json') ? 'application/json' : 'text/javascript');
    const content=fs.readFileSync(path.join(root, files[url.pathname]));
    response.end(url.pathname==='/spm_algorithm.js'?patches.patchRating(content.toString()):content); return;
  }
  response.writeHead(404); response.end('Not found');
});
server.listen(0, '127.0.0.1', () => {
  const receipt = { pid: process.pid, url: `http://127.0.0.1:${server.address().port}/`,
    transport: 'Offline fixture driver; actual upstream renderer; no live tosu connection' };
  fs.writeFileSync(path.join(root, '.cache/local-feedback-preview.json'), JSON.stringify(receipt, null, 2) + '\n');
  console.log(receipt);
});
