'use strict';
// Owned, offline Chrome instance only. No user browser or live tosu access.
const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),{spawn}=require('node:child_process'),assert=require('node:assert/strict');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const json=url=>new Promise((resolve,reject)=>http.get(url,r=>{let s='';r.on('data',c=>s+=c);r.on('end',()=>{try{resolve(JSON.parse(s));}catch(e){reject(e);}});}).on('error',reject));
async function main(){
 const chrome=['C:/Program Files/Google/Chrome/Application/chrome.exe','C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(fs.existsSync);
 if(!chrome)throw Error('Chrome/Edge unavailable');
 const profile=fs.mkdtempSync(path.resolve('.cache/perception-browser-'));
 const proc=spawn(chrome,['--headless=new','--disable-gpu','--hide-scrollbars','--remote-debugging-port=0','--user-data-dir='+profile,'--window-size=1150,900','about:blank'],{stdio:'ignore',windowsHide:true});
 let ws;
 try{
  const portFile=path.join(profile,'DevToolsActivePort');for(let i=0;i<100&&!fs.existsSync(portFile);i++)await sleep(100);
  const port=Number(fs.readFileSync(portFile,'utf8').split('\n')[0]),targets=await json(`http://127.0.0.1:${port}/json/list`);
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});
  const pending=new Map(),errors=[];let id=0;
  ws.onmessage=event=>{const m=JSON.parse(event.data);if(m.id&&pending.has(m.id)){pending.get(m.id)(m);pending.delete(m.id);}else if(m.method==='Runtime.exceptionThrown'||m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m);};
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const i=++id;pending.set(i,m=>m.error?reject(Error(JSON.stringify(m.error))):resolve(m.result));ws.send(JSON.stringify({id:i,method,params}));});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  await cdp('Page.enable');await cdp('Runtime.enable');await cdp('Log.enable');
  const url=JSON.parse(fs.readFileSync('.cache/perception-preview.json','utf8')).url;
  await cdp('Page.navigate',{url});
  for(let i=0;i<100;i++){if(await evaluate('!!window.perceptionQA?.state().tags'))break;await sleep(100);}
  await cdp('Emulation.setDeviceMetricsOverride',{width:430,height:900,deviceScaleFactor:1,mobile:false});
  await evaluate("document.querySelector('aside').style.display='none'");
  const rows=[];
  async function capture(name){
   const state=await evaluate('perceptionQA.state()');assert(state.tags?.percentages);assert(state.tags.perception.difficultyAvailable);
   const visual=await evaluate(`(()=>{const e=document.getElementById('container'),r=e.getBoundingClientRect();return {text:e.innerText,box:{x:r.x,y:r.y,width:r.width,height:r.height,scale:1},overflow:e.scrollWidth>e.clientWidth};})()`);
   assert(!visual.overflow);assert(visual.text.includes('负担占比'));rows.push({name,state,visual});
   const shot=await cdp('Page.captureScreenshot',{format:'png',clip:visual.box});fs.writeFileSync('research/perception/results/final-runtime/'+name+'.png',Buffer.from(shot.data,'base64'));
  }
  await capture('overlay-uniform');
  let s=await evaluate('perceptionQA.state()');assert(s.segments.every(s=>s.family==='break'||s.equivBpm===300));
  const before=JSON.stringify(s.tags);await evaluate("perceptionQA.smooth('Max')");assert.equal(JSON.stringify((await evaluate('perceptionQA.state()')).tags),before);
  await evaluate("perceptionQA.model('SPM Rating v1.2.0-Compact')");assert.equal((await evaluate('perceptionQA.state()')).profile.profile,'compact');await capture('overlay-compact');
  for(const fixture of ['F02','real0088','hybrid','B02']){
   await evaluate(`document.getElementById('qa-fixture').value='${fixture}';document.getElementById('qa-load').click()`);
   for(let i=0;i<100;i++){if(await evaluate("!document.getElementById('qa-status').textContent.includes('Calculating')"))break;await sleep(100);}
   await capture('overlay-'+fixture);
   const now=await evaluate('perceptionQA.state()');
   if(fixture==='hybrid')assert(now.tags.tags.includes('hybrid'));
   if(fixture==='B02')assert((await evaluate("document.getElementById('container').innerText")).includes('主要键型未定'));
  }
  // Recompute a real source with noninteger native clock (DT); display smoothing is downstream.
  await evaluate("document.getElementById('qa-fixture').value='real0088';document.getElementById('qa-rate').value='1.5';document.getElementById('qa-rate').dispatchEvent(new Event('change'))");
  for(let i=0;i<100;i++){if(await evaluate("!document.getElementById('qa-status').textContent.includes('Calculating')"))break;await sleep(100);}
  await capture('overlay-DT');
  assert.equal((await evaluate('perceptionQA.state()')).profile.profile,'compact');
  const applicationErrors=errors.filter(e=>!e.params?.entry?.url?.endsWith('/favicon.ico'));
  assert.equal(applicationErrors.length,0,JSON.stringify(applicationErrors));
  fs.writeFileSync('research/perception/results/final-runtime/overlay-check.json',JSON.stringify({transport:'actual upstream renderer; offline fixtures only',applicationErrors,networkDiagnostics:errors.filter(e=>!applicationErrors.includes(e)),checks:['raw curve consumed','model toggle','smoothing invariance','native DT clock','300 BPM regression','430px panel without horizontal overflow'],cases:rows},null,2)+'\n');
  console.log('Offline actual overlay checks passed:',rows.length);
 }finally{if(ws)ws.close();proc.kill();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
