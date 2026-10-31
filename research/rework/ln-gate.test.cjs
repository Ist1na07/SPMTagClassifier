const test=require('node:test'),assert=require('node:assert/strict');
const {selectNativeSegments}=require('./native-policy.cjs'),{CLASSES}=require('./native.cjs');
test('extra gate protection retains LN but neither promotes Break nor rescues sparse RC',()=>{
 const row=(i,tag,family,burden)=>({start:i*2000,end:(i+1)*2000,tag,family,confidence:1,burden,probability:CLASSES.map(c=>Number(c===tag))});
 const rows=[row(0,'release','ln',1),row(1,'speed','rc',1),row(2,'break','break',1),row(3,'chordjack','rc',100)];rows[3].end=12000;
 const value={partitions:[{cases:[{raw:rows}]}]},base={partition:0,head:0,smoothing:'coherent',burdenGate:.3};
 const tags=extraGate=>selectNativeSegments(value,{...base,extraGate}).map(s=>s.tag);
 assert.deepEqual(tags('all'),['break','break','break','chordjack']);
 assert.deepEqual(tags('ln_protected'),['release','break','break','chordjack']);
 assert.deepEqual(tags('off'),['release','speed','break','chordjack']);
 assert.deepEqual(selectNativeSegments(value,base),selectNativeSegments(value,{...base,extraGate:'all'}));
});
