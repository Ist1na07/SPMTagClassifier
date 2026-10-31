'use strict';
// Sufficient structural templates, not learned probability calibration.
// Numeric tolerance only absorbs floating-point IOI arithmetic under rate
// conversion; it is not an empirical rhythm/skill threshold.
function physicalSupport(window,mode='none'){
 if(!['none','jack','ln','both'].includes(mode))throw Error('Unknown physical support mode');
 if(mode==='none')return null;
 const required=['rowRate','occupancy','release:tailRatePerKey','lnHeadFraction','chordChange','overlap','ioiCv','rowBpm'];
 if(!Number.isFinite(window.contextStart)||!Number.isFinite(window.contextEnd)||!Number.isFinite(window.notes)||
    required.some(k=>!Number.isFinite(window.metrics[k])))throw Error('Physical support requires full finite native-window facts');
 const m=window.metrics,seconds=Math.max(1,window.contextEnd-window.contextStart)/1000;
 const rows=Math.round(m.rowRate*seconds),held=m.occupancy>0,tails=m['release:tailRatePerKey']>0;
 if((mode==='ln'||mode==='both')&&((window.notes===0&&(held||tails))||
    (rows<=1&&window.notes>0&&m.lnHeadFraction===1&&held)))return {tag:'ln',rule:'only_ln_activity_without_a_head_sequence'};
 if((mode==='jack'||mode==='both')&&rows>=3&&m.lnHeadFraction===0&&!held&&!tails&&
    m.chordChange===0&&m.overlap===1&&m.ioiCv<=1e-9&&m.rowBpm>0)
  return {tag:'jack',rule:'periodic_identical_rc_chord'};
 return null;
}
module.exports={physicalSupport};
