const state = {game:null, play:null, shot:null, shotActions:[], tracks:[], frames:[], frame:0, timer:null, activeSource:null, calibrationCurrent:null, calibrationKeyframes:[], fieldDebug:null, detections:[], calibrationMatrices:[], frameDetections:new Map()};
const layerOn = id => $(id)?.checked;
const $ = id => document.getElementById(id);
const setHidden = (id, hidden) => { const node=$(id); if(node) node.hidden=hidden; };
async function json(url, options) { const r=await fetch(url,options); if(!r.ok) throw new Error(await r.text()); return r.json(); }

function sizeVideoStage() {
  const v=$('film'), stage=v.parentElement;
  if(!v.videoWidth||!v.videoHeight)return;
  const available=stage.parentElement.clientWidth;
  const maximumHeight=window.innerHeight*.42;
  stage.style.width=`${Math.min(available,maximumHeight*v.videoWidth/v.videoHeight)}px`;
}

function field() {
  const c=$('field'), x=c.getContext('2d'); x.clearRect(0,0,c.width,c.height); x.fillStyle='#1b673c'; x.fillRect(0,0,c.width,c.height);
  x.strokeStyle='rgba(255,255,255,.8)'; x.lineWidth=2;
  for(let yd=10;yd<=110;yd+=5){x.beginPath();x.moveTo(yd*10,0);x.lineTo(yd*10,c.height);x.stroke();}
  x.fillStyle='rgba(255,255,255,.9)';x.font='18px system-ui';
  for(let yd=20;yd<=100;yd+=10)x.fillText(String(yd<=60?yd-10:110-yd),yd*10-10,30);
}
function currentRows() {
  const key=state.frames[state.frame];
  return state.tracks.filter(r => state.shot ? Number(r.video_timestamp)===Number(key) : Number(r.frame_id)===Number(key));
}
// Interpolate the stored image-to-field homographies to a timestamp, matching
// how projection does it, so the drawn grid is the geometry actually in use.
function matrixAt(timestamp){
  const keys=state.calibrationMatrices;
  if(!keys.length)return null;
  // A homography is projective geometry, not an array of independent numbers.
  // Element-wise interpolation can become singular and make the grid fly across
  // the image. Until temporal registration produces a matrix at this exact
  // frame, show the nearest verified keyframe honestly rather than inventing a
  // matrix that was never calibrated.
  return keys.reduce((best,key)=>Math.abs(key.timestamp_s-timestamp)<Math.abs(best.timestamp_s-timestamp)?key:best).matrix;
}
function invert3(m){
  const [[a,b,c],[d,e,f],[g,h,i]]=m;
  const A=e*i-f*h,B=-(d*i-f*g),C=d*h-e*g;
  const det=a*A+b*B+c*C;
  if(!det||!isFinite(det))return null;
  return [[A/det,(c*h-b*i)/det,(b*f-c*e)/det],
          [B/det,(a*i-c*g)/det,(c*d-a*f)/det],
          [C/det,(b*g-a*h)/det,(a*e-b*d)/det]];
}
function applyH(m,x,y){
  const w=m[2][0]*x+m[2][1]*y+m[2][2];
  if(!w)return null;
  return [(m[0][0]*x+m[0][1]*y+m[0][2])/w,(m[1][0]*x+m[1][1]*y+m[1][2])/w];
}
// Draw the official field template back onto the film. If the registration sat
// on the wrong yard lines, the painted lines and the drawn grid separate by a
// visible five yards, which a residual number cannot show.
function drawFieldGrid(x,timestamp,width,height){
  const forward=matrixAt(timestamp);if(!forward)return false;
  const inverse=invert3(forward);if(!inverse)return false;
  const visible=([px,py])=>px>-width&&px<2*width&&py>-height&&py<2*height;
  x.lineWidth=2;
  for(let yd=0;yd<=120;yd+=5){
    const a=applyH(inverse,yd,0),b=applyH(inverse,yd,160/3);
    if(!a||!b||!(visible(a)||visible(b)))continue;
    const major=yd%10===0;
    x.strokeStyle=major?'rgba(255,90,160,.95)':'rgba(255,90,160,.45)';
    x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();
    if(major&&layerOn('layer-labels')){
      const label=yd<=60?Math.max(0,yd-10):110-yd;
      x.fillStyle='rgba(255,90,160,.95)';x.font='bold 15px system-ui';
      x.fillText(String(label),a[0]+4,Math.max(15,a[1]-4));
    }
  }
  for(const yline of [0,160/3]){
    const a=applyH(inverse,0,yline),b=applyH(inverse,120,yline);
    if(a&&b){x.strokeStyle='rgba(255,90,160,.8)';x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();}
  }
  return true;
}
function drawOverlay(rows) {
  const c=$('overlay'), v=$('film'), x=c.getContext('2d');
  c.width=v.videoWidth||1280; c.height=v.videoHeight||720; x.clearRect(0,0,c.width,c.height);
  const time=v.currentTime;
  const labels=layerOn('layer-labels');
  const notes=[];
  if(layerOn('layer-detections')){
    // Nearest sampled detection frame; detections are stored at 10 Hz.
    let best=null,bestGap=Infinity;
    for(const key of state.frameDetections.keys()){
      const gap=Math.abs(key-time);
      if(gap<bestGap){bestGap=gap;best=key;}
    }
    if(best!=null&&bestGap<=.2){
      const items=state.frameDetections.get(best);
      x.lineWidth=2;
      for(const d of items){
        const alpha=Math.max(.25,Math.min(1,Number(d.confidence??1)));
        x.strokeStyle=`rgba(160,160,170,${alpha})`;
        x.strokeRect(Number(d.x1),Number(d.y1),Number(d.x2-d.x1),Number(d.y2-d.y1));
        if(labels){x.fillStyle=`rgba(200,200,210,${alpha})`;x.font='12px system-ui';
          x.fillText(Number(d.confidence??0).toFixed(2),Number(d.x1),Number(d.y2)+12);}
      }
      notes.push(`${items.length} detections @ ${best.toFixed(2)}s`);
    }
  }
  if(layerOn('layer-tracks')){
    for(const r of rows){
      const x1=r.bbox_x1??r.x1, y1=r.bbox_y1??r.y1, x2=r.bbox_x2??r.x2, y2=r.bbox_y2??r.y2;
      if(x1==null) continue;
      x.strokeStyle=r.team==='team_0'?'#67d5ff':r.team==='team_1'?'#ffd166':r.team==='official'?'#d6a4ff':'#9aa0a6';
      x.lineWidth=3;
      x.strokeRect(Number(x1),Number(y1),Number(x2-x1),Number(y2-y1));
      if(labels){x.fillStyle=x.strokeStyle;x.font='16px system-ui';
        x.fillText(String(r.track_id||'').split(':').at(-1),Number(x1),Math.max(16,Number(y1)-4));}
    }
    if(rows.length)notes.push(`${rows.length} tracks`);
  }
  if(layerOn('layer-grid')&&drawFieldGrid(x,time,c.width,c.height))notes.push('calibrated grid');
  const debug=state.fieldDebug;
  if(debug&&Math.abs(debug.timestamp_s-time)<=.35){
    if(layerOn('layer-all-lines')){
      x.lineWidth=1.5;
      for(const line of debug.line_evidence||[]){
        const [a,b]=line.image_points;
        // Teal: cross-field candidates; yellow: sidelines/boundaries; grey: everything else.
        x.strokeStyle=line.kind==='cross_field'?'rgba(80,255,220,.62)':line.kind==='downfield_boundary'?'rgba(255,225,70,.7)':'rgba(220,220,220,.35)';
        x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();
        if(labels&&line.length_pixels>100){
          x.fillStyle=x.strokeStyle;x.font='11px system-ui';
          x.fillText(`${line.kind} ${line.length_pixels.toFixed(0)}px`,(a[0]+b[0])/2,(a[1]+b[1])/2);
        }
      }
      notes.push(`${(debug.line_evidence||[]).length} fitted candidates from ${debug.raw_line_fragment_count||0} raw fragments`);
    }
    if(layerOn('layer-lines')){
      x.lineWidth=3;
      for(const line of debug.lines){
        const [a,b]=line.image_points;
        x.strokeStyle=line.assigned_field_x==null?'rgba(255,120,120,.95)':'rgba(90,200,255,.95)';
        x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();
        if(labels&&line.assigned_field_x!=null){
          x.fillStyle='#bfe9ff';x.font='bold 15px system-ui';
          x.fillText(`x=${line.assigned_field_x.toFixed(0)}`,a[0]+4,a[1]+18);
        }
      }
      notes.push(`${debug.lines.length} detected lines`);
    }
    if(layerOn('layer-numbers')){
      for(const n of debug.numbers){
        const [cx,cy]=n.center;
        x.strokeStyle=n.trusted?'rgba(130,255,150,.95)':'rgba(255,190,80,.95)';
        x.lineWidth=3;x.beginPath();x.arc(cx,cy,14,0,Math.PI*2);x.stroke();
        if(labels){
          x.fillStyle=n.trusted?'#b7ffc4':'#ffd79a';x.font='bold 15px system-ui';
          x.fillText(`${n.value} ${n.confidence.toFixed(2)}`,cx+18,cy+5);
          x.font='12px system-ui';
          x.fillText(`${n.candidate_field_x.join(' or ')}`,cx+18,cy+21);
        }
      }
      notes.push(`${debug.numbers.length} numbers`);
    }
  } else if(debug&&(layerOn('layer-lines')||layerOn('layer-numbers'))){
    notes.push(`detections are for ${debug.timestamp_s.toFixed(2)}s — re-inspect at this time`);
  }
  const annotations=state.calibrationCurrent?.annotations||[];
  for(const [index,annotation] of annotations.entries()){
    const points=annotation.kind==='line'?annotation.image_points:[annotation.image_point];
    if(annotation.kind==='line'){x.strokeStyle='#ff4f87';x.lineWidth=3;x.beginPath();x.moveTo(...points[0]);x.lineTo(...points[1]);x.stroke();}
    for(const point of points){x.beginPath();x.fillStyle='#ff4f87';x.arc(point[0],point[1],7,0,Math.PI*2);x.fill();}
    x.fillStyle='white';x.fillText(annotation.kind==='number'?String(annotation.value):`L${index+1}`,points[0][0]+9,points[0][1]-9);
  }
  const pending=state.calibrationCurrent?.pendingLine;
  if(pending){x.beginPath();x.fillStyle='#ff4f87';x.arc(pending[0],pending[1],7,0,Math.PI*2);x.fill();}
  const status=$('layer-status');if(status)status.textContent=notes.join(' · ');
}
function draw() {
  field(); const rows=currentRows(), c=$('field'), x=c.getContext('2d');
  for(const r of rows){
    const px=Number(r.field_x??r.x), py=Number(r.field_y??r.y); x.beginPath();
    x.fillStyle=(r.team||'').toLowerCase().includes('home')||r.team==='team_0'?'#67d5ff':'#ffd166';x.arc(px*10,py*10,10,0,Math.PI*2);x.fill();
    x.fillStyle='#08110d';x.font='bold 12px system-ui';x.textAlign='center';x.fillText(r.jersey_number??String(r.track_id||'').split(':').at(-1)??'?',px*10,py*10+4);
  }
  drawOverlay(rows); const t=state.shot?(rows[0]?.video_timestamp??state.frames[state.frame]??0):(rows[0]?.t??0); $('clock').textContent=`${Number(t).toFixed(1)}s`;
}
function configureTimeline(keys) {
  state.frames=keys; state.frame=0; $('timeline').min=0; $('timeline').max=Math.max(0,keys.length-1); $('timeline').value=0; draw();
}
function updateCalibrationText(){
  $('calibration-json').value=JSON.stringify({keyframes:state.calibrationKeyframes},null,2);
  $('landmark-count').textContent=`${state.calibrationCurrent?.annotations.length||0} annotations${state.calibrationCurrent?` at ${state.calibrationCurrent.timestamp_s.toFixed(2)}s`:''}`;
}
function loadActionForm(){
  const action=state.shotActions[Number($('action-select').value)||0];if(!action)return;
  $('action-formation').value=action.formation_start_s;$('action-snap').value=action.snap_s;$('action-dead').value=action.dead_s;$('action-end').value=action.playback_end_s;
  $('film').currentTime=Number(action.formation_start_s);draw();
}

async function loadGames(){
  const games=await json('/api/games'); $('games').innerHTML=games.map(g=>`<option value="${g.game_id}">${g.game_id}</option>`).join('');
  if(games.length){state.game=games[0].game_id;await loadWorkspace();}
}
async function loadWorkspace(){
  const shots=$('workspace-mode').value==='shots'; $('queue-help').textContent=shots?'Camera shots are processed independently before repeated views are paired.':'Only high-value, unreviewed plays are shown. Saving advances to the next one.';
  if(shots) await loadShots(); else await loadPlays();
}
async function loadShots(){
  state.game=$('games').value;state.play=null;const allShots=await json(`/api/games/${state.game}/shots`);const detected=allShots.filter(shot=>shot.artifacts.some(item=>item.kind==='clip_detections'));const shots=detected.length?detected:allShots;$('plays').innerHTML='';
  for(const shot of shots){const b=document.createElement('button');b.className='play';b.innerHTML=`<small>${shot.angle} · ${Number(shot.start_s).toFixed(1)}–${Number(shot.end_s).toFixed(1)}s</small>${shot.clip_id}<small>${shot.action_count} actions · ${shot.calibration_keyframes} calibration keys</small>`;b.onclick=()=>selectShot(shot,b);$('plays').appendChild(b);}
  $('status').textContent=`${shots.length} camera shots`;if(shots.length)await selectShot(shots[0],$('plays').firstChild);
}
async function selectShot(shot,button){
  document.querySelectorAll('.play').forEach(x=>x.classList.remove('active'));button?.classList.add('active');state.shot=shot;state.play=null;state.activeSource=null;state.tracks=[];
  $('play-title').textContent=`${shot.angle} shot`;$('play-detail').textContent=`${shot.clip_id} · ${Number(shot.end_s-shot.start_s).toFixed(1)} seconds`;$('film').src=`/api/video/${state.game}`;
  $('audit').hidden=true;$('timing').hidden=true;$('calibration').hidden=false;$('calibration-status').textContent='';state.calibrationCurrent=null;
  setHidden('field-debug',false);$('debug-status').textContent='';$('debug-time').value=Number(shot.start_s).toFixed(2);state.fieldDebug=null;drawFieldDebug();
  const savedCalibration=await json(`/api/clips/${encodeURIComponent(shot.clip_id)}/calibration`);state.calibrationKeyframes=savedCalibration.keyframes.map(k=>k.annotations?.length?{timestamp_s:k.timestamp_s,annotations:k.annotations}:{timestamp_s:k.timestamp_s,image_points:k.image_points,field_points:k.field_points});updateCalibrationText();
  state.calibrationMatrices=savedCalibration.keyframes.filter(k=>Array.isArray(k.matrix)).map(k=>({timestamp_s:Number(k.timestamp_s),matrix:k.matrix}));
  state.detections=[];state.frameDetections=new Map();
  try{
    state.detections=await json(`/api/clips/${encodeURIComponent(shot.clip_id)}/detections`);
    for(const d of state.detections){
      const key=Number(Number(d.video_timestamp).toFixed(3));
      if(!state.frameDetections.has(key))state.frameDetections.set(key,[]);
      state.frameDetections.get(key).push(d);
    }
  }catch{}
  const actions=await json(`/api/clips/${encodeURIComponent(shot.clip_id)}/actions`);state.shotActions=actions;$('sources').innerHTML='';
  for(const action of actions){const b=document.createElement('button');b.textContent=`Action ${action.action_order} · ${Number(action.snap_s).toFixed(1)}–${Number(action.dead_s).toFixed(1)}s · ${action.status}`;b.onclick=()=>seekSource({start_s:action.formation_start_s,end_s:action.playback_end_s,snap_s:action.snap_s,play_end_s:action.dead_s},b);$('sources').appendChild(b);}
  $('action-review').hidden=!actions.length;$('action-select').innerHTML=actions.map((a,i)=>`<option value="${i}">Action ${a.action_order} · ${a.status}</option>`).join('');$('action-status').textContent='';loadActionForm();
  try{state.tracks=await json(`/api/clips/${encodeURIComponent(shot.clip_id)}/tracks?stride=2`);}catch{}
  configureTimeline([...new Set(state.tracks.map(r=>Number(r.video_timestamp)))].sort((a,b)=>a-b));
  $('diagnostics').textContent=JSON.stringify({shot,actions,track_rows:state.tracks.length},null,2);
  const seek=()=>{$('film').currentTime=Number(shot.start_s);};if($('film').readyState>=1)seek();else $('film').addEventListener('loadedmetadata',seek,{once:true});
}

async function loadPlays(){
  state.game=$('games').value;const plays=await json(`/api/games/${state.game}/audit-queue?limit=30`);$('plays').innerHTML='';let first=null;
  for(const p of plays){const b=document.createElement('button');b.className=`play ${p.processing_status||''}`;b.innerHTML=`<small>Q${p.quarter??'?'} ${p.clock??''} · ${p.down_no??'?'} & ${p.distance??'?'}</small>${p.description}<small>Review: ${(p.review_reasons||[]).join(' · ')}</small>`;b.onclick=()=>selectPlay(p,b);$('plays').appendChild(b);if(!first)first=[p,b];}
  $('status').textContent=plays.length?`${plays.length} important reviews queued`:'Important review queue is clear';if(first)await selectPlay(first[0],first[1]);
}
function sourceStart(source){return Number(source.snap_s??source.start_s);}
function sourceEnd(source){return Number(source.play_end_s??source.end_s);}
function seekSource(source,button){
  const film=$('film');state.activeSource=source;document.querySelectorAll('.sources button').forEach(x=>x.classList.remove('active'));button?.classList.add('active');
  const seek=()=>{film.currentTime=sourceStart(source);film.play();};if(film.readyState>=1)seek();else film.addEventListener('loadedmetadata',seek,{once:true});
}
function loadTimingSource(){const sources=state.play?.sources||[],source=sources[Number($('timing-source').value)||0];if(!source)return;$('timing-snap').value=source.snap_s??source.start_s;$('timing-end').value=source.play_end_s??source.end_s;}
async function selectPlay(p,button){
  document.querySelectorAll('.play').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.shot=null;state.play=p;state.tracks=[];state.activeSource=null;$('calibration').hidden=true;$('action-review').hidden=true;setHidden('field-debug',true);$('overlay').classList.remove('calibrating');
  $('play-title').textContent=`Q${p.quarter??'?'} ${p.clock??''} — ${p.possession??''}`;$('play-detail').textContent=p.description;$('diagnostics').textContent=JSON.stringify({status:p.processing_status,reasons:p.reasons,metrics:p.metrics},null,2);$('film').src=`/api/video/${state.game}`;
  const sources=p.sources||[];$('sources').innerHTML='';let firstButton=null;
  for(const [i,source] of sources.entries()){const b=document.createElement('button');if(!firstButton)firstButton=b;b.textContent=`Source ${i+1} · ${source.angle} · ${(sourceEnd(source)-sourceStart(source)).toFixed(1)}s action`;b.onclick=()=>seekSource(source,b);$('sources').appendChild(b);}
  if(sources.length)seekSource(sources[0],firstButton);$('timing').hidden=!sources.length;$('timing-source').innerHTML=sources.map((s,i)=>`<option value="${i}">Source ${i+1} · ${s.angle}</option>`).join('');$('timing-status').textContent='';loadTimingSource();
  const audit=p.audit||{};$('audit').hidden=!audit.selected;$('audit-mapping').checked=audit.mapping_correct===1;$('audit-sources').checked=audit.sources_correct===1;$('audit-timing').checked=audit.timing_correct===1;$('audit-notes').value=audit.notes||'';$('audit-status').textContent='';
  if(p.play_id){try{state.tracks=await json(`/api/plays/${p.play_id}/tracks`);}catch{}}
  configureTimeline([...new Set(state.tracks.map(r=>Number(r.frame_id)))].sort((a,b)=>a-b));
}

$('film').addEventListener('timeupdate',()=>{
  if(state.shot&&state.frames.length){let best=0;for(let i=1;i<state.frames.length;i++)if(Math.abs(state.frames[i]-$('film').currentTime)<Math.abs(state.frames[best]-$('film').currentTime))best=i;state.frame=best;$('timeline').value=best;draw();}
  if(state.activeSource&&$('film').currentTime>=sourceEnd(state.activeSource)){$('film').pause();$('film').currentTime=sourceEnd(state.activeSource);}
});
$('film').addEventListener('loadedmetadata',()=>{sizeVideoStage();draw();});
window.addEventListener('resize',()=>{sizeVideoStage();draw();});
$('audit-save').onclick=async()=>{if(!state.play?.play_id)return;$('audit-status').textContent='Saving…';try{await json(`/api/plays/${state.play.play_id}/audit`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({mapping_correct:$('audit-mapping').checked,sources_correct:$('audit-sources').checked,timing_correct:$('audit-timing').checked,notes:$('audit-notes').value})});await loadPlays();}catch(e){$('audit-status').textContent=e.message;}};
$('timing-source').onchange=loadTimingSource;
$('timing-save').onclick=async()=>{const index=Number($('timing-source').value)||0,source=state.play?.sources?.[index];if(!source)return;$('timing-status').textContent='Saving…';try{const result=await json(`/api/clips/${encodeURIComponent(source.clip_id)}/timing`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({snap_s:Number($('timing-snap').value),play_end_s:Number($('timing-end').value)})});source.snap_s=result.snap_s;source.play_end_s=result.play_end_s;$('timing-status').textContent='Saved';}catch(e){$('timing-status').textContent=e.message;}};
$('landmark-mode').onchange=()=>{$('landmark-number-label').hidden=$('landmark-mode').value!=='number';};
$('landmark-start').onclick=()=>{state.calibrationCurrent={timestamp_s:$('film').currentTime,mode:$('landmark-mode').value,annotations:[],pendingLine:null};$('overlay').classList.add('calibrating');$('film').pause();updateCalibrationText();draw();};
$('overlay').onclick=e=>{if(!state.calibrationCurrent)return;const rect=$('overlay').getBoundingClientRect(),point=[(e.clientX-rect.left)*$('overlay').width/rect.width,(e.clientY-rect.top)*$('overlay').height/rect.height];if(state.calibrationCurrent.mode==='number'){state.calibrationCurrent.annotations.push({kind:'number',value:Number($('landmark-number').value),image_point:point});}else if(state.calibrationCurrent.pendingLine){state.calibrationCurrent.annotations.push({kind:'line',image_points:[state.calibrationCurrent.pendingLine,point]});state.calibrationCurrent.pendingLine=null;}else state.calibrationCurrent.pendingLine=point;updateCalibrationText();draw();};
$('landmark-undo').onclick=()=>{if(!state.calibrationCurrent)return;if(state.calibrationCurrent.pendingLine)state.calibrationCurrent.pendingLine=null;else state.calibrationCurrent.annotations.pop();updateCalibrationText();draw();};
$('landmark-add').onclick=()=>{const current=state.calibrationCurrent,minimum=current?.mode==='line'?2:4;if(!current||current.pendingLine||current.annotations.length<minimum){$('calibration-status').textContent=current?.mode==='line'?'Click both sideline intersections on at least two yard lines.':'Click at least four painted numbers across both number rows.';return;}state.calibrationKeyframes.push({timestamp_s:current.timestamp_s,annotations:current.annotations});state.calibrationKeyframes.sort((a,b)=>a.timestamp_s-b.timestamp_s);state.calibrationCurrent=null;$('overlay').classList.remove('calibrating');updateCalibrationText();draw();};
$('calibration-save').onclick=async()=>{if(!state.shot)return;$('calibration-status').textContent='Saving…';try{const body=JSON.parse($('calibration-json').value);await json(`/api/clips/${encodeURIComponent(state.shot.clip_id)}/calibration`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});$('calibration-status').textContent='Saved; re-run projection and tracking.';}catch(e){$('calibration-status').textContent=e.message;}};
$('calibration-auto').onclick=async()=>{if(!state.shot)return;$('calibration-auto').disabled=true;$('calibration-status').textContent='Detecting lines, reading numbers, and reconstructing movement…';try{const result=await json(`/api/clips/${encodeURIComponent(state.shot.clip_id)}/auto-calibrate`,{method:'POST'});const actions=result.reconstruction?.actions?.length??0;$('calibration-status').textContent=`Done: ${result.keyframes.length} field keyframes and ${actions} action candidates. Reopen this shot to inspect them.`;}catch(e){$('calibration-status').textContent=e.message;}finally{$('calibration-auto').disabled=false;}};
$('action-select').onchange=loadActionForm;
$('action-save').onclick=async()=>{const action=state.shotActions[Number($('action-select').value)||0];if(!action)return;$('action-status').textContent='Saving…';try{const body={formation_start_s:Number($('action-formation').value),snap_s:Number($('action-snap').value),dead_s:Number($('action-dead').value),playback_end_s:Number($('action-end').value)};const result=await json(`/api/actions/${encodeURIComponent(action.action_id)}/timing`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});Object.assign(action,result);$('action-status').textContent='Verified';$('action-select').options[$('action-select').selectedIndex].text=`Action ${action.action_order} · verified`;}catch(e){$('action-status').textContent=e.message;}};
$('games').onchange=loadWorkspace;$('workspace-mode').onchange=loadWorkspace;
$('timeline').oninput=e=>{state.frame=Number(e.target.value);if(state.shot&&state.frames[state.frame]!=null)$('film').currentTime=state.frames[state.frame];draw();};
$('play-animation').onclick=()=>{if(state.timer){clearInterval(state.timer);state.timer=null;return;}state.timer=setInterval(()=>{state.frame++;if(state.frame>=state.frames.length){clearInterval(state.timer);state.timer=null;return;}$('timeline').value=state.frame;draw();},100);};
field();loadGames().catch(e=>$('status').textContent=e.message);


// --- Field detection debug -------------------------------------------------
// Yard lines repeat every five yards, so a registration can fit its lines
// perfectly and still sit on the wrong ones. This view shows the raw evidence
// behind that choice: each detected line with the yard value it was given, and
// each painted number with its OCR confidence and the two field positions it
// could legally mean.
function drawFieldDebug(){
  const canvas=$('debug-overlay'),image=$('debug-image'),data=state.fieldDebug;
  if(!data||!image.naturalWidth){canvas.width=canvas.height=0;return;}
  canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
  canvas.style.width=`${image.clientWidth}px`;canvas.style.height=`${image.clientHeight}px`;
  const x=canvas.getContext('2d');x.clearRect(0,0,canvas.width,canvas.height);
  x.lineWidth=Math.max(2,canvas.width/700);x.font=`${Math.max(16,canvas.width/70)}px system-ui`;
  for(const line of data.lines){
    const [a,b]=line.image_points;
    x.strokeStyle=line.assigned_field_x==null?'rgba(255,120,120,.95)':'rgba(90,200,255,.95)';
    x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();
    const label=line.assigned_field_x==null?`#${line.index} unassigned`:`#${line.index} x=${line.assigned_field_x.toFixed(0)}`;
    const top=a[1]<b[1]?a:b;
    x.fillStyle='rgba(0,0,0,.65)';const w=x.measureText(label).width+10;
    x.fillRect(top[0]-w/2,top[1]+4,w,26);
    x.fillStyle='#bfe9ff';x.fillText(label,top[0]-w/2+5,top[1]+24);
  }
  for(const number of data.numbers){
    const [cx,cy]=number.center;
    x.strokeStyle=number.trusted?'rgba(130,255,150,.95)':'rgba(255,190,80,.95)';
    x.beginPath();x.arc(cx,cy,Math.max(10,canvas.width/90),0,Math.PI*2);x.stroke();
    const label=`${number.value} (${number.confidence.toFixed(2)})`;
    x.fillStyle='rgba(0,0,0,.65)';const w=x.measureText(label).width+10;
    x.fillRect(cx+12,cy-14,w,26);
    x.fillStyle=number.trusted?'#b7ffc4':'#ffd79a';x.fillText(label,cx+17,cy+6);
  }
}
function renderFieldDebug(){
  const data=state.fieldDebug;if(!data)return;
  const verification=data.verification||{};
  const reasons=(verification.reasons||[]);
  const summary=[
    `<span class="${data.numbers_credible?'ok':'warn'}">numbers ${data.numbers_credible?'credible':'not credible'}</span>`,
    `<span class="${data.absolute_x?'ok':'warn'}">absolute x ${data.absolute_x?'resolved':'unresolved'}</span>`,
    `<span>${data.lines.length} lines · ${data.numbers.length} numbers</span>`,
    `<span>OCR assignment error ${data.ocr_assignment_error==null?'n/a':data.ocr_assignment_error.toFixed(2)}</span>`,
    `<span>confidence ${data.registration?.confidence==null?'n/a':Number(data.registration.confidence).toFixed(2)}</span>`,
    `<span class="${reasons.length?'warn':'ok'}">verification ${reasons.length?reasons.join('; '):'passed'}</span>`,
    data.ocr_unavailable?`<span class="warn">OCR unavailable: ${data.ocr_unavailable}</span>`:'',
  ].join('');
  $('debug-summary').innerHTML=summary;
  const rows=[`<tr><th>Kind</th><th>Detail</th><th>Confidence</th><th>Assigned / candidates</th><th>Nearest line</th></tr>`];
  for(const line of data.lines)
    rows.push(`<tr><td>line</td><td>#${line.index}</td><td>—</td><td>${line.assigned_field_x==null?'<em>none</em>':`x = ${line.assigned_field_x.toFixed(1)} yd`}</td><td>—</td></tr>`);
  for(const number of data.numbers)
    rows.push(`<tr class="${number.trusted?'':'untrusted'}"><td>number</td><td>${number.value}</td><td>${number.confidence.toFixed(3)}</td><td>${number.candidate_field_x.map(v=>`${v.toFixed(0)}`).join(' or ')} yd</td><td>${number.nearest_line_index==null?'—':`#${number.nearest_line_index}`}</td></tr>`);
  $('debug-table').innerHTML=rows.join('');
}
async function inspectFrame(){
  if(!state.shot)return;
  const time=Number($('debug-time').value);
  $('debug-status').textContent='Detecting lines and reading numbers…';
  try{
    const clip=encodeURIComponent(state.shot.clip_id);
    const data=await json(`/api/clips/${clip}/field-detections?timestamp_s=${time}`);
    state.fieldDebug=data;
    const image=$('debug-image');
    image.onload=()=>{drawFieldDebug();};
    image.src=`/api/clips/${clip}/frame.jpg?timestamp_s=${time}`;
    renderFieldDebug();
    $('debug-status').textContent=`${data.lines.length} lines, ${data.numbers.length} numbers at ${data.timestamp_s.toFixed(2)}s`;
  }catch(e){$('debug-status').textContent=e.message;}
}
$('debug-run').onclick=inspectFrame;
$('debug-here').onclick=()=>{$('debug-time').value=$('film').currentTime.toFixed(2);inspectFrame();};
window.addEventListener('resize',drawFieldDebug);


// Keep the overlay aligned with the film while it plays. requestAnimationFrame
// follows the real playback clock rather than the sampled track timeline.
let overlayFrame=null;
function followVideo(){
  const film=$('film');
  if(!film.paused&&!film.ended){
    if(state.shot&&state.frames.length){
      // Snap the field diagram to the nearest sampled track frame.
      let best=0,gap=Infinity;
      for(const [i,key] of state.frames.entries()){
        const d=Math.abs(Number(key)-film.currentTime);
        if(d<gap){gap=d;best=i;}
      }
      if(best!==state.frame){state.frame=best;$('timeline').value=best;}
    }
    draw();
  }
  overlayFrame=requestAnimationFrame(followVideo);
}
overlayFrame=requestAnimationFrame(followVideo);
for(const id of ['layer-tracks','layer-detections','layer-lines','layer-all-lines','layer-numbers','layer-grid','layer-labels'])
  $(id)?.addEventListener('change',()=>draw());
$('film').addEventListener('seeked',()=>draw());
$('film').addEventListener('loadeddata',()=>draw());
