const state = {game:null, play:null, shot:null, shotActions:[], tracks:[], frames:[], frame:0, timer:null, activeSource:null, calibrationCurrent:null, calibrationKeyframes:[]};
const $ = id => document.getElementById(id);
async function json(url, options) { const r=await fetch(url,options); if(!r.ok) throw new Error(await r.text()); return r.json(); }

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
function drawOverlay(rows) {
  const c=$('overlay'), v=$('film'), x=c.getContext('2d'); c.width=v.videoWidth||1280; c.height=v.videoHeight||720; x.clearRect(0,0,c.width,c.height);
  for(const r of rows){
    const x1=r.bbox_x1??r.x1, y1=r.bbox_y1??r.y1, x2=r.bbox_x2??r.x2, y2=r.bbox_y2??r.y2;
    if(x1==null) continue;
    x.strokeStyle=r.team==='team_0'?'#67d5ff':'#ffd166'; x.lineWidth=3;
    x.strokeRect(Number(x1),Number(y1),Number(x2-x1),Number(y2-y1));
    x.fillStyle=x.strokeStyle;x.font='16px system-ui';x.fillText(String(r.track_id||''),Number(x1),Math.max(16,Number(y1)-4));
  }
  for(const [index,point] of (state.calibrationCurrent?.image_points||[]).entries()){
    x.beginPath();x.fillStyle='#ff4f87';x.arc(point[0],point[1],7,0,Math.PI*2);x.fill();x.fillStyle='white';x.fillText(String(index+1),point[0]+9,point[1]-9);
  }
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
  $('landmark-count').textContent=`${state.calibrationCurrent?.image_points.length||0} points${state.calibrationCurrent?` at ${state.calibrationCurrent.timestamp_s.toFixed(2)}s`:''}`;
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
  const savedCalibration=await json(`/api/clips/${encodeURIComponent(shot.clip_id)}/calibration`);state.calibrationKeyframes=savedCalibration.keyframes.map(k=>({timestamp_s:k.timestamp_s,image_points:k.image_points,field_points:k.field_points}));updateCalibrationText();
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
  document.querySelectorAll('.play').forEach(x=>x.classList.remove('active'));button.classList.add('active');state.shot=null;state.play=p;state.tracks=[];state.activeSource=null;$('calibration').hidden=true;$('action-review').hidden=true;$('overlay').classList.remove('calibrating');
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
$('audit-save').onclick=async()=>{if(!state.play?.play_id)return;$('audit-status').textContent='Saving…';try{await json(`/api/plays/${state.play.play_id}/audit`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({mapping_correct:$('audit-mapping').checked,sources_correct:$('audit-sources').checked,timing_correct:$('audit-timing').checked,notes:$('audit-notes').value})});await loadPlays();}catch(e){$('audit-status').textContent=e.message;}};
$('timing-source').onchange=loadTimingSource;
$('timing-save').onclick=async()=>{const index=Number($('timing-source').value)||0,source=state.play?.sources?.[index];if(!source)return;$('timing-status').textContent='Saving…';try{const result=await json(`/api/clips/${encodeURIComponent(source.clip_id)}/timing`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({snap_s:Number($('timing-snap').value),play_end_s:Number($('timing-end').value)})});source.snap_s=result.snap_s;source.play_end_s=result.play_end_s;$('timing-status').textContent='Saved';}catch(e){$('timing-status').textContent=e.message;}};
$('landmark-start').onclick=()=>{state.calibrationCurrent={timestamp_s:$('film').currentTime,image_points:[],field_points:[]};$('overlay').classList.add('calibrating');$('film').pause();updateCalibrationText();draw();};
$('overlay').onclick=e=>{if(!state.calibrationCurrent)return;const rawX=$('landmark-x').value,rawY=$('landmark-y').value,fieldX=Number(rawX),fieldY=Number(rawY);if(rawX===''||rawY===''||!Number.isFinite(fieldX)||!Number.isFinite(fieldY)){ $('calibration-status').textContent='Enter field X and Y before clicking.';return;}const rect=$('overlay').getBoundingClientRect();state.calibrationCurrent.image_points.push([(e.clientX-rect.left)*$('overlay').width/rect.width,(e.clientY-rect.top)*$('overlay').height/rect.height]);state.calibrationCurrent.field_points.push([fieldX,fieldY]);updateCalibrationText();draw();};
$('landmark-undo').onclick=()=>{state.calibrationCurrent?.image_points.pop();state.calibrationCurrent?.field_points.pop();updateCalibrationText();draw();};
$('landmark-add').onclick=()=>{if(!state.calibrationCurrent||state.calibrationCurrent.image_points.length<4){$('calibration-status').textContent='A keyframe needs at least four points.';return;}state.calibrationKeyframes.push(state.calibrationCurrent);state.calibrationKeyframes.sort((a,b)=>a.timestamp_s-b.timestamp_s);state.calibrationCurrent=null;$('overlay').classList.remove('calibrating');updateCalibrationText();draw();};
$('calibration-save').onclick=async()=>{if(!state.shot)return;$('calibration-status').textContent='Saving…';try{const body=JSON.parse($('calibration-json').value);await json(`/api/clips/${encodeURIComponent(state.shot.clip_id)}/calibration`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});$('calibration-status').textContent='Saved; re-run projection and tracking.';}catch(e){$('calibration-status').textContent=e.message;}};
$('action-select').onchange=loadActionForm;
$('action-save').onclick=async()=>{const action=state.shotActions[Number($('action-select').value)||0];if(!action)return;$('action-status').textContent='Saving…';try{const body={formation_start_s:Number($('action-formation').value),snap_s:Number($('action-snap').value),dead_s:Number($('action-dead').value),playback_end_s:Number($('action-end').value)};const result=await json(`/api/actions/${encodeURIComponent(action.action_id)}/timing`,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(body)});Object.assign(action,result);$('action-status').textContent='Verified';$('action-select').options[$('action-select').selectedIndex].text=`Action ${action.action_order} · verified`;}catch(e){$('action-status').textContent=e.message;}};
$('games').onchange=loadWorkspace;$('workspace-mode').onchange=loadWorkspace;
$('timeline').oninput=e=>{state.frame=Number(e.target.value);if(state.shot&&state.frames[state.frame]!=null)$('film').currentTime=state.frames[state.frame];draw();};
$('play-animation').onclick=()=>{if(state.timer){clearInterval(state.timer);state.timer=null;return;}state.timer=setInterval(()=>{state.frame++;if(state.frame>=state.frames.length){clearInterval(state.timer);state.timer=null;return;}$('timeline').value=state.frame;draw();},100);};
field();loadGames().catch(e=>$('status').textContent=e.message);
