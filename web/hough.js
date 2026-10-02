'use strict';

const PARAMETERS = ['blur', 'canny_low', 'canny_high', 'threshold',
                    'min_line_length', 'max_line_gap', 'thickness'];
const DEFAULTS = {blur: 5, canny_low: 50, canny_high: 150, threshold: 45,
                  min_line_length: 60, max_line_gap: 35, thickness: 2};
const FRAME_STEP_S = 1 / 25;
const escapeHtml = (value) => String(value)
  .replaceAll('&', '&amp;')
  .replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;')
  .replaceAll("'", '&#039;');

const element = (id) => document.getElementById(id);
const status = element('status');
const frame = element('frame');
const spinner = element('spinner');
const time = element('time');

let clips = [];
let current = null;
let pending = 0;

function setStatus(message, isError) {
  status.textContent = message;
  status.classList.toggle('error', Boolean(isError));
}

function parameterValues() {
  const values = {};
  for (const name of PARAMETERS) values[name] = Number(element(name).value);
  values.show_edges = element('show_edges').checked;
  values.draw_lines = element('draw_lines').checked;
  return values;
}

function syncOutputs() {
  for (const output of document.querySelectorAll('output[data-for]')) {
    output.textContent = element(output.dataset.for).value;
  }
  element('time-value').textContent = `${Number(time.value).toFixed(2)}s`;
}

function query(values, extra) {
  const parameters = new URLSearchParams({timestamp_s: time.value});
  for (const name of PARAMETERS) {
    if (name === 'thickness' && !extra) continue;
    parameters.set(name, values[name]);
  }
  if (extra) {
    parameters.set('show_edges', values.show_edges);
    parameters.set('draw_lines', values.draw_lines);
  }
  return parameters.toString();
}

function renderStats(payload) {
  const stats = [
    ['Lines', payload.count],
    ['Longest', `${payload.longest} px`],
    ['Median', `${payload.median_length} px`],
    ['Frame', `${payload.width}×${payload.height}`],
    ['Time', `${payload.timestamp_s.toFixed(2)}s`],
  ];
  element('stats').innerHTML = stats
    .map(([label, value]) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`)
    .join('');
  renderHistogram(payload.segments);
}

function renderHistogram(segments) {
  const bins = new Array(36).fill(0);
  for (const segment of segments) {
    bins[Math.min(35, Math.floor(segment.angle / 5))] += 1;
  }
  const peak = Math.max(1, ...bins);
  element('histogram').innerHTML = bins
    .map((count, index) => `<div class="bar" title="${index * 5}–${index * 5 + 5}°: ${count}" ` +
      `style="height:${(count / peak) * 100}%;background:hsl(${index * 5 * 2},80%,60%)"></div>`)
    .join('');
}

async function refresh() {
  if (!current) return;
  syncOutputs();
  const values = parameterValues();
  const token = ++pending;
  spinner.hidden = false;
  const image = new Image();
  image.src = `/api/hough/clips/${encodeURIComponent(current.clip_id)}/annotated.jpg?` +
    query(values, true);
  try {
    const [, response] = await Promise.all([
      image.decode().catch(() => null),
      fetch(`/api/hough/clips/${encodeURIComponent(current.clip_id)}/lines?` + query(values, false)),
    ]);
    if (token !== pending) return;
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${response.status})`);
    }
    const payload = await response.json();
    frame.src = image.src;
    renderStats(payload);
    setStatus(`${payload.count} lines · ${current.clip_id} · ${current.angle}`);
  } catch (error) {
    if (token === pending) setStatus(error.message, true);
  } finally {
    if (token === pending) spinner.hidden = true;
  }
}

function debounce(callback, delay) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => callback(...args), delay);
  };
}

const refreshSoon = debounce(refresh, 140);

function selectClip(clipId) {
  current = clips.find((clip) => clip.clip_id === clipId) || null;
  if (!current) return;
  time.min = current.start_s;
  time.max = current.end_s;
  time.step = FRAME_STEP_S;
  time.value = current.start_s;
  refresh();
}

function populateClips(gameId) {
  const select = element('clips');
  const forGame = clips.filter((clip) => clip.game_id === gameId);
  select.innerHTML = forGame
    .map((clip) => `<option value="${escapeHtml(clip.clip_id)}">${escapeHtml(clip.clip_id)} · ` +
      `${escapeHtml(clip.angle)} · ${clip.start_s.toFixed(1)}–${clip.end_s.toFixed(1)}s</option>`)
    .join('');
  if (forGame.length) selectClip(forGame[0].clip_id);
  else setStatus('No shots with film are registered for this game.', true);
}

async function initialize() {
  syncOutputs();
  try {
    const response = await fetch('/api/hough/clips');
    if (!response.ok) throw new Error(`Could not list shots (${response.status})`);
    clips = await response.json();
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  if (!clips.length) {
    setStatus('No registered shots have film on this machine.', true);
    return;
  }
  const games = [...new Set(clips.map((clip) => clip.game_id))];
  element('games').innerHTML = games
    .map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join('');
  populateClips(games[0]);
}

element('games').addEventListener('change', (event) => populateClips(event.target.value));
element('clips').addEventListener('change', (event) => selectClip(event.target.value));
time.addEventListener('input', () => { syncOutputs(); refreshSoon(); });

for (const name of PARAMETERS) {
  element(name).addEventListener('input', () => { syncOutputs(); refreshSoon(); });
}
for (const name of ['show_edges', 'draw_lines']) {
  element(name).addEventListener('change', refresh);
}
for (const button of document.querySelectorAll('.stepper button')) {
  button.addEventListener('click', () => {
    const next = Number(time.value) + Number(button.dataset.step) * FRAME_STEP_S;
    time.value = Math.min(Number(time.max), Math.max(Number(time.min), next));
    syncOutputs();
    refresh();
  });
}
element('reset').addEventListener('click', () => {
  for (const [name, value] of Object.entries(DEFAULTS)) element(name).value = value;
  refresh();
});

initialize();
