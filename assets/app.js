/* Studio Trainer frontend — vanilla JS, no build step, fully offline.
   Settings persist as a KEYED payload (no positional list). Long ops are SSE
   streams. Presets are a frontend convenience that resolve to field values. */

// ---- Lucide line icons (MIT), inlined so nothing is fetched at runtime ----
const ICONS = {
  play: '<polygon points="6 3 20 12 6 21 6 3" fill="currentColor" stroke="none"/>',
  stop: '<circle cx="12" cy="12" r="10"/><rect width="6" height="6" x="9" y="9" rx="1"/>',
  folder: '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
  crop: '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
  tag: '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>',
  scissors: '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" x2="8.12" y1="4" y2="15.88"/><line x1="14.47" x2="20" y1="14.48" y2="20"/><line x1="8.12" x2="12" y1="8.12" y2="12"/>',
  refresh: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M3 21v-5h5"/>',
};
function svg(name) {
  return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ''}</svg>`;
}
document.querySelectorAll('[data-icon]').forEach(b => { b.innerHTML = svg(b.dataset.icon) + '<span>' + b.textContent.trim() + '</span>'; });

// ---- helpers ----
const $ = sel => document.querySelector(sel);
const byKey = k => document.querySelector(`[data-key="${k}"]`);
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
let savedAdamLr = '0.00005';
let running = false;

// ---- settings (keyed) ----
function collectSettings() {
  const s = {};
  document.querySelectorAll('[data-key]').forEach(el => {
    const k = el.dataset.key;
    if (el.type === 'checkbox') s[k] = el.checked;
    else if (el.dataset.type === 'number') s[k] = el.value === '' ? 0 : Number(el.value);
    else s[k] = el.value;
  });
  return s;
}
function populate(s) {
  document.querySelectorAll('[data-key]').forEach(el => {
    const k = el.dataset.key;
    if (!(k in s)) return;
    if (el.type === 'checkbox') el.checked = !!s[k];
    else el.value = s[k];
  });
}
let saveTimer = null;
function persist() { clearTimeout(saveTimer); saveTimer = setTimeout(persistNow, 350); }
function persistNow() {
  fetch('/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(collectSettings()) });
}

// ---- console ----
const con = $('#console');
let lastProgress = null;
function logLine(text, cls) {
  const div = document.createElement('div');
  if (cls) div.className = cls;
  div.textContent = text;
  con.appendChild(div);
  con.scrollTop = con.scrollHeight;
  return div;
}
function onEvent(ev) {
  if (ev.type === 'progress') {
    if (lastProgress) lastProgress.textContent = ev.line;
    else lastProgress = logLine(ev.line, 'prog');
    con.scrollTop = con.scrollHeight;
    return;
  }
  lastProgress = null;
  if (ev.type === 'log') logLine(ev.line, /warn/i.test(ev.line) ? 'warn' : '');
  else if (ev.type === 'preview') renderGallery(ev.images || []);
  else if (ev.type === 'done') { logLine('✓ ' + (ev.message || 'done'), 'ok'); setIdle(); }
  else if (ev.type === 'error') {
    logLine('✗ ' + (ev.message || 'error'), 'err');
    (ev.tail || []).forEach(l => logLine('  ' + l, 'err'));
    setIdle();
  }
}

// ---- gallery ----
function renderGallery(images) {
  const g = $('#gallery');
  if (!images.length) { g.innerHTML = '<div class="empty">No previews yet</div>'; return; }
  g.innerHTML = images.map(p => `<img src="/preview?path=${encodeURIComponent(p)}" loading="lazy" />`).join('');
}

// ---- SSE over fetch (POST) ----
async function streamSSE(url, body, onEv) {
  let res;
  try { res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) }); }
  catch (e) { onEv({ type: 'error', message: 'Network error: ' + e }); return; }
  if (res.status === 409) { onEv({ type: 'error', message: 'A training run is already active.' }); return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const line = chunk.split('\n').find(l => l.startsWith('data:'));
      if (line) { try { onEv(JSON.parse(line.slice(5).trim())); } catch (e) {} }
    }
  }
}

// ---- training lifecycle ----
function setBusy() { running = true; $('#start-btn').disabled = true; }
function setIdle() { running = false; $('#start-btn').disabled = false; }

$('#start-btn').onclick = () => { if (running) return; setBusy(); streamSSE('/train/start', collectSettings(), onEvent); };
$('#stop-btn').onclick = () => fetch('/train/stop', { method: 'POST' }).then(r => r.json()).then(d => logLine(d.message, 'warn'));
$('#folder-btn').onclick = () => fetch('/folder/output', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(collectSettings()) });
$('#open-ds-btn').onclick = () => fetch('/folder/dataset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(collectSettings()) }).then(r => r.json()).then(d => logLine(d.message));
$('#bucket-btn').onclick = () => streamSSE('/bucket', collectSettings(), onEvent);
$('#tag-btn').onclick = () => streamSSE('/tag', collectSettings(), onEvent);
$('#prune-btn').onclick = () => streamSSE('/prune', collectSettings(), onEvent);

// ---- updater ----
$('#update-check-btn').onclick = async () => {
  const r = await fetch('/update/check', { method: 'POST' }).then(r => r.json());
  $('#update-status').textContent = r.message || '';
  $('#update-apply-btn').style.display = (r.status === 'ok' && !r.up_to_date) ? '' : 'none';
};
$('#update-apply-btn').onclick = async () => {
  $('#update-status').textContent = 'Downloading…';
  const r = await fetch('/update/apply', { method: 'POST' }).then(r => r.json());
  $('#update-status').textContent = r.message || '';
};

// ---- optimizer <-> LR coupling (Trap 3) ----
function onOptimizerChange() {
  const opt = $('#optimizer').value, lr = $('#lr');
  if (opt === 'Prodigy') { if (lr.value !== '1.0') savedAdamLr = lr.value; lr.value = '1.0'; }
  else { if (lr.value === '1.0') lr.value = savedAdamLr || '0.00005'; }
  persistNow();
}
$('#optimizer').addEventListener('change', onOptimizerChange);

// ---- full fine-tune: rank inert ----
function syncFullFt() { byKey('network_rank').disabled = $('#full-ft').checked; }
$('#full-ft').addEventListener('change', () => { syncFullFt(); persistNow(); });

// ---- block-swap slider readout ----
function syncSwapOut() { $('#swap-out').textContent = $('#swap').value; }
$('#swap').addEventListener('input', syncSwapOut);

// ---- layered presets ----
function presetsActive() { return $('#lora-type').value !== 'Custom' || $('#vram-preset').value !== 'Custom'; }
async function resolvePresets() {
  const body = {
    lora_type: $('#lora-type').value,
    vram_tier: $('#vram-preset').value,
    dataset_path: byKey('dataset_path').value,
    train_batch_size: Number(byKey('train_batch_size').value || 1),
    gradient_accumulation_steps: Number(byKey('gradient_accumulation_steps').value || 1),
  };
  const r = await fetch('/presets/resolve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(r => r.json());
  applyUpdates(r.updates || {});
  fillPresetInfo(r);
}
function applyUpdates(u) {
  Object.entries(u).forEach(([k, v]) => {
    const el = byKey(k);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
  });
  if ('blocks_to_swap' in u) syncSwapOut();
  if (u.optimizer && u.optimizer !== 'Prodigy' && 'learning_rate' in u) savedAdamLr = String(u.learning_rate);
  persistNow();
}
function fillPresetInfo(r) {
  if (!presetsActive()) { $('#preset-info').innerHTML = ''; return; }
  const parts = [];
  (r.info || []).forEach(t => parts.push(`<div>${esc(t)}</div>`));
  (r.notes || []).forEach(t => parts.push(`<div class="note">${esc(t)}</div>`));
  (r.caveats || []).forEach(t => parts.push(`<div class="caveat">${esc(t)}</div>`));
  $('#preset-info').innerHTML = parts.join('');
}
$('#lora-type').addEventListener('change', resolvePresets);
$('#vram-preset').addEventListener('change', resolvePresets);

// ---- generic persistence + dataset-driven step recompute ----
document.querySelectorAll('[data-key]').forEach(el => {
  el.addEventListener('input', persist);
  el.addEventListener('change', persist);
});
byKey('dataset_path').addEventListener('change', () => { if ($('#lora-type').value !== 'Custom') resolvePresets(); });

// ---- init ----
(async function init() {
  try {
    const s = await fetch('/settings').then(r => r.json());
    populate(s);
    if (s.optimizer && s.optimizer !== 'Prodigy' && s.learning_rate !== '1.0') savedAdamLr = String(s.learning_rate);
  } catch (e) { logLine('Could not load settings: ' + e, 'err'); }
  syncSwapOut(); syncFullFt();
  // Preset dropdowns are NOT persisted — they always start at Custom.
  $('#lora-type').value = 'Custom'; $('#vram-preset').value = 'Custom';
  fetch('/update/check', { method: 'POST' }).then(r => r.json()).then(r => { $('#version').textContent = 'v' + (r.current || '—'); }).catch(() => {});
})();
