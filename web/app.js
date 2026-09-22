/**
 * D-SAAT — Real-Time Telemetry Web Cockpit Logic
 */

// ── State Management ────────────────────────────────────────────────────────
const state = {
  demoMode: false,
  audioMuted: false,
  lastAlertLevel: 0,
  historyMaxPoints: 60,
  history: {
    times: [],
    rppgHr: [],
    watchHr: [],
    riskScore: []
  }
};

// ── Web Audio Synthesizer for In-Browser Alerts ─────────────────────────────
let audioCtx = null;

function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

function playChime(level) {
  if (state.audioMuted || !audioCtx) return;

  const now = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc.connect(gain);
  gain.connect(audioCtx.destination);

  if (level === 1) {
    // Gentle warning dual-tone
    osc.type = 'sine';
    osc.frequency.setValueAtTime(880, now);
    osc.frequency.exponentialRampToValueAtTime(1174.66, now + 0.15);
    gain.gain.setValueAtTime(0.12, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.start(now);
    osc.stop(now + 0.36);
  } else if (level === 2) {
    // Urgent double pulse
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(659.25, now);
    osc.frequency.setValueAtTime(880, now + 0.12);
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
    osc.start(now);
    osc.stop(now + 0.41);
  } else if (level >= 3) {
    // Critical alert beep
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(987.77, now);
    osc.frequency.setValueAtTime(493.88, now + 0.1);
    gain.gain.setValueAtTime(0.25, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
    osc.start(now);
    osc.stop(now + 0.31);
  }
}

// ── DOM References ──────────────────────────────────────────────────────────
const dom = {
  modeBadge: document.getElementById('modeBadge'),
  btnDemoToggle: document.getElementById('btnDemoToggle'),
  btnDemoText: document.getElementById('btnDemoText'),
  btnAudioToggle: document.getElementById('btnAudioToggle'),
  audioIcon: document.getElementById('audioIcon'),
  audioText: document.getElementById('audioText'),
  sessionTime: document.getElementById('sessionTime'),
  alertBanner: document.getElementById('alertBanner'),
  alertTag: document.getElementById('alertTag'),
  alertMessage: document.getElementById('alertMessage'),
  bannerScore: document.getElementById('bannerScore'),
  gaugeFill: document.getElementById('gaugeFill'),
  gaugeScore: document.getElementById('gaugeScore'),
  gaugeStatus: document.getElementById('gaugeStatus'),
  valVisual: document.getElementById('valVisual'),
  barVisual: document.getElementById('barVisual'),
  valRppg: document.getElementById('valRppg'),
  barRppg: document.getElementById('barRppg'),
  valAudio: document.getElementById('valAudio'),
  barAudio: document.getElementById('barAudio'),
  valSmartwatch: document.getElementById('valSmartwatch'),
  barSmartwatch: document.getElementById('barSmartwatch'),
  metricEar: document.getElementById('metricEar'),
  metricPerclos: document.getElementById('metricPerclos'),
  metricBlink: document.getElementById('metricBlink'),
  metricMar: document.getElementById('metricMar'),
  metricRppgHr: document.getElementById('metricRppgHr'),
  metricRppgHrv: document.getElementById('metricRppgHrv'),
  metricResp: document.getElementById('metricResp'),
  metricCrossVal: document.getElementById('metricCrossVal'),
  metricWatchHr: document.getElementById('metricWatchHr'),
  metricWatchSpo2: document.getElementById('metricWatchSpo2'),
  metricWatchStress: document.getElementById('metricWatchStress'),
  metricWatchHrv: document.getElementById('metricWatchHrv'),
  hudFps: document.getElementById('hudFps'),
  hudPose: document.getElementById('hudPose'),
  hudMeshStatus: document.getElementById('hudMeshStatus'),
  radarCanvas: document.getElementById('radarChart'),
  telemetryCanvas: document.getElementById('telemetryChart')
};

// ── Setup Audio & Demo Listeners ────────────────────────────────────────────
window.addEventListener('click', () => initAudio(), { once: true });

dom.btnAudioToggle.addEventListener('click', () => {
  initAudio();
  state.audioMuted = !state.audioMuted;
  if (state.audioMuted) {
    dom.audioIcon.textContent = '🔇';
    dom.audioText.textContent = 'Alert Sound: MUTED';
    dom.btnAudioToggle.classList.remove('active');
  } else {
    dom.audioIcon.textContent = '🔔';
    dom.audioText.textContent = 'Alert Sound: ON';
    dom.btnAudioToggle.classList.add('active');
  }
});

dom.btnDemoToggle.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/demo_mode', { method: 'POST' });
    const data = await res.json();
    state.demoMode = data.demo_mode;
    updateModeDisplay(state.demoMode);
  } catch (err) {
    console.error('Failed to toggle demo mode:', err);
  }
});

function updateModeDisplay(isDemo) {
  if (isDemo) {
    dom.modeBadge.textContent = 'DEMO SIMULATION';
    dom.modeBadge.className = 'capsule-badge capsule-demo';
    dom.btnDemoToggle.classList.add('active');
  } else {
    dom.modeBadge.textContent = 'LIVE FEED';
    dom.modeBadge.className = 'capsule-badge capsule-live';
    dom.btnDemoToggle.classList.remove('active');
  }
}

// ── Telemetry Fetch & UI Render Loop ─────────────────────────────────────────
async function fetchTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    if (!res.ok) return;
    const snap = await res.json();
    renderSnapshot(snap);
  } catch (err) {
    // Keep trying smoothly in the background
  }
}

function renderSnapshot(s) {
  state.demoMode = !!s.demo_mode;
  updateModeDisplay(state.demoMode);

  // 1. Session Duration & FPS
  const durSec = Math.floor(s.session_duration || 0);
  const mins = String(Math.floor(durSec / 60)).padStart(2, '0');
  const secs = String(durSec % 60).padStart(2, '0');
  const fps = (s.actual_fps || 30.0).toFixed(1);
  dom.sessionTime.textContent = `SESSION ${mins}:${secs} · ${fps} FPS`;
  dom.hudFps.textContent = `${fps} FPS`;

  // 2. Risk Level & Alert Banner
  const level = s.alert_level || 0;
  const label = s.alert_label || 'SAFE';
  const score = s.smoothed_score || 0.0;
  const scorePct = (score * 100).toFixed(1);
  dom.bannerScore.textContent = `${scorePct}%`;

  const levelClasses = ['safe', 'warning', 'danger', 'critical'];
  const colors = ['#10b981', '#f59e0b', '#f97316', '#ef4444'];
  const curColor = colors[Math.min(level, 3)];

  dom.alertBanner.className = `alert-banner ${levelClasses[Math.min(level, 3)]}`;
  dom.alertTag.textContent = `LEVEL ${level} · ${label}`;
  dom.alertMessage.textContent = s.alert_message || 'Nominal driver alertness maintained.';

  // Play auditory alert on transition or periodic danger/critical
  if (level > 0 && (level !== state.lastAlertLevel || Math.random() < 0.15)) {
    playChime(level);
  }
  state.lastAlertLevel = level;

  // 3. Circular Gauge
  const circumference = 314.15; // 2 * PI * 50
  const offset = circumference - (score * circumference);
  dom.gaugeFill.style.strokeDashoffset = offset;
  dom.gaugeFill.style.stroke = curColor;
  dom.gaugeScore.textContent = `${Math.round(score * 100)}%`;
  dom.gaugeScore.style.color = curColor;
  dom.gaugeStatus.textContent = label;

  // 4. Modality Breakdown Bars
  const vScore = s.visual_score || 0.0;
  const rScore = s.rppg_score || 0.0;
  const aScore = s.audio_score || 0.0;
  const wScore = s.smartwatch_score || 0.0;

  dom.valVisual.textContent = `${(vScore * 100).toFixed(1)}%`;
  dom.barVisual.style.width = `${Math.min(100, vScore * 100)}%`;

  dom.valRppg.textContent = `${(rScore * 100).toFixed(1)}%`;
  dom.barRppg.style.width = `${Math.min(100, rScore * 100)}%`;

  dom.valAudio.textContent = `${(aScore * 100).toFixed(1)}%`;
  dom.barAudio.style.width = `${Math.min(100, aScore * 100)}%`;

  dom.valSmartwatch.textContent = `${(wScore * 100).toFixed(1)}%`;
  dom.barSmartwatch.style.width = `${Math.min(100, wScore * 100)}%`;

  // 5. Physiological Metrics
  dom.metricEar.textContent = (s.ear || 0.0).toFixed(2);
  dom.metricPerclos.innerHTML = `${((s.perclos || 0.0) * 100).toFixed(1)} <span class="metric-unit">%</span>`;
  dom.metricBlink.innerHTML = `${Math.round(s.blink_rate || 0)} <span class="metric-unit">/min</span>`;
  dom.metricMar.textContent = (s.mar || 0.0).toFixed(2);

  const rppgHr = s.heart_rate || 72.0;
  dom.metricRppgHr.innerHTML = `${Math.round(rppgHr)} <span class="metric-unit">BPM</span>`;
  dom.metricRppgHrv.innerHTML = `${Math.round(s.hrv_rmssd || 35)} <span class="metric-unit">ms</span>`;
  dom.metricResp.innerHTML = `${Math.round(s.breathing_rate || 16)} <span class="metric-unit">br/min</span>`;

  const cvStatus = s.cross_val_status || 'AGREE';
  dom.metricCrossVal.textContent = cvStatus;
  dom.metricCrossVal.style.color = cvStatus === 'AGREE' ? 'var(--status-safe)' : 'var(--status-warn)';

  const watchHr = s.smartwatch_hr || rppgHr;
  dom.metricWatchHr.innerHTML = `${Math.round(watchHr)} <span class="metric-unit">BPM</span>`;
  dom.metricWatchSpo2.innerHTML = `${(s.smartwatch_spo2 || 98.5).toFixed(1)} <span class="metric-unit">%</span>`;
  dom.metricWatchStress.innerHTML = `${Math.round(s.smartwatch_stress || 24)} <span class="metric-unit">/100</span>`;
  dom.metricWatchHrv.innerHTML = `${Math.round(s.smartwatch_hrv || 40)} <span class="metric-unit">ms</span>`;

  // HUD Pose Readout
  const p = Math.round(s.pitch || 0);
  const y = Math.round(s.yaw || 0);
  const r = Math.round(s.roll || 0);
  dom.hudPose.textContent = `HEAD: P ${p}° / Y ${y}° / R ${r}°`;

  // 6. Update Rolling History & Charts
  updateHistory(rppgHr, watchHr, score * 100);
  drawRadar(vScore, rScore, aScore, wScore);
  drawTelemetryChart();
}

function updateHistory(rppgHr, watchHr, score) {
  const h = state.history;
  const now = new Date();
  const timeStr = `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

  h.times.push(timeStr);
  h.rppgHr.push(rppgHr);
  h.watchHr.push(watchHr);
  h.riskScore.push(score);

  if (h.times.length > state.historyMaxPoints) {
    h.times.shift();
    h.rppgHr.shift();
    h.watchHr.shift();
    h.riskScore.shift();
  }
}

// ── Radar Chart Rendering (Native Canvas) ───────────────────────────────────
function drawRadar(vis, rppg, audio, watch) {
  const canvas = dom.radarCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2 + 8;
  const maxR = 60;

  ctx.clearRect(0, 0, w, h);

  const axes = [
    { label: 'Visual', angle: -Math.PI / 2, val: Math.min(1, Math.max(0, vis)) },
    { label: 'rPPG', angle: 0, val: Math.min(1, Math.max(0, rppg)) },
    { label: 'Audio', angle: Math.PI / 2, val: Math.min(1, Math.max(0, audio)) },
    { label: 'Watch', angle: Math.PI, val: Math.min(1, Math.max(0, watch)) }
  ];

  // Concentric background rings
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  [0.33, 0.66, 1.0].forEach(factor => {
    ctx.beginPath();
    axes.forEach((axis, i) => {
      const x = cx + Math.cos(axis.angle) * maxR * factor;
      const y = cy + Math.sin(axis.angle) * maxR * factor;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  });

  // Axis spokes & labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'center';
  axes.forEach(axis => {
    const x = cx + Math.cos(axis.angle) * maxR;
    const y = cy + Math.sin(axis.angle) * maxR;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    const lx = cx + Math.cos(axis.angle) * (maxR + 14);
    const ly = cy + Math.sin(axis.angle) * (maxR + 14) + 3;
    ctx.fillText(axis.label, lx, ly);
  });

  // Modality values polygon
  ctx.beginPath();
  axes.forEach((axis, i) => {
    const r = Math.max(10, axis.val * maxR);
    const x = cx + Math.cos(axis.angle) * r;
    const y = cy + Math.sin(axis.angle) * r;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(56, 189, 248, 0.25)';
  ctx.fill();
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Highlight points
  axes.forEach(axis => {
    const r = Math.max(10, axis.val * maxR);
    const x = cx + Math.cos(axis.angle) * r;
    const y = cy + Math.sin(axis.angle) * r;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#f8fafc';
    ctx.fill();
  });
}

// ── 60-Second Rolling Telemetry Spline Chart ─────────────────────────────────
function drawTelemetryChart() {
  const canvas = dom.telemetryCanvas;
  if (!canvas) return;

  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width;
    canvas.height = rect.height;
  }

  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 12, right: 16, bottom: 20, left: 36 };

  ctx.clearRect(0, 0, w, h);

  const innerW = w - pad.left - pad.right;
  const innerH = h - pad.top - pad.bottom;
  const data = state.history;
  const len = data.times.length;
  if (len < 2) return;

  // Grid lines (BPM scale: 40 to 120)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#475569';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'right';

  [50, 75, 100].forEach(val => {
    const y = pad.top + innerH - ((val - 40) / 80) * innerH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
    ctx.fillText(`${val}`, pad.left - 6, y + 3);
  });

  function drawSeries(arr, color, minY, maxY, isDashed = false) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(isDashed ? [4, 4] : []);

    for (let i = 0; i < len; i++) {
      const x = pad.left + (i / (state.historyMaxPoints - 1)) * innerW;
      const normalized = Math.min(1, Math.max(0, (arr[i] - minY) / (maxY - minY)));
      const y = pad.top + innerH - normalized * innerH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw Series: rPPG HR (Pink), Watch HR (Purple), Risk % (Cyan, scaled 0 to 100)
  drawSeries(data.rppgHr, '#ec4899', 40, 120);
  drawSeries(data.watchHr, '#a855f7', 40, 120, true);
  drawSeries(data.riskScore, '#38bdf8', 0, 100);
}

// ── Start High-Frequency Telemetry Poller ────────────────────────────────────
setInterval(fetchTelemetry, 120);
fetchTelemetry();
