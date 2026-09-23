/**
 * D-SAAT — Real-Time Telemetry Web Cockpit Logic
 */

// ── State Management ────────────────────────────────────────────────────────
const state = {
  demoMode: false,
  audioMuted: false,
  lastAlertLevel: 0,
  historyMaxPoints: 60,
  browserCamActive: false,
  browserCamStream: null,
  browserCamInterval: null,
  isSendingFrame: false,
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
  if (state.audioMuted) return;
  initAudio();
  if (!audioCtx) return;

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
    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.start(now);
    osc.stop(now + 0.36);
  } else if (level === 2) {
    // Urgent double pulse
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(659.25, now);
    osc.frequency.setValueAtTime(880, now + 0.12);
    gain.gain.setValueAtTime(0.3, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
    osc.start(now);
    osc.stop(now + 0.41);
  } else if (level >= 3) {
    // Critical alert alarm
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(987.77, now);
    osc.frequency.setValueAtTime(493.88, now + 0.1);
    gain.gain.setValueAtTime(0.35, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.start(now);
    osc.stop(now + 0.36);
  }
}

function playTestChime() {
  initAudio();
  if (!audioCtx) return;
  const now = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.type = 'sine';
  osc.frequency.setValueAtTime(587.33, now);
  osc.frequency.exponentialRampToValueAtTime(880, now + 0.15);
  gain.gain.setValueAtTime(0.3, now);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
  osc.start(now);
  osc.stop(now + 0.46);
}

// ── DOM References ──────────────────────────────────────────────────────────
const dom = {
  modeBadge: document.getElementById('modeBadge'),
  btnBrowserCam: document.getElementById('btnBrowserCam'),
  camIcon: document.getElementById('camIcon'),
  camText: document.getElementById('camText'),
  clientVideo: document.getElementById('clientVideo'),
  clientCanvas: document.getElementById('clientCanvas'),
  videoFeed: document.getElementById('videoFeed'),
  btnDemoToggle: document.getElementById('btnDemoToggle'),
  btnDemoText: document.getElementById('btnDemoText'),
  btnAudioToggle: document.getElementById('btnAudioToggle'),
  btnTestAudio: document.getElementById('btnTestAudio'),
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
  gaugeSubtext: document.getElementById('gaugeSubtext'),
  valVisual: document.getElementById('valVisual'),
  barVisual: document.getElementById('barVisual'),
  valRppg: document.getElementById('valRppg'),
  barRppg: document.getElementById('barRppg'),
  valAudio: document.getElementById('valAudio'),
  barAudio: document.getElementById('barAudio'),
  valSmartwatch: document.getElementById('valSmartwatch'),
  barSmartwatch: document.getElementById('barSmartwatch'),
  camPromptOverlay: document.getElementById('camPromptOverlay'),
  btnOverlayStartCam: document.getElementById('btnOverlayStartCam'),
  faceTrackingBadge: document.getElementById('faceTrackingBadge'),
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
  hudEyeStatus: document.getElementById('hudEyeStatus'),
  hudMeshStatus: document.getElementById('hudMeshStatus'),
  radarCanvas: document.getElementById('radarChart'),
  telemetryCanvas: document.getElementById('telemetryChart')
};

// ── Setup Listeners ─────────────────────────────────────────────────────────
window.addEventListener('click', () => initAudio(), { once: true });

if (dom.btnOverlayStartCam) {
  dom.btnOverlayStartCam.addEventListener('click', () => {
    startBrowserWebcam();
  });
}

if (dom.btnTestAudio) {
  dom.btnTestAudio.addEventListener('click', () => {
    playTestChime();
  });
}

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
  if (state.browserCamActive) {
    stopBrowserWebcam();
  }
  try {
    const res = await fetch('/api/demo_mode', { method: 'POST' });
    const data = await res.json();
    state.demoMode = data.demo_mode;
    updateModeDisplay();
  } catch (err) {
    console.error('Failed to toggle demo mode:', err);
  }
});

if (dom.btnBrowserCam) {
  dom.btnBrowserCam.addEventListener('click', () => {
    if (state.browserCamActive) {
      stopBrowserWebcam();
    } else {
      startBrowserWebcam();
    }
  });
}

function updateModeDisplay() {
  if (state.browserCamActive) {
    dom.modeBadge.textContent = 'BROWSER WEBCAM (LIVE)';
    dom.modeBadge.className = 'capsule-badge capsule-live';
    if (dom.btnBrowserCam) dom.btnBrowserCam.classList.add('active');
    if (dom.camIcon) dom.camIcon.textContent = '📸';
    if (dom.camText) dom.camText.textContent = 'Browser Cam: ON';
    if (dom.camPromptOverlay) dom.camPromptOverlay.style.display = 'none';
    dom.btnDemoToggle.classList.remove('active');
  } else if (state.demoMode) {
    dom.modeBadge.textContent = 'DEMO SIMULATION';
    dom.modeBadge.className = 'capsule-badge capsule-demo';
    dom.btnDemoToggle.classList.add('active');
    if (dom.btnBrowserCam) dom.btnBrowserCam.classList.remove('active');
    if (dom.camIcon) dom.camIcon.textContent = '📷';
    if (dom.camText) dom.camText.textContent = 'Browser Cam: OFF';
    if (dom.camPromptOverlay) dom.camPromptOverlay.style.display = 'none';
  } else {
    dom.modeBadge.textContent = 'STANDBY · START CAMERA';
    dom.modeBadge.className = 'capsule-badge capsule-demo';
    dom.btnDemoToggle.classList.remove('active');
    if (dom.btnBrowserCam) dom.btnBrowserCam.classList.remove('active');
    if (dom.camIcon) dom.camIcon.textContent = '📷';
    if (dom.camText) dom.camText.textContent = 'Browser Cam: OFF';
    if (dom.camPromptOverlay) dom.camPromptOverlay.style.display = 'flex';
  }
}

// ── Browser Camera Streaming ────────────────────────────────────────────────
async function startBrowserWebcam() {
  initAudio();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Camera API requires HTTPS or localhost. Please ensure your browser URL starts with https://");
    return;
  }

  const constraints = {
    audio: false,
    video: {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 }
    }
  };

  try {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (constrainErr) {
      console.warn("Retrying with universal mobile constraints:", constrainErr);
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    }

    state.browserCamStream = stream;
    dom.clientVideo.srcObject = stream;
    dom.clientVideo.style.display = 'block';
    if (dom.videoFeed) dom.videoFeed.style.display = 'none';

    try {
      await dom.clientVideo.play();
    } catch (playErr) {
      console.warn("Video play warning:", playErr);
    }

    // Wait until video has real dimensions on mobile/desktop
    await new Promise((resolve) => {
      if (dom.clientVideo.videoWidth > 0) return resolve();
      dom.clientVideo.onloadedmetadata = () => resolve();
      dom.clientVideo.onloadeddata = () => resolve();
      setTimeout(resolve, 800);
    });

    state.browserCamActive = true;
    state.demoMode = false;
    updateModeDisplay();

    // Stream lightweight frames to server every 120ms (~8 FPS AI processing)
    state.browserCamInterval = setInterval(sendClientFrame, 120);
  } catch (err) {
    console.error("Camera access error:", err);
    alert("Could not access camera: " + (err.message || err.name) + "\n\nPlease ensure you tapped 'Allow' for camera permissions.");
    stopBrowserWebcam();
  }
}

function stopBrowserWebcam() {
  state.browserCamActive = false;
  if (state.browserCamInterval) {
    clearInterval(state.browserCamInterval);
    state.browserCamInterval = null;
  }
  if (state.browserCamStream) {
    state.browserCamStream.getTracks().forEach(t => t.stop());
    state.browserCamStream = null;
  }
  if (dom.clientVideo) {
    dom.clientVideo.srcObject = null;
    dom.clientVideo.style.display = 'none';
  }
  if (dom.videoFeed) {
    dom.videoFeed.style.display = 'block';
    dom.videoFeed.src = '/api/video_feed?t=' + Date.now();
  }
  updateModeDisplay();
}

async function sendClientFrame() {
  if (!state.browserCamActive || state.isSendingFrame) return;
  const video = dom.clientVideo;
  if (!video || video.videoWidth === 0) return;

  const canvas = dom.clientCanvas;
  const ctx = canvas.getContext('2d');
  
  // Mobile-safe lightweight frame (max 480 width for fast cellular upload)
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;
  const targetW = Math.min(480, srcW);
  const targetH = Math.round(targetW * (srcH / srcW));
  canvas.width = targetW;
  canvas.height = targetH;
  ctx.drawImage(video, 0, 0, targetW, targetH);

  const dataUrl = canvas.toDataURL('image/jpeg', 0.60);
  state.isSendingFrame = true;

  try {
    const res = await fetch('/api/process_client_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.telemetry) {
        renderSnapshot(data.telemetry);
      }
    }
  } catch (err) {
    console.warn("Client frame upload error:", err);
  } finally {
    state.isSendingFrame = false;
  }
}

// ── Telemetry Fetch & UI Render Loop ─────────────────────────────────────────
async function fetchTelemetry() {
  if (state.browserCamActive) return; // Updated directly from frame stream
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
  if (!state.browserCamActive) {
    state.demoMode = !!s.demo_mode;
  }
  updateModeDisplay();

  // Handle Standby state (prior to camera activation or demo mode)
  const isStandby = !state.browserCamActive && !state.demoMode && s.pipeline_running === false;
  if (isStandby) {
    if (dom.faceTrackingBadge) {
      dom.faceTrackingBadge.textContent = '○ Awaiting Cam';
      dom.faceTrackingBadge.style.color = 'var(--text-muted)';
    }
    if (dom.hudMeshStatus) dom.hudMeshStatus.textContent = 'STANDBY';
    if (dom.hudEyeStatus) {
      dom.hudEyeStatus.textContent = 'EYES: --';
      dom.hudEyeStatus.style.color = 'var(--text-muted)';
    }
    dom.sessionTime.textContent = 'SESSION 00:00 · STANDBY';
    dom.hudFps.textContent = '-- FPS';
    dom.hudPose.textContent = 'HEAD P:0° Y:0° R:0°';

    dom.alertBanner.className = 'alert-banner safe';
    dom.alertTag.textContent = 'STANDBY · READY';
    dom.alertMessage.textContent = 'Awaiting driver camera activation or demo simulation...';
    dom.bannerScore.textContent = '--%';
    dom.bannerScore.style.color = 'var(--text-muted)';

    const circumference = 314.15;
    dom.gaugeFill.style.strokeDashoffset = circumference;
    dom.gaugeFill.style.stroke = 'var(--text-muted)';
    dom.gaugeScore.textContent = '--%';
    dom.gaugeScore.style.color = 'var(--text-muted)';
    dom.gaugeStatus.textContent = 'STANDBY';
    dom.gaugeStatus.style.color = 'var(--text-muted)';
    if (dom.gaugeSubtext) {
      dom.gaugeSubtext.textContent = 'Enable camera to start driver safety index';
      dom.gaugeSubtext.style.color = 'var(--text-muted)';
    }

    // Reset modality bars
    dom.valVisual.textContent = '0.0%';
    dom.barVisual.style.width = '0%';
    dom.valRppg.textContent = '0.0%';
    dom.barRppg.style.width = '0%';
    dom.valAudio.textContent = '0.0%';
    dom.barAudio.style.width = '0%';
    dom.valSmartwatch.textContent = '0.0%';
    dom.barSmartwatch.style.width = '0%';

    dom.metricEar.textContent = '--';
    dom.metricPerclos.innerHTML = `-- <span class="metric-unit">%</span>`;
    dom.metricBlink.innerHTML = `-- <span class="metric-unit">/min</span>`;
    dom.metricMar.textContent = '--';
    dom.metricRppgHr.innerHTML = `-- <span class="metric-unit">BPM</span>`;
    dom.metricRppgHrv.innerHTML = `-- <span class="metric-unit">ms</span>`;
    dom.metricResp.innerHTML = `-- <span class="metric-unit">br/min</span>`;
    dom.metricCrossVal.textContent = 'STANDBY';
    dom.metricCrossVal.style.color = 'var(--text-muted)';
    dom.metricWatchHr.innerHTML = `-- <span class="metric-unit">BPM</span>`;
    dom.metricWatchSpo2.innerHTML = `-- <span class="metric-unit">%</span>`;
    dom.metricWatchStress.innerHTML = `-- <span class="metric-unit">/100</span>`;
    dom.metricWatchHrv.innerHTML = `-- <span class="metric-unit">ms</span>`;
    return;
  }

  const faceDetected = Boolean(s.face_detected);
  const faceLost = s.face_lost === true || s.face_lost === 1;

  if (dom.faceTrackingBadge) {
    if (!faceDetected && state.browserCamActive) {
      dom.faceTrackingBadge.textContent = '⚠️ Face Not Detected';
      dom.faceTrackingBadge.style.color = 'var(--status-warn)';
      if (dom.hudMeshStatus) dom.hudMeshStatus.textContent = 'NO FACE';
    } else {
      dom.faceTrackingBadge.textContent = '● Tracking Active';
      dom.faceTrackingBadge.style.color = 'var(--status-safe)';
      if (dom.hudMeshStatus) dom.hudMeshStatus.textContent = 'MESH 468';
    }
  }

  // 1. Session Duration & FPS
  const durSec = Math.floor(s.session_duration || 0);
  const mins = String(Math.floor(durSec / 60)).padStart(2, '0');
  const secs = String(durSec % 60).padStart(2, '0');
  const fps = (s.actual_fps || 30.0).toFixed(1);
  dom.sessionTime.textContent = `SESSION ${mins}:${secs} · ${fps} FPS`;
  dom.hudFps.textContent = `${fps} FPS`;

  // 2. Risk Level, Safety Index & Alert Banner
  let level = s.alert_level || 0;
  let label = s.alert_label || 'SAFE';
  let riskScore = s.smoothed_score || 0.0;

  // Immediate Feature Checks (AI Blendshapes + EAR + MAR + Head Pose)
  const earVal = s.ear || 0.0;
  const blinkScore = s.blink_score || Math.max(s.eye_blink_left || 0, s.eye_blink_right || 0);
  const isEyeClosed = !!(s.eye_closed || (earVal > 0.0 && earVal < 0.255) || blinkScore >= 0.38);
  const marVal = s.mar || 0.0;
  const isYawning = !!(s.yawn_detected_visual || marVal >= 0.58);
  const isDistracted = !!(s.head_pose_alert || Math.abs(s.yaw || 0) > 28 || (s.pitch || 0) < -22);

  if (dom.hudEyeStatus) {
    if (!faceDetected && state.browserCamActive) {
      dom.hudEyeStatus.textContent = 'EYES: NO FACE';
      dom.hudEyeStatus.style.color = '#f59e0b';
    } else if (isEyeClosed) {
      dom.hudEyeStatus.textContent = `EYES: CLOSED (${earVal > 0 ? earVal.toFixed(2) : 'SHUT'})`;
      dom.hudEyeStatus.style.color = '#ef4444';
    } else if (isYawning) {
      dom.hudEyeStatus.textContent = `MOUTH: YAWN (${marVal.toFixed(2)})`;
      dom.hudEyeStatus.style.color = '#f97316';
    } else {
      dom.hudEyeStatus.textContent = `EYES: OPEN (${earVal.toFixed(2)})`;
      dom.hudEyeStatus.style.color = 'var(--accent-cyan)';
    }
  }

  const wrapper = document.querySelector('.video-hud-wrapper');
  if (wrapper) {
    if (isEyeClosed && faceDetected) {
      wrapper.style.borderColor = '#ef4444';
      wrapper.style.boxShadow = '0 0 28px rgba(239, 68, 68, 0.75)';
    } else if (!faceDetected && state.browserCamActive) {
      wrapper.style.borderColor = '#f59e0b';
      wrapper.style.boxShadow = '0 0 20px rgba(245, 158, 11, 0.5)';
    } else if (isYawning) {
      wrapper.style.borderColor = '#f97316';
      wrapper.style.boxShadow = '0 0 20px rgba(249, 115, 22, 0.5)';
    } else {
      wrapper.style.borderColor = 'rgba(56, 189, 248, 0.2)';
      wrapper.style.boxShadow = '0 0 20px rgba(0, 0, 0, 0.6)';
    }
  }

  // Elevate risk based on detected events
  if (!faceDetected && state.browserCamActive) {
    if (faceLost) {
      level = Math.max(level, 2);
      label = 'DANGER';
      riskScore = Math.max(riskScore, 0.65);
    } else {
      level = Math.max(level, 1);
      label = 'ATTENTION LOST';
      riskScore = Math.max(riskScore, 0.45);
    }
  } else if (isEyeClosed && faceDetected) {
    level = Math.max(level, 3);
    label = 'CRITICAL';
    riskScore = Math.max(riskScore, 0.85);
  } else if (isYawning && faceDetected) {
    level = Math.max(level, 1);
    label = level >= 2 ? 'DANGER' : 'WARNING';
    riskScore = Math.max(riskScore, 0.40);
  } else if (isDistracted && faceDetected) {
    level = Math.max(level, 1);
    label = label === 'SAFE' ? 'DISTRACTED' : label;
    riskScore = Math.max(riskScore, 0.35);
  }

  let riskPct = Math.round(riskScore * 100);
  let safetyPct = Math.max(0, Math.min(100, 100 - riskPct));

  // Determine alert color based on safety / risk:
  const levelClasses = ['safe', 'warning', 'danger', 'critical'];
  const colors = ['#10b981', '#f59e0b', '#f97316', '#ef4444'];
  const curColor = colors[Math.min(level, 3)];
  const circumference = 314.15;

  if (!faceDetected && state.browserCamActive) {
    dom.alertBanner.className = 'alert-banner warning';
    dom.alertTag.textContent = faceLost ? 'DRIVER ATTENTION LOST' : 'AWAITING DRIVER FACE';
    dom.alertMessage.textContent = 'Driver face not detected in camera view! Maintain visual attention on the road.';
    dom.bannerScore.textContent = 'NO FACE';
    dom.bannerScore.style.color = '#f59e0b';

    dom.gaugeFill.style.strokeDashoffset = circumference * 0.4;
    dom.gaugeFill.style.stroke = '#f59e0b';
    dom.gaugeScore.textContent = 'NO FACE';
    dom.gaugeScore.style.color = '#f59e0b';
    dom.gaugeStatus.textContent = 'ATTENTION LOST';
    dom.gaugeStatus.style.color = '#f59e0b';
    if (dom.gaugeSubtext) {
      dom.gaugeSubtext.textContent = 'Driver face not in camera view';
      dom.gaugeSubtext.style.color = '#f59e0b';
    }
  } else {
    dom.alertBanner.className = `alert-banner ${levelClasses[Math.min(level, 3)]}`;
    dom.alertTag.textContent = `LEVEL ${level} · ${label}`;
    let msg = s.alert_message;
    if (isEyeClosed) {
      msg = '⚠️ MICROSLEEP ALERT: Driver eyes closed!';
    } else if (isYawning) {
      msg = '⚠️ YAWNING DETECTED: Signs of driver fatigue observed';
    } else if (isDistracted) {
      msg = '⚠️ DISTRACTION: Driver looking away from road';
    } else if (level === 0) {
      msg = 'Nominal driver alertness maintained. All systems nominal.';
    }
    dom.alertMessage.textContent = msg;
    dom.bannerScore.textContent = `${safetyPct}% SAFE`;
    dom.bannerScore.style.color = curColor;

    // 3. Circular Gauge — Driver Safety Score
    const safetyFraction = safetyPct / 100.0;
    const offset = circumference * (1.0 - safetyFraction);
    dom.gaugeFill.style.strokeDashoffset = offset;
    dom.gaugeFill.style.stroke = curColor;
    dom.gaugeScore.textContent = `${safetyPct}%`;
    dom.gaugeScore.style.color = curColor;

    let statusText = label;
    if (isEyeClosed) statusText = 'MICROSLEEP';
    else if (isYawning) statusText = 'YAWN / DROWSY';
    else if (isDistracted) statusText = 'DISTRACTED';
    else if (level === 0) statusText = 'VIGILANT';

    dom.gaugeStatus.textContent = statusText;
    dom.gaugeStatus.style.color = curColor;
    if (dom.gaugeSubtext) {
      dom.gaugeSubtext.textContent = `Safety Index · Fatigue Risk: ${riskPct}%`;
      dom.gaugeSubtext.style.color = curColor;
    }
  }

  // Trigger auditory alert on transition or when eyes closed
  if ((level > 0 || isEyeClosed) && (level !== state.lastAlertLevel || Math.random() < 0.25)) {
    playChime(Math.max(level, isEyeClosed ? 3 : 1));
  }
  state.lastAlertLevel = level;

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
  dom.hudPose.textContent = `HEAD P:${p}° Y:${y}° R:${r}°`;

  // 6. Update Rolling History & Charts
  updateHistory(rppgHr, watchHr, riskScore * 100);
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
  const cy = h / 2;
  const maxR = 48;

  ctx.clearRect(0, 0, w, h);

  const axes = [
    { label: 'Visual', angle: -Math.PI / 2, val: Math.min(1, Math.max(0, vis)), align: 'center', baseline: 'bottom', offsetR: 6 },
    { label: 'rPPG', angle: 0, val: Math.min(1, Math.max(0, rppg)), align: 'left', baseline: 'middle', offsetR: 8 },
    { label: 'Audio', angle: Math.PI / 2, val: Math.min(1, Math.max(0, audio)), align: 'center', baseline: 'top', offsetR: 6 },
    { label: 'Watch', angle: Math.PI, val: Math.min(1, Math.max(0, watch)), align: 'right', baseline: 'middle', offsetR: 8 }
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
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px Inter, sans-serif';
  axes.forEach(axis => {
    const x = cx + Math.cos(axis.angle) * maxR;
    const y = cy + Math.sin(axis.angle) * maxR;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();

    const lx = cx + Math.cos(axis.angle) * (maxR + axis.offsetR);
    const ly = cy + Math.sin(axis.angle) * (maxR + axis.offsetR);
    ctx.textAlign = axis.align;
    ctx.textBaseline = axis.baseline;
    ctx.fillText(axis.label, lx, ly);
  });

  // Modality values polygon
  ctx.beginPath();
  axes.forEach((axis, i) => {
    const r = Math.max(8, axis.val * maxR);
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
    const r = Math.max(8, axis.val * maxR);
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
  if (rect.width > 0 && rect.height > 0 && (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height))) {
    canvas.width = Math.round(rect.width);
    canvas.height = Math.round(rect.height);
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
