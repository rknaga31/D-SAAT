# 🚗 D-SAAT

> **Driver Somnolence, Alertness & Autonomic Telemetry**

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%201.28+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20MediaPipe-green.svg?logo=opencv&logoColor=white)](https://mediapipe.dev)
[![Tests](https://img.shields.io/badge/Tests-49%2F49%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Multimodal%204--Stream%20Fusion-purple.svg)]()

**A real-time, software-first multimodal driver drowsiness, fatigue, and physiological monitoring system.**  
*Camera + Microphone + Wearable PPG Telemetry · Contactless rPPG · Adaptive Sensor Fusion · Obsidian Cockpit HUD*

[Features](#-key-capabilities) •
[Architecture](#-system-architecture) •
[Quick Start](#-quick-start) •
[Dashboard](#-obsidian-telemetry-cockpit) •
[Algorithms](#-mathematical--algorithmic-foundations) •
[Configuration](#-configuration) •
[Testing](#-test-suite--verification)

</div>

---

## 📌 Overview

Driver fatigue and drowsiness remain leading contributors to worldwide vehicular collisions. **D-SAAT** (Driver Somnolence, Alertness & Autonomic Telemetry) provides a modular, low-latency, multimodal telemetric pipeline that continuously tracks driver alertness, cognitive workload, and physiological strain in real time.

By coupling **contactless computer vision** (MediaPipe 468-point facial mesh + CHROM rPPG hemodynamics), **ambient acoustic signal processing** (breathing rate & yawning acoustic energy), and **wearable smartwatch telemetry** (PPG pulse, SpO2, stress, and HRV), the system achieves robust, fault-tolerant drowsiness detection that dramatically outperforms single-modality solutions.

---

## 🎯 Key Capabilities

| Modality | Source | Extracted Features | Physiological Significance |
| :--- | :--- | :--- | :--- |
| **Facial Dynamics** | Webcam (30 FPS) | **EAR** (Eye Aspect Ratio), **MAR** (Mouth Aspect Ratio), **PERCLOS** (60s rolling window), **Blink Frequency**, **3D Head Pose** (Pitch / Yaw / Roll) | Microsleep detection, persistent gaze distraction, head nodding, and physical yawning |
| **Contactless rPPG** | Webcam (Forehead / Cheek ROI) | **Heart Rate (BPM)** via CHROM chrominance analysis, **HRV-RMSSD** via inter-beat interval (IBI) analysis | Autonomic nervous system (ANS) shifts, bradycardia, and drowsiness-induced parasympathetic activation |
| **Acoustic Intelligence** | Microphone (22,050 Hz) | **Breathing Rate** (breaths/min), **Yawn Acoustic Signatures** (spectral energy in 200–2,000 Hz), **MFCCs** (speech vs. ambient silence) | Respiratory slowing, vocal fatigue, and deep yawning exhalations |
| **Smartwatch PPG** | Wearable API / Direct Stream | **Heart Rate (BPM)**, **SpO2 (%)**, **HRV (RMSSD)**, **Stress Index** | Ground-truth physiological cross-validation and motion-artifact compensation |

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                    HARDWARE CAPTURE LAYER                   │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
                     ┌────────────────────────────────────────┼────────────────────────────────────────┐
                     ▼                                        ▼                                        ▼
             Webcam (30 FPS)                         Microphone (22 kHz)                     Smartwatch (Wearable API)
                     │                                        │                                        │
                     ▼                                        ▼                                        ▼
             VideoCaptureBuffer                       AudioCaptureBuffer                     PPG Telemetry Poller
                     │                                        │                                        │
     ┌───────────────┴───────────────┐                        │                                        │
     ▼                               ▼                        ▼                                        ▼
VisualFeatureExtractor        RPPGExtractor         AudioFeatureExtractor                   SmartwatchFeatureExtractor
(MediaPipe FaceMesh)          (CHROM Method)        (Spectral Energy/MFCC)                  (PPG HR, SpO2, HRV, Stress)
  • EAR & MAR Formula           • Green/Red Chrom       • Respiration Rate                    • Wearable HR & SpO2
  • PERCLOS (60s rolling)       • Bandpass (0.75-3Hz)   • Yawn Acoustic Energy                • Autonomic Stress Index
  • 3D Head Pose (P/Y/R)        • FFT Peak → BPM        • Voice Activity Detection            • Connection Watchdog
     │                               │                        │                                        │
     └───────────────────────┬───────┴────────────────────────┼────────────────────────────────────────┘
                             │                                │
                             ▼                                ▼
              ┌─────────────────────────────────────────────────────────────┐
              │             CROSS-MODAL VALIDATION ENGINE                   │
              │  • Facial rPPG vs. Smartwatch PPG HR Divergence Check       │
              │  • Dynamic Confidence Re-weighting & Anomaly Flagging       │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │           ADAPTIVE MULTIMODAL FUSION & RISK SCORER          │
              │  • Dynamic Weight Normalization: w_vis, w_rppg, w_aud, w_sw │
              │  • Exponential Moving Average (EMA, α=0.30) Smoothing      │
              │  • 4-Tier Risk Assessment: SAFE / WARNING / DANGER / CRIT   │
              └──────────────────────────────┬──────────────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
      ┌─────────────────────────────┐                 ┌─────────────────────────────┐
      │        ALERT MANAGER        │                 │    SHARED TELEMETRY STATE   │
      │  • Multi-tone Audio Beeps   │                 │    (Thread-Safe Ring Store) │
      │  • Synthesized Speech (TTS) │                 └──────────────┬──────────────┘
      │  • Alert Throttling & Cooldown│                              │
      └─────────────────────────────┘                                ▼
                                                      ┌─────────────────────────────┐
                                                      │  OBSIDIAN TELEMETRY COCKPIT │
                                                      │  (Streamlit Real-Time HUD)  │
                                                      │  • Camera HUD & Mesh Overlay│
                                                      │  • 4-Axis Modality Radar    │
                                                      │  • Physiological Splines    │
                                                      │  • Hardware-Free Demo Mode  │
                                                      └─────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites
- **Python 3.10 to 3.13**
- Standard USB Webcam or integrated laptop camera
- Microphone (built-in or external)

### Step 1: Clone the Repository
```bash
git clone https://github.com/rknaga31/D-SAAT.git
cd D-SAAT
```

### Step 2: Set Up Virtual Environment (Recommended)
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows Audio Tip**: If `sounddevice` or `PortAudio` reports a missing DLL, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) or run:
> ```bash
> pip install sounddevice --find-links https://github.com/spatialaudio/python-sounddevice/releases
> ```

> **MediaPipe GPU Warning**: On some headless systems or systems without dedicated OpenGL drivers, set:
> ```bash
> $env:MEDIAPIPE_DISABLE_GPU="1"   # PowerShell
> export MEDIAPIPE_DISABLE_GPU=1    # Bash
> ```

---

## 🚀 Quick Start

### Option A: Complete Live Pipeline + Telemetry HUD (Recommended)
Launch the background multimodal analysis engine in one terminal, and the Streamlit telemetry cockpit in another:

```bash
# Terminal 1 — Start the real-time processing engine
python main.py --headless

# Terminal 2 — Launch the Obsidian Telemetry Dashboard
streamlit run dashboard/app.py
```
Open **http://localhost:8501** in your browser to inspect real-time telemetric feeds.

---

### Option B: Standalone Interactive Demo Mode (No Hardware Needed)
Perfect for presentations, development, or evaluating the system without a webcam or microphone:

```bash
streamlit run dashboard/app.py
```
Toggle **"Demo Mode"** in the sidebar to simulate all driving fatigue scenarios (Alert, Mild Fatigue, Severe Microsleep, Critical Drowsiness) with dynamic synthetic biometric generation.

---

### Option C: Pipeline with Desktop OpenCV HUD
Run a standalone desktop window with bounding boxes, facial landmark meshes, and live metric readouts:

```bash
python main.py
# Press 'q' inside the OpenCV window to exit cleanly
```

---

## 🖥️ Obsidian Telemetry Cockpit

The web dashboard is engineered with a custom **Obsidian / Slate dark telemetry theme** ([`.streamlit/config.toml`](.streamlit/config.toml)) crafted for vehicular cockpits and telemetry rooms:

- **🎥 Live Video & HUD Overlay**: Displays real-time face tracking, eye contour landmarks, and forehead rPPG bounding box.
- **🛡️ 4-Tier Risk Banner**: Dynamic banner transitioning across **SAFE** (Emerald), **WARNING** (Amber), **DANGER** (Orange), and **CRITICAL** (Crimson).
- **🔺 4-Axis Modality Radar**: Real-time balance chart tracking relative risk scores across Visual, rPPG, Audio, and Smartwatch streams.
- **📊 Vital Statistics Panel**: Instantaneous metrics for EAR, MAR, PERCLOS, Blink Rate, Contactless rPPG HR, Smartwatch HR, SpO2, HRV, and Stress Index.
- **📈 60-Second Rolling Telemetry**: Smooth spline charts contrasting Camera rPPG HR against Smartwatch PPG HR and tracking multi-sensor risk progression.
- **🎛️ Dynamic Sensitivity Controls**: Sliders to adjust EAR threshold, alarm cooldowns, and sensor weights on the fly.

---

## 🔬 Mathematical & Algorithmic Foundations

### 1. Eye Aspect Ratio (EAR)
Calculated from 6 key 2D landmarks per eye:
$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
*When $\text{EAR} < 0.21$ consecutively for more than 450 ms, an eye closure event is recorded.*

### 2. PERCLOS (Percentage of Eye Closure)
Evaluated over a rolling 60-second temporal window:
$$\text{PERCLOS} = \frac{\sum_{t \in T} \mathbb{I}(\text{EAR}_t < \tau_{\text{EAR}})}{|T|} \times 100\%$$
- $\text{PERCLOS} \ge 15\% \implies \text{WARNING}$
- $\text{PERCLOS} \ge 30\% \implies \text{DANGER / CRITICAL}$

### 3. Contactless rPPG (CHROM Method)
Extracts blood volume pulse (BVP) from skin color variations:
1. Spatial averaging of RGB channels across forehead/cheek ROIs: $S = [R(t), G(t), B(t)]^T$
2. Zero-mean normalization: $R_n(t) = R(t)/\mu_R - 1$, $G_n(t) = G(t)/\mu_G - 1$, $B_n(t) = B(t)/\mu_B - 1$
3. Chrominance orthogonal signal projection:
   $$X_{\text{chrom}}(t) = 3 R_n(t) - 2 G_n(t)$$
   $$Y_{\text{chrom}}(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$
4. Filtered pulse signal:
   $$S_{\text{BVP}}(t) = X(t) - \alpha Y(t) \quad \text{where } \alpha = \frac{\sigma(X)}{\sigma(Y)}$$
5. Bandpass filtering ($0.75 - 3.0\text{ Hz} \equiv 45 - 180\text{ BPM}$), Fast Fourier Transform (FFT) peak discovery, and Inter-Beat Interval (IBI) calculation for RMSSD heart rate variability.

### 4. Cross-Modal Agreement & Adaptive Sensor Fusion
When both rPPG and Smartwatch PPG are active, the system evaluates their absolute divergence:
$$\Delta_{\text{HR}} = |\text{HR}_{\text{rPPG}} - \text{HR}_{\text{smartwatch}}|$$
If $\Delta_{\text{HR}} > 18\text{ BPM}$, confidence in the rPPG signal is downscaled due to potential illumination drift or head motion. The composite risk score is synthesized through normalized confidence weighting and Exponential Moving Average (EMA):
$$\text{Risk}_{\text{raw}} = \sum_{m \in M} w_m \cdot S_m, \quad \sum_{m \in M} w_m = 1.0$$
$$\text{Risk}_{\text{fused}}(t) = \alpha \cdot \text{Risk}_{\text{raw}}(t) + (1 - \alpha) \cdot \text{Risk}_{\text{fused}}(t-1) \quad (\alpha = 0.30)$$

---

## 🎛️ Configuration

All pipeline parameters, signal thresholds, and alarm stages can be customized in [`config.yaml`](config.yaml):

```yaml
visual:
  ear_threshold: 0.21            # Eye closure trigger threshold
  mar_threshold: 0.65            # Yawn detection threshold
  perclos_threshold_warning: 0.15 # 15% eye closure → Warning
  perclos_threshold_danger: 0.30  # 30% eye closure → Danger
  blink_rate_low: 8.0            # Saccadic suppression threshold (blinks/min)

rppg:
  window_sec: 10                 # Rolling buffer for spectral analysis
  normal_hr_min: 55              # Resting baseline low (BPM)
  normal_hr_max: 100             # Resting baseline high (BPM)

smartwatch:
  enabled: true                  # Enable wearable telemetry stream
  poll_interval_sec: 5.0         # Telemetry synchronization frequency
  spo2_warning_threshold: 94.0   # Hypoxia alert threshold (%)
  stress_threshold_high: 65.0    # Elevated sympathetic arousal index

fusion:
  weight_visual: 0.30            # Facial dynamics weight
  weight_rppg: 0.20              # Contactless rPPG weight
  weight_audio: 0.25             # Acoustic features weight
  weight_smartwatch: 0.25        # Wearable PPG weight
  alpha_ema: 0.30                # Temporal smoothing constant

alert:
  level_0_max: 0.25              # SAFE: 0.00 - 0.25
  level_1_max: 0.50              # WARNING: 0.26 - 0.50 (Auditory Chime)
  level_2_max: 0.75              # DANGER: 0.51 - 0.75 (Urgent Chime)
  # > 0.75                       # CRITICAL: (Continuous Audio Alarm + TTS Warning)
```

---

## 🧪 Test Suite & Verification

The project includes an exhaustive unit and integration test suite covering mathematical formulas, signal filters, thread-safe buffers, cross-modal synchronization, and alert handling:

```bash
# Run the complete test suite
pytest -v

# Run tests with detailed console output and timing
python -m pytest tests/ -v --durations=10
```

### Coverage Highlights:
- **`tests/test_pipeline.py`** (30 tests):
  - Thread-safe buffer enqueue/dequeue and overflow eviction
  - Mathematical integrity of EAR, MAR, and Head Pose matrices
  - Synthetic 72 BPM rPPG pulse verification via CHROM FFT
  - Audio feature extraction on pure tone and background noise
  - Multimodal fusion weight normalization and EMA dynamics
  - Alert manager cooldown timers and suppression logic
- **`tests/test_smartwatch.py`** (19 tests):
  - Smartwatch telemetry polling and mock data streams
  - Cross-modal HR divergence verification between camera and wearable
  - SpO2 desaturation and autonomic stress escalation rules
  - Missing modality fallback without pipeline interruption

```
============================== test session starts ===============================
platform win32 -- Python 3.13.0, pytest-9.1.1, pluggy-1.6.0
collected 49 items

tests/test_pipeline.py ..............................                      [ 61%]
tests/test_smartwatch.py ...................                               [100%]

======================= 49 passed, 9 warnings in 23.91s ========================
```

---

## 📁 Repository Structure

```
D-SAAT/
├── .gitignore                   # Clean ignore configuration
├── LICENSE                      # MIT Open-Source License
├── README.md                    # Comprehensive technical documentation
├── config.yaml                  # System-wide configuration & threshold tuning
├── conftest.py                  # Pytest fixtures and environment setup
├── main.py                      # Core multithreaded pipeline orchestrator
├── requirements.txt             # Locked dependencies & test requirements
│
├── .streamlit/
│   └── config.toml              # Obsidian / Slate cockpit theme styling
│
├── dashboard/
│   └── app.py                   # Streamlit live telemetry cockpit & demo mode
│
├── models/
│   └── face_landmarker.task     # MediaPipe FaceMesh model bundle (3.7 MB)
│
├── src/
│   ├── __init__.py
│   ├── alert/
│   │   ├── __init__.py
│   │   └── alert_manager.py     # Multi-level acoustic tones & TTS speech synthesis
│   │
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── audio_capture.py     # Threaded sounddevice microphone stream
│   │   └── video_capture.py     # Threaded OpenCV webcam stream & ring buffer
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── audio_features.py    # Respiration rate, yawn acoustics, and MFCCs
│   │   ├── rppg_features.py     # CHROM rPPG signal extraction, FFT & HRV
│   │   ├── smartwatch_features.py # Wearable PPG, SpO2, HRV & stress telemetry
│   │   └── visual_features.py   # EAR, MAR, PERCLOS, blink rate & 3D head pose
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── cross_validation.py  # Cross-modal agreement (rPPG vs. Smartwatch)
│   │   └── multimodal_fusion.py # Adaptive confidence-weighted EMA fusion
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── risk_scorer.py       # 4-tier risk classification engine
│   │
│   └── utils/
│       ├── __init__.py
│       ├── buffer.py            # Thread-safe ring buffer & shared telemetry state
│       └── logger.py            # Rotating file and console telemetry logger
│
└── tests/
    ├── __init__.py
    ├── test_pipeline.py         # 30 unit & integration tests for core pipeline
    └── test_smartwatch.py       # 19 unit tests for wearable & cross-modal validation
```

---

## 📚 References & Literature

1. **Facial Dynamics**: Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. Computer Vision Winter Workshop.
2. **Contactless rPPG**: de Haan, G., & Jeanne, V. (2013). *Robust Pulse Rate from Chrominance-Based rPPG*. IEEE Transactions on Biomedical Engineering, 60(10), 2878–2886.
3. **PERCLOS Metric**: Wierwille, W. W., et al. (1994). *Evaluation of Techniques for Detecting Driver Drowsiness*. National Highway Traffic Safety Administration (NHTSA).
4. **Heart Rate Variability**: Shaffer, F., & Ginsberg, J. P. (2017). *An Overview of Heart Rate Variability Metrics and Norms*. Frontiers in Public Health, 5, 258.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**R NAGA ABISHEIK**  
- GitHub: [@rknaga31](https://github.com/rknaga31)  
- Email: [nagaabisheik.r@gmail.com](mailto:nagaabisheik.r@gmail.com)

---

<div align="center">
  <sub>Built with ❤️ for automotive safety and intelligent cockpit systems.</sub>
</div>
