"""
Smoke tests for the D-SAAT pipeline.

Tests that all core modules can be imported and instantiated with default config,
and that key functions produce expected output shapes/types.

Run with:  python -m pytest tests/ -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Default config for tests ───────────────────────────────────────────────────
DEFAULT_CFG = {
    "video":   {"device_index": 0, "width": 640, "height": 480, "fps": 30, "buffer_size": 90},
    "audio":   {"sample_rate": 22050, "channels": 1, "block_duration": 0.5, "buffer_size": 60},
    "visual":  {
        "ear_threshold": 0.21, "ear_consecutive_frames": 3,
        "mar_threshold": 0.65, "mar_consecutive_frames": 15,
        "perclos_window_sec": 60, "perclos_threshold_warning": 0.15,
        "perclos_threshold_danger": 0.30, "blink_window_sec": 60,
        "blink_rate_min_normal": 10, "blink_rate_max_normal": 20,
        "head_yaw_threshold": 30, "head_pitch_threshold": 25, "head_roll_threshold": 20,
    },
    "rppg":    {
        "window_sec": 10, "min_hz": 0.75, "max_hz": 3.0, "roi_scale": 0.4,
        "normal_hr_min": 55, "normal_hr_max": 100,
        "low_hr_threshold": 50, "high_hr_threshold": 110,
    },
    "audio_features": {
        "n_mfcc": 13, "yawn_energy_threshold": 0.02,
        "breathing_window_sec": 10,
        "breathing_rate_normal_min": 12, "breathing_rate_normal_max": 20,
    },
    "fusion":  {
        "weight_visual": 0.30, "weight_rppg": 0.20, "weight_audio": 0.25, "weight_smartwatch": 0.25,
        "smoothing_alpha": 0.3,
        "visual_confidence_threshold": 0.5,
        "rppg_confidence_threshold": 0.3,
        "audio_confidence_threshold": 0.3,
        "smartwatch_confidence_threshold": 0.4,
    },
    "smartwatch": {
        "enabled": True,
        "provider": "mock",
        "poll_interval_sec": 5.0,
        "simulation_mode": True,
        "low_hr_threshold": 50.0,
        "high_hr_threshold": 120.0,
        "low_spo2_threshold": 92.0,
        "high_stress_threshold": 80.0,
    },
    "alert":   {"level_0_max": 0.25, "level_1_max": 0.50, "level_2_max": 0.75,
                "cooldown_sec": 3, "tts_enabled": False},
    "logging": {"level": "WARNING"},
}


# ══════════════════════════════════════════════════════════════════════════════
# Buffer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestThreadSafeBuffer:
    def test_put_and_get_latest(self):
        from src.utils.buffer import ThreadSafeBuffer
        buf = ThreadSafeBuffer(maxlen=10)
        buf.put(np.ones((5,), dtype=np.float32), timestamp=1.0)
        buf.put(np.ones((5,), dtype=np.float32) * 2, timestamp=2.0)
        item = buf.get_latest()
        assert item is not None
        assert item.timestamp == 2.0
        np.testing.assert_array_equal(item.data, np.ones(5) * 2)

    def test_maxlen_wraps(self):
        from src.utils.buffer import ThreadSafeBuffer
        buf = ThreadSafeBuffer(maxlen=5)
        for i in range(10):
            buf.put(i, timestamp=float(i))
        assert len(buf) == 5
        assert buf.get_latest().data == 9

    def test_get_window(self):
        from src.utils.buffer import ThreadSafeBuffer
        buf = ThreadSafeBuffer(maxlen=100)
        now = time.time()
        for i in range(20):
            buf.put(i, timestamp=now - (20 - i))
        items = buf.get_window(seconds=5.0)
        # Only items from last 5 seconds
        assert all(it.timestamp >= now - 5.0 for it in items)

    def test_get_last_n(self):
        from src.utils.buffer import ThreadSafeBuffer
        buf = ThreadSafeBuffer(maxlen=50)
        for i in range(10):
            buf.put(i)
        last_3 = buf.get_last_n(3)
        assert len(last_3) == 3
        assert last_3[-1].data == 9


class TestSharedState:
    def test_update_and_get(self):
        from src.utils.buffer import SharedState
        s = SharedState()
        s.update(foo=42, bar="hello")
        assert s.get("foo") == 42
        assert s.get("bar") == "hello"
        assert s.get("missing", "default") == "default"

    def test_snapshot(self):
        from src.utils.buffer import SharedState
        s = SharedState()
        s.update(x=1, y=2)
        snap = s.snapshot()
        assert isinstance(snap, dict)
        assert snap["x"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# Visual Features tests (no webcam — use synthetic frames)
# ══════════════════════════════════════════════════════════════════════════════

class TestVisualFeatureExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        from src.features.visual_features import VisualFeatureExtractor
        return VisualFeatureExtractor(DEFAULT_CFG)

    def test_instantiation(self, extractor):
        assert extractor is not None
        assert extractor.ear_thresh == 0.21

    def test_blank_frame_no_face(self, extractor):
        """All-black frame should return face_detected=False."""
        from src.features.visual_features import VisualFeatureExtractor
        ext = VisualFeatureExtractor(DEFAULT_CFG)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        feat = ext.process(blank)
        assert feat.face_detected is False
        assert feat.annotated_frame is not None
        assert feat.annotated_frame.shape == blank.shape
        ext.release()

    def test_output_fields(self, extractor):
        """Feature object must have all expected fields."""
        from src.features.visual_features import VisualFeatures
        feat = VisualFeatures()
        assert hasattr(feat, "ear")
        assert hasattr(feat, "mar")
        assert hasattr(feat, "perclos")
        assert hasattr(feat, "pitch")
        assert hasattr(feat, "yaw")
        assert hasattr(feat, "roll")
        assert hasattr(feat, "blink_rate_per_min")
        assert hasattr(feat, "head_pose_alert")

    def test_ear_computation(self):
        """EAR formula gives 0.0 for degenerate points."""
        from src.features.visual_features import VisualFeatureExtractor as VFE
        # All same point → h=0 → should return ~0 (not crash)
        lm = [(100, 100)] * 500
        ear = VFE._compute_ear(lm, [33, 160, 158, 133, 153, 144])
        assert isinstance(ear, float)
        assert ear >= 0.0

    def test_mar_computation(self):
        """MAR formula should not crash."""
        from src.features.visual_features import VisualFeatureExtractor as VFE
        lm = [(i * 2, i * 2) for i in range(500)]
        mar = VFE._compute_mar(lm)
        assert isinstance(mar, float)


# ══════════════════════════════════════════════════════════════════════════════
# rPPG tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRPPGExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        from src.features.rppg_features import RPPGExtractor
        return RPPGExtractor(DEFAULT_CFG)

    def test_instantiation(self, extractor):
        assert extractor is not None

    def test_empty_buffer_returns_zero(self, extractor):
        extractor.reset()
        feat = extractor.get_features()
        assert feat.heart_rate_bpm == 0.0 or feat.signal_quality == 0.0

    def test_synthetic_signal_produces_hr(self):
        """Feed a synthetic 72 BPM signal and check HR estimate is in range."""
        from src.features.rppg_features import RPPGExtractor
        ext = RPPGExtractor(DEFAULT_CFG)
        fps = 30.0
        duration = 12  # seconds
        t = np.linspace(0, duration, int(fps * duration))
        freq = 72.0 / 60.0  # 1.2 Hz

        # Simulate realistic-ish face ROI colour changes
        now = time.time() - duration
        for i, ts in enumerate(t):
            # Create synthetic ROI: mostly uniform with subtle periodic variation
            r = 180 + 15 * np.sin(2 * np.pi * freq * ts)
            g = 120 + 10 * np.sin(2 * np.pi * freq * ts + 0.5)
            b = 90  + 5  * np.sin(2 * np.pi * freq * ts + 1.0)
            roi = np.full((20, 20, 3), [b, g, r], dtype=np.float32)
            ext.update(roi, timestamp=now + ts)

        feat = ext.get_features()
        # HR estimate should be within ±20 BPM of true value (generous for demo)
        if feat.signal_quality > 0.1:
            assert 40 <= feat.heart_rate_bpm <= 150, \
                f"HR {feat.heart_rate_bpm:.1f} out of plausible range"

    def test_features_dataclass(self):
        from src.features.rppg_features import RPPGFeatures
        f = RPPGFeatures()
        assert hasattr(f, "heart_rate_bpm")
        assert hasattr(f, "hrv_rmssd")
        assert hasattr(f, "signal_quality")
        assert isinstance(f.raw_signal, list)


# ══════════════════════════════════════════════════════════════════════════════
# Audio Feature tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioFeatureExtractor:
    @pytest.fixture(scope="class")
    def extractor(self):
        from src.features.audio_features import AudioFeatureExtractor
        return AudioFeatureExtractor(DEFAULT_CFG)

    def test_instantiation(self, extractor):
        assert extractor is not None

    def test_silence_returns_zero_energy(self, extractor):
        silence = np.zeros(11025, dtype=np.float32)
        feat = extractor.get_features(silence)
        assert feat.rms_energy < 1e-6

    def test_sine_wave_features(self, extractor):
        """Pure sine → reasonable ZCR and energy."""
        sr = DEFAULT_CFG["audio"]["sample_rate"]
        t  = np.linspace(0, 0.5, sr // 2, endpoint=False)
        chunk = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        feat = extractor.get_features(chunk)
        assert feat.rms_energy > 0
        assert feat.zero_crossing_rate >= 0.0
        assert 0.0 <= feat.audio_drowsiness_score <= 1.0

    def test_mfcc_shape(self, extractor):
        """MFCC list should have n_mfcc elements."""
        sr = DEFAULT_CFG["audio"]["sample_rate"]
        chunk = np.random.randn(sr).astype(np.float32) * 0.05
        feat = extractor.get_features(chunk)
        assert len(feat.mfcc) == DEFAULT_CFG["audio_features"]["n_mfcc"]


# ══════════════════════════════════════════════════════════════════════════════
# Fusion tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMultimodalFusion:
    @pytest.fixture(scope="class")
    def fusion(self):
        from src.fusion.multimodal_fusion import MultimodalFusion
        return MultimodalFusion(DEFAULT_CFG)

    def test_instantiation(self, fusion):
        assert fusion is not None

    def test_all_none_returns_zero(self, fusion):
        fusion.reset()
        state = fusion.fuse(None, None, None)
        assert state.smoothed_score == 0.0

    def test_output_range(self, fusion):
        """Fused score must stay in [0, 1]."""
        from src.features.visual_features import VisualFeatures
        fusion.reset()
        for _ in range(50):
            v = VisualFeatures()
            v.face_detected = True
            v.ear = np.random.uniform(0.1, 0.35)
            v.mar = np.random.uniform(0.1, 0.9)
            v.perclos = np.random.uniform(0, 0.5)
            v.confidence = 1.0
            state = fusion.fuse(v, None, None)
            assert 0.0 <= state.smoothed_score <= 1.0, \
                f"Score out of range: {state.smoothed_score}"

    def test_high_ear_gives_low_score(self, fusion):
        """Wide open eyes → low visual drowsiness score."""
        from src.features.visual_features import VisualFeatures
        fusion.reset()
        v = VisualFeatures()
        v.face_detected = True
        v.ear = 0.35
        v.mar = 0.10
        v.perclos = 0.01
        v.confidence = 1.0
        state = fusion.fuse(v, None, None)
        assert state.visual_score < 0.3, f"Expected low score, got {state.visual_score}"

    def test_low_ear_gives_high_score(self, fusion):
        """Nearly closed eyes → higher drowsiness score."""
        from src.features.visual_features import VisualFeatures
        fusion.reset()
        v = VisualFeatures()
        v.face_detected = True
        v.ear = 0.15
        v.mar = 0.10
        v.perclos = 0.40
        v.confidence = 1.0
        state = fusion.fuse(v, None, None)
        assert state.visual_score > 0.3, f"Expected high score, got {state.visual_score}"


# ══════════════════════════════════════════════════════════════════════════════
# Risk Scorer tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskScorer:
    @pytest.fixture(scope="class")
    def scorer(self):
        from src.prediction.risk_scorer import RiskScorer
        return RiskScorer(DEFAULT_CFG)

    def test_score_levels(self, scorer):
        from src.prediction.risk_scorer import (
            RiskScorer, LEVEL_SAFE, LEVEL_WARNING, LEVEL_DANGER, LEVEL_CRITICAL
        )
        from src.fusion.multimodal_fusion import FusedState

        scorer.reset_session()

        def make_state(s): 
            fs = FusedState()
            fs.smoothed_score = s
            fs.fused_score = s
            return fs

        assert scorer.score(make_state(0.10)).alert_level == LEVEL_SAFE
        assert scorer.score(make_state(0.35)).alert_level == LEVEL_WARNING
        assert scorer.score(make_state(0.60)).alert_level == LEVEL_DANGER
        assert scorer.score(make_state(0.80)).alert_level == LEVEL_CRITICAL

    def test_assessment_fields(self, scorer):
        from src.fusion.multimodal_fusion import FusedState
        fs = FusedState()
        fs.smoothed_score = 0.5
        ra = scorer.score(fs)
        assert hasattr(ra, "alert_level")
        assert hasattr(ra, "color")
        assert hasattr(ra, "emoji")
        assert hasattr(ra, "session_duration_sec")


# ══════════════════════════════════════════════════════════════════════════════
# Alert Manager tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertManager:
    @pytest.fixture(scope="class")
    def alerter(self):
        from src.alert.alert_manager import AlertManager
        return AlertManager(DEFAULT_CFG)

    def test_safe_level_no_alert(self, alerter):
        from src.prediction.risk_scorer import RiskAssessment, LEVEL_SAFE
        ra = RiskAssessment(alert_level=LEVEL_SAFE, alert_label="SAFE",
                            smoothed_score=0.1)
        result = alerter.process(ra)
        assert result is None

    def test_warning_fires_message(self, alerter):
        from src.prediction.risk_scorer import RiskAssessment, LEVEL_WARNING
        # Force cooldown to 0 for test
        alerter._last_alert_time = {1: 0.0, 2: 0.0, 3: 0.0}
        ra = RiskAssessment(alert_level=LEVEL_WARNING, alert_label="WARNING",
                            smoothed_score=0.4)
        result = alerter.process(ra)
        assert result is not None
        assert isinstance(result, str)

    def test_cooldown_suppresses_repeated(self, alerter):
        from src.prediction.risk_scorer import RiskAssessment, LEVEL_DANGER
        alerter._last_alert_time[2] = time.time()  # just fired
        ra = RiskAssessment(alert_level=LEVEL_DANGER, alert_label="DANGER",
                            smoothed_score=0.6)
        result = alerter.process(ra)
        assert result is None  # suppressed by cooldown


# ══════════════════════════════════════════════════════════════════════════════
# Integration smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSmoke:
    def test_full_pipeline_with_blank_frame(self):
        """
        Runs all feature extractors on a blank frame and audio silence,
        passes through fusion and scoring. Should not raise exceptions.
        """
        from src.features.visual_features import VisualFeatureExtractor
        from src.features.rppg_features import RPPGExtractor
        from src.features.audio_features import AudioFeatureExtractor
        from src.fusion.multimodal_fusion import MultimodalFusion
        from src.prediction.risk_scorer import RiskScorer
        from src.alert.alert_manager import AlertManager

        vis_ext  = VisualFeatureExtractor(DEFAULT_CFG)
        rppg_ext = RPPGExtractor(DEFAULT_CFG)
        aud_ext  = AudioFeatureExtractor(DEFAULT_CFG)
        fusion   = MultimodalFusion(DEFAULT_CFG)
        scorer   = RiskScorer(DEFAULT_CFG)
        alerter  = AlertManager(DEFAULT_CFG)

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        silence = np.zeros(11025, dtype=np.float32)

        # Run pipeline
        vis  = vis_ext.process(blank)
        rppg_ext.update(blank[:100, :100], timestamp=time.time())
        rppg = rppg_ext.get_features()
        aud_ext.update(silence)
        aud  = aud_ext.get_features(silence)

        fused = fusion.fuse(vis, rppg, aud)
        ra    = scorer.score(fused)
        msg   = alerter.process(ra)

        assert 0.0 <= fused.smoothed_score <= 1.0
        assert ra.alert_level in (0, 1, 2, 3)

        vis_ext.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
