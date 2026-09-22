"""
Tests for Smartwatch Integration (4th Modality)

Validates:
  1. Smartwatch feature extraction, normalization to 1 Hz, and validation against rPPG.
  2. Cross-validation between rPPG and smartwatch HR (AGREE / DISAGREE / WATCH ONLY / CAMERA ONLY).
  3. 4-modality fusion with dynamic weight redistribution when smartwatch is disconnected.
  4. Rule-based physiological alert triggers (HR anomaly agreement, SpO2 < 92%, stress > 80 + visual fatigue).
  5. End-to-end integration and graceful degradation.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add project root so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.smartwatch_features import (
    SmartwatchFeatures,
    SmartwatchFeatureExtractor,
    fetch_health_data,
    normalize_to_1hz,
    validate_against_rppg,
)
from src.fusion.cross_validation import (
    cross_validate_heart_rate,
    CrossValidator,
    STATUS_AGREE,
    STATUS_DISAGREE,
    STATUS_MODERATE,
    STATUS_WATCH_ONLY,
    STATUS_CAMERA_ONLY,
    STATUS_NONE,
)
from src.fusion.multimodal_fusion import MultimodalFusion, FusedState
from src.prediction.risk_scorer import (
    RiskScorer,
    RiskAssessment,
    LEVEL_SAFE,
    LEVEL_WARNING,
    LEVEL_DANGER,
    LEVEL_CRITICAL,
)
from src.features.visual_features import VisualFeatures
from src.features.rppg_features import RPPGFeatures
from src.features.audio_features import AudioFeatures

TEST_CFG = {
    "fusion": {
        "weight_visual": 0.30,
        "weight_rppg": 0.20,
        "weight_audio": 0.25,
        "weight_smartwatch": 0.25,
        "smoothing_alpha": 0.3,
        "visual_confidence_threshold": 0.5,
        "rppg_confidence_threshold": 0.3,
        "audio_confidence_threshold": 0.3,
        "smartwatch_confidence_threshold": 0.4,
    },
    "smartwatch": {
        "enabled": True,
        "provider": "mock",
        "poll_interval_sec": 1.0,
        "simulation_mode": True,
        "low_hr_threshold": 50.0,
        "high_hr_threshold": 120.0,
        "low_spo2_threshold": 92.0,
        "high_stress_threshold": 80.0,
    },
    "alert": {
        "level_0_max": 0.25,
        "level_1_max": 0.50,
        "level_2_max": 0.75,
        "cooldown_sec": 0.1,
        "tts_enabled": False,
    },
    "visual": {
        "ear_threshold": 0.21,
        "mar_threshold": 0.65,
        "perclos_threshold_warning": 0.15,
        "perclos_threshold_danger": 0.30,
    },
    "rppg": {
        "normal_hr_min": 55.0,
        "normal_hr_max": 100.0,
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Smartwatch Features Module Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSmartwatchFeaturesModule:
    def test_fetch_health_data_schema(self):
        data = fetch_health_data(config=TEST_CFG, provider="mock")
        assert isinstance(data, dict)
        assert "heart_rate" in data
        assert "spo2" in data
        assert "stress_level" in data
        assert "hrv_rmssd" in data
        assert "confidence" in data
        assert "is_connected" in data
        assert data["heart_rate"] > 0
        assert 90 <= data["spo2"] <= 100
        assert 0 <= data["stress_level"] <= 100

    def test_normalize_to_1hz(self):
        # Raw dataset with 5-second intervals
        raw_dataset = [
            {"time": "12:00:00", "value": 70.0},
            {"time": "12:00:05", "value": 75.0},
            {"time": "12:00:10", "value": 80.0},
        ]
        norm = normalize_to_1hz({"dataset": raw_dataset})
        assert "timestamps" in norm
        assert "values" in norm
        assert norm["sample_rate_hz"] == 1.0
        # Check that 10-second span produced 11 points (0s to 10s inclusive)
        assert len(norm["values"]) == 11
        assert norm["values"][0] == 70.0
        assert norm["values"][-1] == 80.0
        # Intermediate interpolated value at 5 seconds
        assert norm["values"][5] == 75.0

    def test_normalize_to_1hz_empty_data(self):
        norm = normalize_to_1hz({})
        assert len(norm["values"]) == 60
        assert norm["sample_rate_hz"] == 1.0

    def test_validate_against_rppg(self):
        # High confidence when diff < 5 BPM
        conf_close = validate_against_rppg(rppg_hr=72.0, watch_hr=73.5)
        assert conf_close >= 0.90

        # Moderate confidence when 5 <= diff <= 15 BPM
        conf_mid = validate_against_rppg(rppg_hr=70.0, watch_hr=78.0)
        assert 0.40 <= conf_mid <= 0.90

        # Low confidence / anomaly when diff > 15 BPM
        conf_far = validate_against_rppg(rppg_hr=70.0, watch_hr=95.0)
        assert conf_far <= 0.30

        # Zero when missing
        assert validate_against_rppg(0.0, 72.0) == 0.0

    def test_smartwatch_extractor_lifecycle(self):
        extractor = SmartwatchFeatureExtractor(TEST_CFG)
        feat = extractor.get_features()
        assert isinstance(feat, SmartwatchFeatures)
        assert extractor.is_connected() is True
        assert feat.heart_rate_bpm > 0
        extractor.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Cross-Validation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    def test_status_agree(self):
        res = cross_validate_heart_rate(
            rppg_hr=72.0, rppg_quality=0.8,
            watch_hr=74.0, watch_connected=True, watch_confidence=0.9
        )
        assert res.status == STATUS_AGREE
        assert res.anomaly_flag is False
        assert res.confidence >= 0.90
        assert res.diff_bpm == 2.0
        assert 72.0 <= res.consensus_hr <= 74.0

    def test_status_disagree_flags_anomaly(self):
        res = cross_validate_heart_rate(
            rppg_hr=68.0, rppg_quality=0.8,
            watch_hr=92.0, watch_connected=True, watch_confidence=0.9
        )
        assert res.status == STATUS_DISAGREE
        assert res.anomaly_flag is True
        assert res.confidence <= 0.35
        assert res.diff_bpm == 24.0

    def test_status_moderate(self):
        res = cross_validate_heart_rate(
            rppg_hr=70.0, rppg_quality=0.8,
            watch_hr=78.0, watch_connected=True, watch_confidence=0.9
        )
        assert res.status == STATUS_MODERATE
        assert res.anomaly_flag is False
        assert 0.50 <= res.confidence <= 0.90

    def test_status_watch_only(self):
        res = cross_validate_heart_rate(
            rppg_hr=0.0, rppg_quality=0.0,
            watch_hr=75.0, watch_connected=True, watch_confidence=0.9
        )
        assert res.status == STATUS_WATCH_ONLY
        assert res.consensus_hr == 75.0
        assert res.anomaly_flag is False

    def test_status_camera_only(self):
        res = cross_validate_heart_rate(
            rppg_hr=72.0, rppg_quality=0.7,
            watch_hr=0.0, watch_connected=False, watch_confidence=0.0
        )
        assert res.status == STATUS_CAMERA_ONLY
        assert res.consensus_hr == 72.0
        assert res.anomaly_flag is False

    def test_status_none(self):
        res = cross_validate_heart_rate(
            rppg_hr=0.0, rppg_quality=0.0,
            watch_hr=0.0, watch_connected=False, watch_confidence=0.0
        )
        assert res.status == STATUS_NONE
        assert res.confidence == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. 4-Modality Fusion & Dynamic Gating Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMultimodalFusion4Modalities:
    @pytest.fixture
    def fusion(self):
        return MultimodalFusion(TEST_CFG)

    def test_4_modality_weights_active(self, fusion):
        vis = VisualFeatures(face_detected=True, confidence=1.0)
        rppg = RPPGFeatures(signal_quality=0.8, heart_rate_bpm=72.0)
        aud = AudioFeatures(confidence=0.8)
        watch = SmartwatchFeatures(is_connected=True, confidence=0.9, heart_rate_bpm=73.0)

        state = fusion.fuse(vis, rppg, aud, watch)
        # All weights active
        assert abs(state.w_visual - 0.30) < 0.01
        assert abs(state.w_rppg - 0.20) < 0.01
        assert abs(state.w_audio - 0.25) < 0.01
        assert abs(state.w_smartwatch - 0.25) < 0.01
        assert state.cross_validation_status == STATUS_AGREE

    def test_smartwatch_disconnected_weight_redistribution(self, fusion):
        vis = VisualFeatures(face_detected=True, confidence=1.0)
        rppg = RPPGFeatures(signal_quality=0.8, heart_rate_bpm=70.0)
        aud = AudioFeatures(confidence=0.8)
        # Smartwatch disconnected
        watch_offline = SmartwatchFeatures(is_connected=False, confidence=0.0)

        state = fusion.fuse(vis, rppg, aud, watch_offline)
        # Smartwatch weight must be 0
        assert state.w_smartwatch == 0.0
        # Redistributed across active 3 modalities:
        # Base ratio 0.30 / 0.20 / 0.25 on total 0.75:
        # 0.30/0.75 = 0.40, 0.20/0.75 = 0.2667, 0.25/0.75 = 0.3333
        assert abs(state.w_visual - 0.40) < 0.02
        assert abs(state.w_rppg - 0.267) < 0.02
        assert abs(state.w_audio - 0.333) < 0.02
        # Sum of active weights must be exactly 1.0
        assert abs((state.w_visual + state.w_rppg + state.w_audio) - 1.0) < 1e-5
        assert state.cross_validation_status == STATUS_CAMERA_ONLY

    def test_smartwatch_subscore_computation(self, fusion):
        # Normal physiology
        normal_sw = SmartwatchFeatures(
            is_connected=True, confidence=0.9,
            heart_rate_bpm=72.0, spo2=98.5, stress_level=25.0, hrv_rmssd=40.0
        )
        score_normal = fusion._smartwatch_subscore(normal_sw)
        assert score_normal == 0.0

        # High risk physiology: bradycardia + hypoxia + extreme stress
        abnormal_sw = SmartwatchFeatures(
            is_connected=True, confidence=0.9,
            heart_rate_bpm=42.0, spo2=89.0, stress_level=90.0, hrv_rmssd=12.0
        )
        score_abnormal = fusion._smartwatch_subscore(abnormal_sw)
        assert score_abnormal > 0.60


# ══════════════════════════════════════════════════════════════════════════════
# 4. Smartwatch Physiological Alert Rules Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSmartwatchAlertRules:
    @pytest.fixture
    def scorer(self):
        return RiskScorer(TEST_CFG)

    def test_rule_hr_anomaly_agree_triggers_level_3(self, scorer):
        """If smartwatch HR < 50 or > 120 AND rPPG agrees → Level 3 alert."""
        state = FusedState(
            smoothed_score=0.20,  # Low continuous score (normally SAFE)
            smartwatch_connected=True,
            smartwatch_hr=44.0,   # Bradycardia
            heart_rate=45.0,
            cross_validation_status=STATUS_AGREE,
            cross_validation_diff=1.0,
        )
        assessment = scorer.score(state)
        assert assessment.alert_level == LEVEL_CRITICAL
        assert assessment.alert_label == "CRITICAL"
        assert "Abnormal HR" in assessment.alert_reason

    def test_rule_hr_anomaly_disagree_does_not_override_level_3(self, scorer):
        """If smartwatch HR < 50 but rPPG disagrees → Do not force Level 3."""
        state = FusedState(
            smoothed_score=0.20,
            smartwatch_connected=True,
            smartwatch_hr=44.0,
            heart_rate=75.0,
            cross_validation_status=STATUS_DISAGREE,
            cross_validation_diff=31.0,
        )
        assessment = scorer.score(state)
        # Should stay at baseline score level (SAFE) because sensors disagree
        assert assessment.alert_level == LEVEL_SAFE

    def test_rule_low_spo2_triggers_level_3(self, scorer):
        """If smartwatch SpO2 < 92% → Level 3 alert."""
        state = FusedState(
            smoothed_score=0.20,
            smartwatch_connected=True,
            smartwatch_spo2=89.5,
        )
        assessment = scorer.score(state)
        assert assessment.alert_level == LEVEL_CRITICAL
        assert "hypoxemia" in assessment.alert_reason.lower()

    def test_rule_high_stress_with_visual_fatigue_triggers_level_2(self, scorer):
        """If smartwatch stress > 80 AND visual fatigue high → Level 2 alert."""
        state = FusedState(
            smoothed_score=0.20,
            smartwatch_connected=True,
            smartwatch_stress=85.0,
            visual_score=0.35,  # High visual fatigue
        )
        assessment = scorer.score(state)
        assert assessment.alert_level >= LEVEL_DANGER
        assert "High stress" in assessment.alert_reason


# ══════════════════════════════════════════════════════════════════════════════
# 5. Full End-to-End Pipeline Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationWithSmartwatch:
    def test_full_pipeline_with_smartwatch_active(self):
        vis = VisualFeatures(face_detected=True, confidence=1.0, ear=0.28, perclos=0.05)
        rppg = RPPGFeatures(signal_quality=0.8, heart_rate_bpm=72.0)
        aud = AudioFeatures(confidence=0.8, audio_drowsiness_score=0.1)

        extractor = SmartwatchFeatureExtractor(TEST_CFG)
        watch_feat = extractor.get_features()
        fusion = MultimodalFusion(TEST_CFG)
        scorer = RiskScorer(TEST_CFG)

        fused = fusion.fuse(vis, rppg, aud, watch_feat)
        ra = scorer.score(fused)

        assert 0.0 <= fused.smoothed_score <= 1.0
        assert fused.smartwatch_connected is True
        assert fused.smartwatch_hr > 0
        assert fused.cross_validation_status in (STATUS_AGREE, STATUS_MODERATE, STATUS_WATCH_ONLY)
        assert ra.alert_level in (LEVEL_SAFE, LEVEL_WARNING, LEVEL_DANGER, LEVEL_CRITICAL)

        extractor.stop()
