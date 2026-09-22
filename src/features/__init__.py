"""
D-SAAT — features package
"""
from .visual_features import VisualFeatureExtractor, VisualFeatures
from .rppg_features import RPPGExtractor, RPPGFeatures
from .audio_features import AudioFeatureExtractor, AudioFeatures
from .smartwatch_features import (
    SmartwatchFeatureExtractor,
    SmartwatchFeatures,
    fetch_health_data,
    normalize_to_1hz,
    validate_against_rppg,
)

__all__ = [
    "VisualFeatureExtractor", "VisualFeatures",
    "RPPGExtractor", "RPPGFeatures",
    "AudioFeatureExtractor", "AudioFeatures",
    "SmartwatchFeatureExtractor", "SmartwatchFeatures",
    "fetch_health_data", "normalize_to_1hz", "validate_against_rppg",
]
