"""
VirtualPhysio-Driver — fusion package
"""
from .multimodal_fusion import MultimodalFusion, FusedState
from .cross_validation import (
    CrossValidator,
    CrossValidationResult,
    cross_validate_heart_rate,
    STATUS_AGREE,
    STATUS_DISAGREE,
    STATUS_MODERATE,
    STATUS_WATCH_ONLY,
    STATUS_CAMERA_ONLY,
    STATUS_NONE,
)

__all__ = [
    "MultimodalFusion",
    "FusedState",
    "CrossValidator",
    "CrossValidationResult",
    "cross_validate_heart_rate",
    "STATUS_AGREE",
    "STATUS_DISAGREE",
    "STATUS_MODERATE",
    "STATUS_WATCH_ONLY",
    "STATUS_CAMERA_ONLY",
    "STATUS_NONE",
]
