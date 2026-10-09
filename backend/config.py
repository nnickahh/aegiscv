"""Configuration settings for the AI Emergency Coordinator (OpenCV 5 + AWS)."""
import os
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"
S3_MIRROR_DIR = DATA_DIR / "s3_bucket"
DB_PATH = DATA_DIR / "emergency_coordinator.db"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
S3_MIRROR_DIR.mkdir(parents=True, exist_ok=True)

# Canonical OpenCV frame processing dimensions
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
TARGET_FPS = 24

# Default Restricted Area Polygon (Normalized [0..1] and Pixel coordinates for 960x540)
# Represents the High-Voltage / Hazmat Containment Zone in the top-right quadrant
RESTRICTED_ZONE_POLYGON: List[Tuple[int, int]] = [
    (640, 95),
    (920, 95),
    (940, 315),
    (610, 315),
]

# OpenCV 5 Pipeline Default Thresholds
DEFAULT_CV_CONFIG = {
    "confidence_threshold": 0.48,
    "fall_aspect_ratio_threshold": 1.15,
    "fall_immobile_velocity_threshold": 1.8,
    "fall_confirm_frames": 10,
    "fire_min_area": 220,
    "smoke_min_area": 450,
    "crowd_min_people": 4,
    "crowd_cluster_radius_px": 260.0,
    "clahe_enabled": False,
    "sensitivity_mode": "STANDARD",  # STANDARD | HIGH_SENSITIVITY | VERIFICATION_SCAN
    "roi_zoom_active": False,
    "roi_zoom_box": None,  # [x1, y1, x2, y2]
    "roi_target_label": None,
}

# AWS Configuration (supports live AWS IAM credentials via env vars + hybrid cloud mirror)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "opencv5-ai-emergency-coordinator-evidence")
AWS_DYNAMODB_TABLE = os.getenv("AWS_DYNAMODB_TABLE", "EmergencyCoordinatorIncidents")
AWS_CLOUDWATCH_NAMESPACE = os.getenv("AWS_CLOUDWATCH_NAMESPACE", "OpenCV5/EmergencyCoordinator")
AWS_BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
AWS_SNS_TOPIC_ARN = os.getenv("AWS_SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:000000000000:EmergencyResponderDispatch")
