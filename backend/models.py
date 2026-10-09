"""Pydantic models for OpenCV 5 detections, AI Agent state, Incidents, Timeline, and AWS telemetry."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    id: str
    category: str  # person | fall | fire | smoke | crowd | movement | restricted_area
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    bbox: List[int]  # [x, y, w, h] in 960x540 coordinates
    world_pos: List[float]  # [nx, ny] normalized 0.0-1.0 scene map coordinates
    velocity: float = 0.0
    posture_state: str = "NORMAL"  # UPRIGHT | FALLING | FALLEN_IMMOBILE | RECOVERING | N/A
    in_restricted_zone: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ROIZoomTelemetry(BaseModel):
    active: bool = False
    target_id: Optional[str] = None
    target_category: Optional[str] = None
    bbox: Optional[List[int]] = None
    micro_motion_score: float = 0.0
    sharpness_score: float = 0.0
    clahe_contrast_gain: float = 1.0
    verification_conclusion: str = "IDLE"


class OpenCVFrameTelemetry(BaseModel):
    frame_index: int
    timestamp: str
    fps: float
    processing_ms: float
    motion_energy: float
    optical_flow_peak: float
    active_cv_mode: str
    clahe_active: bool
    roi_zoom: ROIZoomTelemetry
    people_count: int
    hazard_count: int
    crowd_density: float
    detections: List[DetectionItem]


class AgentDecision(BaseModel):
    decision_id: str
    timestamp: str
    current_stage: str  # SEE | UNDERSTAND | PRIORITIZE | ACT | VERIFY
    headline: str
    situation_understanding: str
    priority_level: str  # CRITICAL | HIGH | MEDIUM | LOW | NOMINAL
    priority_score: int
    recommended_action: str
    active_cv_command: Optional[str] = None
    confidence: float
    uncertainty_score: float
    uncertainty_flag: bool = False
    uncertainty_reason: Optional[str] = None
    requires_human_approval: bool = False
    approval_incident_id: Optional[str] = None
    decision_changed: bool = False
    previous_decision: Optional[str] = None
    change_reason: Optional[str] = None
    verification_status: str = "MONITORING"  # MONITORING | REQUESTING_MORE_ANALYSIS | ACTION_PENDING | VERIFYING_OUTCOME | VERIFIED_SAFE
    reasoning_trace: List[str] = Field(default_factory=list)


class IncidentRecord(BaseModel):
    id: str
    title: str
    category: str  # fall | fire | smoke | crowd | restricted_area | compound
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW | RESOLVED
    priority_score: int
    status: str  # DETECTING | ANALYZING_ROI | AWAITING_APPROVAL | DISPATCHED | VERIFYING | VERIFIED | RESOLVED
    pipeline_stage: str  # SEE | UNDERSTAND | PRIORITIZE | ACT | VERIFY
    confidence: float
    uncertainty: float
    world_pos: List[float]
    bbox: List[int]
    recommended_action: str
    human_approval_status: str  # NOT_REQUIRED | PENDING | APPROVED | OVERRIDDEN
    cv_evidence_summary: str
    s3_evidence_key: Optional[str] = None
    created_at: str
    updated_at: str
    evolution_log: List[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    id: str
    timestamp: str
    stage: str  # SEE | UNDERSTAND | PRIORITIZE | ACT | VERIFY
    source: str  # OPENCV_5 | AI_AGENT | HUMAN_OPERATOR | AWS_CLOUD
    title: str
    detail: str
    severity: str  # INFO | WARNING | HIGH | CRITICAL | SUCCESS


class HumanApprovalRequest(BaseModel):
    incident_id: str
    action: str  # APPROVE | REQUEST_RESCAN | DISMISS
    operator_note: Optional[str] = None


class ScenarioSwitchRequest(BaseModel):
    scenario_id: str  # normal_patrol | fall_emergency | fall_recovery | fire_smoke | crowd_surge | restricted_intrusion | multi_crisis
    view_mode: Optional[str] = None


class ViewModeRequest(BaseModel):
    view_mode: str  # opencv_annotated | raw_video | optical_flow | split_comparison


class AWSSystemStatus(BaseModel):
    connection_mode: str  # LIVE_AWS_IAM | HYBRID_CLOUD_MIRROR
    region: str
    latency_ms: float
    s3_bucket: str
    s3_objects_count: int
    dynamodb_table: str
    dynamodb_items_count: int
    cloudwatch_namespace: str
    cloudwatch_metrics_pushed: int
    bedrock_model: str
    bedrock_invocations: int
    sns_alerts_dispatched: int
    services_health: Dict[str, str]
    recent_cloud_events: List[Dict[str, Any]]
