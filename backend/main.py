"""
FastAPI Backend for AI Emergency Coordinator (OpenCV 5 + AWS Hackathon).

Serves:
- Real-time OpenCV 5 video analysis pipeline (MJPEG stream + low-latency frame API)
- Custom video file upload (`.mp4`/`.avi`/`.webm`) & pre-generated sample video loader
- Live browser webcam frame analysis endpoint
- Closed-loop AI Agent state, Incident Panel, Event Timeline, and 2D Scene Map coordinates
- Human-in-the-Loop approval & active OpenCV re-scan controls
- AWS Cloud status & SDK payload inspector (Bedrock, S3, DynamoDB, CloudWatch, SNS)
- Quantitative Evaluation & Live Benchmark Runner
- Static Frontend Command Center SPA
"""
import asyncio
import base64
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.ai_agent import AIEmergencyCoordinatorAgent
from backend.aws_service import AWSCloudCoordinator
from backend.config import (
    BASE_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RESTRICTED_ZONE_POLYGON,
    S3_MIRROR_DIR,
    SAMPLE_VIDEOS_DIR,
)
from backend.database import DatabaseManager
from backend.evaluator import SystemEvaluator
from backend.models import (
    HumanApprovalRequest,
    ScenarioSwitchRequest,
    ViewModeRequest,
)
from backend.opencv_pipeline import OpenCV5VisionPipeline
from backend.scenario_synthesizer import (
    SCENARIO_CATALOG,
    SurveillanceSceneSynthesizer,
    generate_sample_mp4_files,
)

FRONTEND_DIR = BASE_DIR / "frontend"


class WebcamFramePayload(BaseModel):
    image_base64: str


class SampleVideoSelectPayload(BaseModel):
    filename: str


class CoordinatorRuntime:
    """Thread-safe runtime orchestrating the video source, OpenCV 5 pipeline, AI Agent, and AWS services."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.db = DatabaseManager()
        self.aws = AWSCloudCoordinator(self.db)
        self.vision = OpenCV5VisionPipeline()
        self.agent = AIEmergencyCoordinatorAgent(self.vision, self.db, self.aws)
        self.synthesizer = SurveillanceSceneSynthesizer()
        self.evaluator = SystemEvaluator(self.db, self.aws)

        self.source_mode: str = "SIMULATION"  # SIMULATION | UPLOADED_VIDEO | WEBCAM
        self.view_mode: str = "opencv_annotated"  # opencv_annotated | raw_video | optical_flow | split_comparison
        self.uploaded_video_path: Optional[Path] = None
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.paused: bool = False
        self.last_step_ts: float = 0.0
        self.cached_jpeg: bytes = b""

        # Warm up initial frame so state is immediately populated on startup
        self.step_pipeline(force=True)

    def set_scenario(self, scenario_id: str) -> None:
        with self.lock:
            if scenario_id not in SCENARIO_CATALOG:
                raise ValueError(f"Unknown scenario_id: {scenario_id}")
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self.source_mode = "SIMULATION"
            self.synthesizer.set_scenario(scenario_id)
            self.vision.reset_state()
            self.agent.reset_agent()
        self.step_pipeline(force=True)

    def load_video_file(self, video_path: Path) -> None:
        with self.lock:
            if self.video_capture is not None:
                self.video_capture.release()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV VideoCapture failed to open {video_path.name}")
            self.video_capture = cap
            self.uploaded_video_path = video_path
            self.source_mode = "UPLOADED_VIDEO"
            self.vision.reset_state()
            self.agent.reset_agent()
        self.step_pipeline(force=True)

    def step_pipeline(self, force: bool = False) -> bytes:
        """Advance video source by one frame, run OpenCV 5 analysis + AI Agent, and encode output JPEG."""
        now = time.perf_counter()
        # Rate-limit step to ~20 FPS even if multiple clients request frames simultaneously
        if not force and (now - self.last_step_ts) < 0.048 and self.cached_jpeg:
            return self.cached_jpeg

        with self.lock:
            if self.paused and not force and self.cached_jpeg:
                return self.cached_jpeg

            raw_frame: Optional[np.ndarray] = None
            if self.source_mode == "UPLOADED_VIDEO" and self.video_capture is not None:
                ret, frame = self.video_capture.read()
                if not ret:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    raw_frame = frame
                else:
                    self.source_mode = "SIMULATION"

            if raw_frame is None:
                raw_frame = self.synthesizer.generate_next_raw_frame()

            telemetry, annotated = self.vision.process_frame(raw_frame)
            self.agent.evaluate_telemetry(telemetry, annotated)

            display_frame = self._select_display_frame()
            ok, encoded = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            if ok:
                self.cached_jpeg = encoded.tobytes()
            self.last_step_ts = time.perf_counter()
            return self.cached_jpeg

    def process_webcam_frame(self, bgr_frame: np.ndarray) -> bytes:
        """Process a live webcam frame uploaded from the browser."""
        with self.lock:
            self.source_mode = "WEBCAM"
            telemetry, annotated = self.vision.process_frame(bgr_frame)
            self.agent.evaluate_telemetry(telemetry, annotated)
            display_frame = self._select_display_frame()
            ok, encoded = cv2.imencode(".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                self.cached_jpeg = encoded.tobytes()
            self.last_step_ts = time.perf_counter()
            return self.cached_jpeg

    def _select_display_frame(self) -> np.ndarray:
        if self.view_mode == "raw_video" and self.vision.last_raw_frame is not None:
            return self.vision.last_raw_frame
        if self.view_mode == "optical_flow" and self.vision.last_flow_frame is not None:
            return self.vision.last_flow_frame
        if self.view_mode == "split_comparison" and self.vision.last_split_frame is not None:
            return self.vision.last_split_frame
        if self.vision.last_annotated_frame is not None:
            return self.vision.last_annotated_frame
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


runtime = CoordinatorRuntime()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Ensure sample MP4 emergency videos are generated on startup."""
    generate_sample_mp4_files()
    yield


app = FastAPI(
    title="AI Emergency Coordinator — OpenCV 5 + AWS",
    description="Closed-Loop Agentic Computer Vision Emergency Response System (See → Understand → Prioritize → Act → Verify)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "project": "AI Emergency Coordinator",
        "opencv_version": cv2.__version__,
        "source_mode": runtime.source_mode,
        "active_scenario": runtime.synthesizer.current_scenario,
    }


@app.get("/api/scenarios")
async def list_scenarios() -> Dict[str, Any]:
    sample_files = [p.name for p in sorted(SAMPLE_VIDEOS_DIR.glob("*.mp4"))]
    return {
        "scenarios": list(SCENARIO_CATALOG.values()),
        "active_scenario": runtime.synthesizer.current_scenario,
        "sample_videos": sample_files,
    }


@app.post("/api/scenario/select")
async def select_scenario(req: ScenarioSwitchRequest) -> Dict[str, Any]:
    try:
        runtime.set_scenario(req.scenario_id)
        if req.view_mode:
            runtime.view_mode = req.view_mode
        return {
            "status": "switched",
            "scenario_id": req.scenario_id,
            "scenario": SCENARIO_CATALOG[req.scenario_id],
            "view_mode": runtime.view_mode,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/video/view-mode")
async def set_view_mode(req: ViewModeRequest) -> Dict[str, Any]:
    valid_modes = {"opencv_annotated", "raw_video", "optical_flow", "split_comparison"}
    if req.view_mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid view_mode. Must be one of {valid_modes}")
    runtime.view_mode = req.view_mode
    runtime.step_pipeline(force=True)
    return {"status": "ok", "view_mode": runtime.view_mode}


@app.post("/api/video/pause")
async def toggle_pause() -> Dict[str, Any]:
    runtime.paused = not runtime.paused
    return {"status": "ok", "paused": runtime.paused}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an uploaded `.mp4`/`.avi`/`.webm` emergency video and run OpenCV 5 analysis on it."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing video filename")
    safe_name = Path(file.filename).name
    dest_path = SAMPLE_VIDEOS_DIR / f"uploaded_{safe_name}"
    content = await file.read()
    if len(content) < 512:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty or invalid")
    dest_path.write_bytes(content)

    try:
        runtime.load_video_file(dest_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "uploaded",
        "filename": dest_path.name,
        "source_mode": runtime.source_mode,
    }


@app.post("/api/video/load-sample")
async def load_sample_video(req: SampleVideoSelectPayload) -> Dict[str, Any]:
    """Load one of the pre-generated `.mp4` sample videos through `cv2.VideoCapture`."""
    target = SAMPLE_VIDEOS_DIR / Path(req.filename).name
    if not target.exists():
        generate_sample_mp4_files()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Sample video {req.filename} not found")
    runtime.load_video_file(target)
    return {
        "status": "loaded",
        "filename": target.name,
        "source_mode": runtime.source_mode,
    }


@app.post("/api/video/webcam-frame")
async def analyze_webcam_frame(payload: WebcamFramePayload) -> Dict[str, Any]:
    """Decode base64 webcam frame from browser, analyze with OpenCV 5 + AI Agent, return telemetry."""
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(raw_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid webcam image frame")
    jpg_out = runtime.process_webcam_frame(bgr)
    b64_out = "data:image/jpeg;base64," + base64.b64encode(jpg_out).decode("ascii")
    return {"status": "ok", "annotated_frame_base64": b64_out}


@app.get("/api/video/frame")
async def get_single_video_frame() -> Response:
    """Return the latest processed OpenCV 5 video frame as `image/jpeg`."""
    jpg_bytes = runtime.step_pipeline()
    return Response(
        content=jpg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/video/stream")
async def stream_mjpeg_video() -> StreamingResponse:
    """Continuous multipart MJPEG stream of the OpenCV 5 pipeline output."""

    async def frame_generator():
        while True:
            jpg_bytes = runtime.step_pipeline()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
            await asyncio.sleep(0.055)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


class RestrictedZoneUpdatePayload(BaseModel):
    points_normalized: Optional[List[List[float]]] = None
    vertices: Optional[List[List[float]]] = None
    reset: bool = False


@app.post("/api/video/restricted-zone")
async def update_restricted_zone(payload: RestrictedZoneUpdatePayload) -> Dict[str, Any]:
    """Allow operator/judge to interactively draw a custom Restricted Hazard Polygon on the video."""
    pts = payload.vertices or payload.points_normalized
    with runtime.lock:
        if payload.reset or not pts:
            runtime.vision.reset_restricted_polygon()
            mode = "default"
        else:
            if len(pts) < 3:
                raise HTTPException(status_code=400, detail="Polygon requires at least 3 vertices")
            px_pts = [
                [int(round(pt[0] * FRAME_WIDTH)), int(round(pt[1] * FRAME_HEIGHT))]
                for pt in pts
            ]
            runtime.vision.set_restricted_polygon(px_pts)
            mode = "custom"

    runtime.step_pipeline(force=True)
    norm_polygon = [
        [round(int(x) / FRAME_WIDTH, 3), round(int(y) / FRAME_HEIGHT, 3)]
        for x, y in runtime.vision.restricted_polygon
    ]
    return {
        "status": "ok",
        "mode": mode,
        "polygon": norm_polygon,
        "restricted_zone_polygon": norm_polygon,
    }


class ROILockPayload(BaseModel):
    x_norm: float
    y_norm: float


@app.post("/api/video/roi-lock")
async def set_manual_roi_target_lock(payload: ROILockPayload) -> Dict[str, Any]:
    """Lock the OpenCV 2.4x Picture-in-Picture ROI Zoom onto an operator-clicked coordinate."""
    with runtime.lock:
        box = runtime.vision.set_manual_roi_lock(payload.x_norm, payload.y_norm)
    runtime.step_pipeline(force=True)
    return {"status": "ok", "roi_box": box}


class CVTunablesPayload(BaseModel):
    degraded_lighting_stress: Optional[bool] = None
    clahe_override: Optional[str] = None
    fall_aspect_threshold: Optional[float] = None


@app.post("/api/video/cv-tunables")
async def update_cv_tunables(payload: CVTunablesPayload) -> Dict[str, Any]:
    """Update interactive OpenCV 5 pipeline tunables and camera stress-test mode."""
    with runtime.lock:
        if payload.degraded_lighting_stress is not None:
            runtime.vision.degraded_lighting_stress = bool(payload.degraded_lighting_stress)
        if payload.clahe_override in ("AUTO", "FORCE_ON", "FORCE_OFF"):
            runtime.vision.clahe_override = payload.clahe_override
        if payload.fall_aspect_threshold is not None:
            runtime.vision.fall_aspect_threshold = max(0.85, min(2.2, float(payload.fall_aspect_threshold)))
    runtime.step_pipeline(force=True)
    return {
        "status": "ok",
        "degraded_lighting_stress": runtime.vision.degraded_lighting_stress,
        "clahe_override": runtime.vision.clahe_override,
        "fall_aspect_threshold": runtime.vision.fall_aspect_threshold,
    }


@app.get("/api/state")
async def get_command_center_state(advance: bool = True) -> Dict[str, Any]:
    """
    Return the complete synchronized state for the Command Center UI:
    - OpenCV 5 frame telemetry & detections
    - AI Agent decision (See → Understand → Prioritize → Act → Verify), reasoning trace, uncertainty
    - Active Incidents & Human-in-the-Loop approval status
    - Step-by-step Event Timeline
    - 2D Scene Map spatial coordinates & Restricted Zone polygon
    - AWS System Status & recent cloud operations
    """
    if advance and runtime.source_mode != "WEBCAM":
        runtime.step_pipeline()

    telemetry = runtime.vision.last_telemetry
    decision = runtime.agent.last_decision
    incidents = runtime.db.get_incidents(limit=15)
    timeline = runtime.db.get_timeline_events(limit=30)
    aws_status = runtime.aws.get_system_status()

    norm_polygon = [
        [round(int(x) / FRAME_WIDTH, 3), round(int(y) / FRAME_HEIGHT, 3)]
        for x, y in runtime.vision.restricted_polygon
    ]

    return {
        "source_mode": runtime.source_mode,
        "active_scenario": runtime.synthesizer.current_scenario,
        "scenario_info": SCENARIO_CATALOG.get(runtime.synthesizer.current_scenario, {}),
        "view_mode": runtime.view_mode,
        "paused": runtime.paused,
        "opencv_version": cv2.__version__,
        "restricted_zone_polygon": norm_polygon,
        "telemetry": telemetry.model_dump() if telemetry else None,
        "agent_decision": decision.model_dump() if decision else None,
        "incidents": [inc.model_dump() for inc in incidents],
        "timeline": [evt.model_dump() for evt in timeline],
        "aws_status": aws_status.model_dump(),
    }


@app.post("/api/incidents/approval")
async def submit_human_approval(req: HumanApprovalRequest) -> Dict[str, Any]:
    """Process Human-in-the-Loop approval (`APPROVE`, `REQUEST_RESCAN`, `DISMISS`)."""
    updated = runtime.agent.handle_human_approval(
        incident_id=req.incident_id,
        action=req.action,
        operator_note=req.operator_note,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Incident {req.incident_id} not found")
    return {
        "status": "updated",
        "action": req.action,
        "incident": updated.model_dump(),
    }


@app.get("/api/incidents/export-report")
async def export_incident_audit_report() -> Response:
    """Export a formatted Markdown AWS + OpenCV 5 Post-Incident Audit Report."""
    incidents = runtime.db.get_incidents(limit=25)
    timeline = runtime.db.get_timeline_events(limit=30)
    aws_status = runtime.aws.get_system_status()
    tel = runtime.vision.last_telemetry
    dec = runtime.agent.last_decision

    lines = [
        "# AI Emergency Coordinator — Post-Incident Audit Report (OpenCV 5 + AWS)",
        f"- **Generated At**: {tel.timestamp if tel else 'Live Session'} UTC",
        f"- **OpenCV Engine Version**: {cv2.__version__} (Mode: {tel.active_cv_mode if tel else 'STANDARD'})",
        f"- **AWS Connection Mode**: {aws_status.connection_mode} ({aws_status.region})",
        f"- **S3 Evidence Bucket**: `s3://{aws_status.s3_bucket}` ({aws_status.s3_objects_count} objects)",
        f"- **DynamoDB Table**: `{aws_status.dynamodb_table}` ({aws_status.dynamodb_items_count} state syncs)",
        "",
        "## 1. Current AI Coordinator Decision State",
    ]
    if dec:
        lines.extend([
            f"- **Stage**: `{dec.current_stage}` | **Priority**: `{dec.priority_level}` ({dec.priority_score}/100)",
            f"- **Headline**: {dec.headline}",
            f"- **Situation Understanding**: {dec.situation_understanding}",
            f"- **Recommended Action**: {dec.recommended_action}",
            f"- **Active OpenCV Feedback Command**: `{dec.active_cv_command}`",
            f"- **Confidence**: {round(dec.confidence * 100, 1)}% (Uncertainty: {round(dec.uncertainty_score * 100, 1)}%)",
            "",
            "### Reasoning Trace (`See → Understand → Prioritize → Act → Verify`)",
        ])
        for step in dec.reasoning_trace:
            lines.append(f"- {step}")

    lines.extend(["", "## 2. Recorded Emergency Incidents (Amazon DynamoDB)"])
    if not incidents:
        lines.append("_No incidents recorded in current session._")
    else:
        for inc in incidents:
            lines.extend([
                f"### {inc.id}: {inc.title}",
                f"- **Severity**: `{inc.severity}` (Priority {inc.priority_score}/100) | **Status**: `{inc.status}` | **Stage**: `{inc.pipeline_stage}`",
                f"- **Human Approval**: `{inc.human_approval_status}` | **S3 Evidence Key**: `{inc.s3_evidence_key}`",
                f"- **OpenCV Evidence Summary**: `{inc.cv_evidence_summary}`",
                f"- **Recommended Action**: {inc.recommended_action}",
                "- **Evolution History**:",
            ])
            for note in inc.evolution_log:
                lines.append(f"  - {note}")
            lines.append("")

    lines.extend(["## 3. Chronological Event Timeline", ""])
    for ev in timeline[:20]:
        lines.append(f"- **`[{ev.timestamp}]` [{ev.stage}] ({ev.source})** **{ev.title}** — {ev.detail}")

    report_md = "\n".join(lines)
    return Response(
        content=report_md,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="AI_Emergency_Coordinator_Audit_Report.md"'
        },
    )


@app.get("/api/evaluation")
async def get_evaluation_metrics() -> Dict[str, Any]:
    """Return quantitative evaluation report, failure cases, and limitations."""
    return runtime.evaluator.get_evaluation_report()


@app.post("/api/evaluation/run-benchmark")
async def trigger_live_benchmark() -> Dict[str, Any]:
    """Run live OpenCV 5 + AI Agent evaluation benchmark across all scenarios."""
    report = runtime.evaluator.run_live_evaluation_benchmark()
    return report


@app.get("/api/s3-evidence")
async def list_s3_evidence_artifacts() -> Dict[str, Any]:
    """List archived emergency snapshots & manifests in the S3 bucket mirror."""
    files = sorted(S3_MIRROR_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "bucket": runtime.aws.s3_bucket,
        "artifacts": [
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "url": f"/api/s3-evidence/{f.name}",
            }
            for f in files[:16]
        ],
    }


@app.get("/api/s3-evidence/{filename}")
async def get_s3_evidence_file(filename: str) -> FileResponse:
    safe_path = S3_MIRROR_DIR / Path(filename).name
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="Evidence artifact not found")
    return FileResponse(str(safe_path))


# Mount static frontend assets
if FRONTEND_DIR.exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


@app.get("/")
async def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"message": "Frontend initializing..."})
    return FileResponse(str(index_path))
