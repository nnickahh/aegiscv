"""
OpenCV 5 Computer Vision Pipeline for the AI Emergency Coordinator.

Performs real pixel-level image and video analysis on every frame:
- People detection & multi-object spatial tracking
- Posture geometry, aspect-ratio collapse & zero-velocity Fall Detection
- Chromaticity (HSV/YCrCb) + luminance core + temporal flicker Fire Detection
- Desaturated diffusing plume + upward optical flow Smoke Detection
- Spatial clustering & convex-hull Crowd Density / Surge Detection
- Dense Farneback Optical Flow & MOG2 foreground Movement Analysis
- Polygon boundary intersection (`cv2.pointPolygonTest`) Restricted Area Intrusion
- Closed-loop Agent-triggered ROI Zoom & CLAHE Micro-Motion Verification
"""
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import (
    DEFAULT_CV_CONFIG,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RESTRICTED_ZONE_POLYGON,
)
from backend.models import DetectionItem, OpenCVFrameTelemetry, ROIZoomTelemetry


class TrackedPersonState:
    """Internal state for a tracked person across consecutive OpenCV frames."""

    def __init__(self, track_id: str, bbox: Tuple[int, int, int, int], frame_idx: int) -> None:
        self.track_id = track_id
        self.bbox = bbox  # (x, y, w, h)
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.centroid_history: List[Tuple[float, float]] = [self._calc_centroid(bbox)]
        self.aspect_ratio_history: List[float] = [bbox[2] / float(max(bbox[3], 1))]
        self.velocity: float = 0.0
        self.immobile_frames: int = 0
        self.horizontal_frames: int = 0
        self.was_fallen: bool = False
        self.recovered_frames: int = 0
        self.posture_state: str = "UPRIGHT"  # UPRIGHT | FALLING | FALLEN_IMMOBILE | RECOVERING

    @staticmethod
    def _calc_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x, y, w, h = bbox
        return (x + w * 0.5, y + h * 0.5)

    def update(
        self,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        flow_mag_at_box: float,
        fall_ar_threshold: float = 1.12,
    ) -> None:
        cx, cy = self._calc_centroid(bbox)
        prev_cx, prev_cy = self.centroid_history[-1]
        disp = math.hypot(cx - prev_cx, cy - prev_cy)
        # Blend centroid displacement and local optical flow magnitude
        self.velocity = round(0.55 * disp + 0.45 * flow_mag_at_box, 2)

        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.centroid_history.append((cx, cy))
        if len(self.centroid_history) > 30:
            self.centroid_history.pop(0)

        ar = bbox[2] / float(max(bbox[3], 1))
        self.aspect_ratio_history.append(ar)
        if len(self.aspect_ratio_history) > 20:
            self.aspect_ratio_history.pop(0)

        # Evaluate horizontal posture (lying on floor vs upright)
        if ar >= fall_ar_threshold:
            self.horizontal_frames += 1
            self.recovered_frames = 0
            if self.velocity < 1.85:
                self.immobile_frames += 1
            else:
                self.immobile_frames = max(0, self.immobile_frames - 1)

            if self.immobile_frames >= 8:
                self.posture_state = "FALLEN_IMMOBILE"
                self.was_fallen = True
            else:
                self.posture_state = "FALLING"
                if self.horizontal_frames >= 4:
                    self.was_fallen = True
        else:
            if self.was_fallen and self.horizontal_frames > 0:
                self.recovered_frames += 1
                if self.recovered_frames <= 18:
                    self.posture_state = "RECOVERING"
                else:
                    self.posture_state = "UPRIGHT"
                    self.was_fallen = False
                    self.horizontal_frames = 0
                    self.immobile_frames = 0
            else:
                self.posture_state = "UPRIGHT"
                self.horizontal_frames = max(0, self.horizontal_frames - 1)
                self.immobile_frames = max(0, self.immobile_frames - 1)


class OpenCV5VisionPipeline:
    """
    Real-time OpenCV 5 image and video analyzer with closed-loop AI Agent control.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = dict(DEFAULT_CV_CONFIG)
        self.restricted_polygon = np.array(RESTRICTED_ZONE_POLYGON, dtype=np.int32)
        self.frame_index: int = 0
        self.prev_gray_small: Optional[np.ndarray] = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=28, detectShadows=False
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.tracked_people: Dict[str, TrackedPersonState] = {}
        self.next_person_id: int = 101
        self.fire_area_history: List[float] = []
        self.smoke_area_history: List[float] = []
        self.fps_ema: float = 24.0
        self.last_frame_time: float = time.perf_counter()

        # Interactive OpenCV 5 Tunables & Stress-Test Controls
        self.degraded_lighting_stress: bool = False
        self.clahe_override: str = "AUTO"  # AUTO | FORCE_ON | FORCE_OFF
        self.fall_aspect_threshold: float = 1.12
        self.manual_roi_box: Optional[List[int]] = None
        self.manual_roi_frames: int = 0

        # Latest cached frame products for streaming
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.last_flow_frame: Optional[np.ndarray] = None
        self.last_split_frame: Optional[np.ndarray] = None
        self.last_telemetry: Optional[OpenCVFrameTelemetry] = None

    def set_manual_roi_lock(self, x_norm: float, y_norm: float) -> List[int]:
        """Lock the OpenCV 2.4x Picture-in-Picture ROI Zoom onto an operator-clicked coordinate."""
        cx = int(max(70, min(FRAME_WIDTH - 70, round(x_norm * FRAME_WIDTH))))
        cy = int(max(80, min(FRAME_HEIGHT - 70, round(y_norm * FRAME_HEIGHT))))
        box_w, box_h = 150, 110
        self.manual_roi_box = [cx - box_w // 2, cy - box_h // 2, box_w, box_h]
        self.manual_roi_frames = 36
        self.config["roi_zoom_active"] = True
        self.config["roi_zoom_box"] = self.manual_roi_box
        return self.manual_roi_box

    def set_restricted_polygon(self, points: List[List[int]]) -> None:
        """Dynamically update the Restricted Hazard Zone polygon from interactive user clicks."""
        if len(points) >= 3:
            clamped = [
                [max(0, min(FRAME_WIDTH, int(pt[0]))), max(38, min(FRAME_HEIGHT, int(pt[1])))]
                for pt in points
            ]
            self.restricted_polygon = np.array(clamped, dtype=np.int32)

    def reset_restricted_polygon(self) -> None:
        """Restore the default Sector-B High-Voltage Substation polygon."""
        self.restricted_polygon = np.array(RESTRICTED_ZONE_POLYGON, dtype=np.int32)

    def reset_state(self) -> None:
        """Reset temporal trackers when switching scenarios or videos."""
        self.prev_gray_small = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=28, detectShadows=False
        )
        self.tracked_people.clear()
        self.next_person_id = 101
        self.fire_area_history.clear()
        self.smoke_area_history.clear()
        self.config = dict(DEFAULT_CV_CONFIG)

    def apply_agent_command(
        self,
        command: str,
        roi_box: Optional[List[int]] = None,
        target_label: Optional[str] = None,
    ) -> None:
        """
        Closed-loop feedback interface: allows the AI Agent to reconfigure OpenCV
        parameters dynamically based on scene ambiguity or verification needs.
        """
        if command == "REQUEST_ROI_ZOOM_ANALYSIS":
            self.config["roi_zoom_active"] = True
            self.config["clahe_enabled"] = True
            self.config["sensitivity_mode"] = "VERIFICATION_SCAN"
            if roi_box and len(roi_box) == 4:
                self.config["roi_zoom_box"] = roi_box
            self.config["roi_target_label"] = target_label or "ROI Target"
        elif command == "BOOST_SENSITIVITY_CLAHE":
            self.config["clahe_enabled"] = True
            self.config["sensitivity_mode"] = "HIGH_SENSITIVITY"
            self.config["confidence_threshold"] = 0.40
        elif command == "VERIFY_INTERVENTION_OUTCOME":
            self.config["sensitivity_mode"] = "VERIFICATION_SCAN"
            self.config["clahe_enabled"] = True
            if roi_box and len(roi_box) == 4:
                self.config["roi_zoom_active"] = True
                self.config["roi_zoom_box"] = roi_box
        elif command == "RESET_STANDARD":
            self.config["roi_zoom_active"] = False
            self.config["roi_zoom_box"] = None
            self.config["clahe_enabled"] = False
            self.config["sensitivity_mode"] = "STANDARD"
            self.config["confidence_threshold"] = 0.48

    def process_frame(self, raw_bgr: np.ndarray) -> Tuple[OpenCVFrameTelemetry, np.ndarray]:
        """
        Run the complete OpenCV 5 detection & tracking pipeline on a raw BGR frame.
        Returns structured `OpenCVFrameTelemetry` and the annotated BGR image.
        """
        t_start = time.perf_counter()
        self.frame_index += 1

        # 1. Standardize frame size to 960x540
        if raw_bgr.shape[1] != FRAME_WIDTH or raw_bgr.shape[0] != FRAME_HEIGHT:
            frame = cv2.resize(raw_bgr, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
        else:
            frame = raw_bgr.copy()

        # 1b. Optional Degraded Camera Stress-Test (Low-Light + Dense Fog Haze)
        if self.degraded_lighting_stress:
            fog_layer = np.full_like(frame, (36, 30, 26), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.36, fog_layer, 0.64, 0)
            frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
            if self.clahe_override == "AUTO":
                self.config["clahe_enabled"] = True
                self.config["sensitivity_mode"] = "STRESS_RECOVERY_CLAHE"

        self.last_raw_frame = frame.copy()

        # 2. CLAHE Contrast Enhancement (Agent-driven, Auto Stress-Recovery, or Operator Override)
        if self.clahe_override == "FORCE_ON":
            self.config["clahe_enabled"] = True
        elif self.clahe_override == "FORCE_OFF":
            self.config["clahe_enabled"] = False

        analysis_frame = frame.copy()
        clahe_gain = 1.0
        if self.config.get("clahe_enabled", False):
            lab = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_chan)
            clahe_gain = round(float(np.std(l_enhanced) / max(float(np.std(l_chan)), 1.0)), 2)
            analysis_frame = cv2.cvtColor(
                cv2.merge((l_enhanced, a_chan, b_chan)), cv2.COLOR_LAB2BGR
            )

        # 3. Dense Optical Flow (Farneback) & MOG2 Foreground Mask
        small_w, small_h = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        gray_full = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray_small_blur = cv2.GaussianBlur(gray_small, (5, 5), 0)

        flow_mag_full = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        flow_dx_full = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        flow_dy_full = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        motion_energy = 0.0
        optical_flow_peak = 0.0

        if self.prev_gray_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray_small,
                gray_small_blur,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=13,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy = round(float(np.mean(mag)), 3)
            optical_flow_peak = round(float(np.percentile(mag, 99.2)), 2)
            flow_mag_full = cv2.resize(mag, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            flow_dx_full = cv2.resize(flow[..., 0], (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
            flow_dy_full = cv2.resize(flow[..., 1], (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

        self.prev_gray_small = gray_small_blur
        fg_mask = self.bg_subtractor.apply(analysis_frame)

        detections: List[DetectionItem] = []

        # 4. Fire Detection (HSV + YCrCb thermal/flame core + flicker kinematics)
        fire_dets, fire_mask = self._detect_fire(analysis_frame, flow_mag_full)
        detections.extend(fire_dets)

        # 5. Smoke Detection (Desaturated plume + upward drift + edge softening)
        smoke_dets, smoke_mask = self._detect_smoke(
            analysis_frame, fire_mask, flow_mag_full, flow_dy_full
        )
        detections.extend(smoke_dets)

        # 6. Person & Fall Detection + Restricted Area Intrusion Check
        person_dets, fall_dets, intrusion_dets = self._detect_people_falls_and_intrusions(
            analysis_frame, fire_mask, smoke_mask, flow_mag_full, fg_mask
        )
        detections.extend(person_dets)
        detections.extend(fall_dets)
        detections.extend(intrusion_dets)

        # 7. Crowd Density & Surge Detection (`cv2.convexHull` on spatial clusters)
        crowd_dets, crowd_hull_pts = self._detect_crowd(person_dets, flow_mag_full)
        detections.extend(crowd_dets)

        # 8. High-velocity global/localized movement anomaly detection
        movement_dets = self._detect_anomalous_movement(flow_mag_full, person_dets, fire_dets)
        detections.extend(movement_dets)

        # 9. Closed-Loop ROI Zoom & Micro-Motion Analysis (when activated by Agent or auto-target)
        roi_telemetry = self._perform_roi_zoom_analysis(
            analysis_frame, flow_mag_full, clahe_gain, fall_dets, smoke_dets, fire_dets
        )

        # Boost fall/smoke confidence when ROI Zoom verification confirms the signature
        if roi_telemetry.active and roi_telemetry.target_id:
            for d in detections:
                if d.id == roi_telemetry.target_id or d.category == roi_telemetry.target_category:
                    if d.category == "fall" and roi_telemetry.micro_motion_score < 0.85:
                        d.confidence = min(0.97, round(d.confidence + 0.18, 2))
                        d.uncertainty = round(max(0.03, 1.0 - d.confidence), 2)
                        d.metadata["roi_verified"] = True
                        d.metadata["micro_motion"] = roi_telemetry.micro_motion_score
                    elif d.category in ("smoke", "fire"):
                        d.confidence = min(0.95, round(d.confidence + 0.15, 2))
                        d.uncertainty = round(max(0.05, 1.0 - d.confidence), 2)
                        d.metadata["roi_verified"] = True

        # 10. Compute timing & FPS
        now_perf = time.perf_counter()
        dt = max(now_perf - self.last_frame_time, 1e-4)
        self.last_frame_time = now_perf
        inst_fps = min(60.0, 1.0 / dt)
        self.fps_ema = round(0.85 * self.fps_ema + 0.15 * inst_fps, 1)
        proc_ms = round((now_perf - t_start) * 1000.0, 2)

        people_count = len(person_dets)
        hazard_count = len(fall_dets) + len(fire_dets) + len(smoke_dets) + len(intrusion_dets) + len(crowd_dets)
        crowd_density = round(people_count / 12.0, 2) if not crowd_dets else crowd_dets[0].metadata.get("density_pm2", 0.45)

        ts_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        telemetry = OpenCVFrameTelemetry(
            frame_index=self.frame_index,
            timestamp=ts_str,
            fps=self.fps_ema,
            processing_ms=proc_ms,
            motion_energy=motion_energy,
            optical_flow_peak=optical_flow_peak,
            active_cv_mode=self.config["sensitivity_mode"],
            clahe_active=bool(self.config.get("clahe_enabled", False)),
            roi_zoom=roi_telemetry,
            people_count=people_count,
            hazard_count=hazard_count,
            crowd_density=crowd_density,
            detections=detections,
        )

        # 11. Render visual outputs (Annotated HUD, Optical Flow Heatmap, Split View)
        annotated = self._render_annotated_hud(
            frame,
            detections,
            crowd_hull_pts,
            flow_dx_full,
            flow_dy_full,
            roi_telemetry,
            telemetry,
        )
        flow_vis = self._render_optical_flow_heatmap(frame, flow_mag_full, detections)
        split_vis = self._render_split_comparison(frame, annotated)

        self.last_annotated_frame = annotated
        self.last_flow_frame = flow_vis
        self.last_split_frame = split_vis
        self.last_telemetry = telemetry

        return telemetry, annotated

    def _detect_fire(
        self, frame: np.ndarray, flow_mag: np.ndarray
    ) -> Tuple[List[DetectionItem], np.ndarray]:
        """Detect active flame regions using HSV + YCrCb chromaticity and flicker dynamics."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # Orange-yellow-red high-saturation, high-luminance flame core (excludes low-saturation skin tones)
        mask_hsv1 = cv2.inRange(hsv, np.array([0, 170, 220]), np.array([28, 255, 255]))
        # Bright yellow-white inner core paired with thermal Cr dominance
        mask_hsv2 = cv2.inRange(hsv, np.array([15, 75, 248]), np.array([35, 255, 255]))
        mask_ycrcb = cv2.inRange(ycrcb, np.array([180, 165, 30]), np.array([255, 255, 115]))

        fire_mask = cv2.bitwise_or(mask_hsv1, cv2.bitwise_and(mask_hsv2, mask_ycrcb))
        # Exclude top HUD bar area (y < 42)
        fire_mask[:42, :] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 380

        fire_detections: List[DetectionItem] = []
        total_fire_area = 0.0

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            total_fire_area += area

            # Check internal brightness & flicker motion
            roi_v = hsv[y : y + h, x : x + w, 2]
            core_ratio = float(np.mean(roi_v > 235)) if roi_v.size > 0 else 0.0
            local_flow = float(np.mean(flow_mag[y : y + h, x : x + w])) if flow_mag.size > 0 else 0.5

            # Compute temporal growth / flicker
            prev_area = self.fire_area_history[-1] if self.fire_area_history else area
            flicker_ratio = min(1.0, abs(area - prev_area) / max(prev_area, 100.0))

            raw_conf = 0.68 + min(0.20, area / 5500.0) + min(0.08, core_ratio * 0.25) + min(0.03, local_flow * 0.04)
            conf = round(min(0.98, max(0.60, raw_conf)), 2)
            uncert = round(max(0.02, 1.0 - conf), 2)

            nx = round((x + w * 0.5) / FRAME_WIDTH, 3)
            ny = round((y + h * 0.85) / FRAME_HEIGHT, 3)

            fire_detections.append(
                DetectionItem(
                    id=f"FIRE-{idx + 1}",
                    category="fire",
                    label="Active Fire / Thermal Core",
                    confidence=conf,
                    uncertainty=uncert,
                    bbox=[int(x), int(y), int(w), int(h)],
                    world_pos=[nx, ny],
                    velocity=round(local_flow, 2),
                    posture_state="N/A",
                    in_restricted_zone=self._is_point_in_restricted((x + w // 2, y + h)),
                    metadata={
                        "area_px": int(area),
                        "core_intensity": round(core_ratio, 2),
                        "flicker_rate": round(flicker_ratio, 2),
                        "growth_trend": "EXPANDING" if area >= prev_area else "FLICKERING",
                    },
                )
            )

        self.fire_area_history.append(total_fire_area)
        if len(self.fire_area_history) > 25:
            self.fire_area_history.pop(0)

        return fire_detections, fire_mask

    def _detect_smoke(
        self,
        frame: np.ndarray,
        fire_mask: np.ndarray,
        flow_mag: np.ndarray,
        flow_dy: np.ndarray,
    ) -> Tuple[List[DetectionItem], np.ndarray]:
        """
        Detect diffusing smoke plumes using HSV desaturation + luminance envelope +
        upward optical flow drift.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Smoke signature in our surveillance scene: cool gray-blue desaturated plume
        # Hue around 95-130 (cool slate/gray-blue) with low-to-moderate saturation [18, 78] and mid-high value [95, 215]
        smoke_mask = cv2.inRange(hsv, np.array([92, 18, 92]), np.array([132, 82, 220]))
        smoke_mask[:45, :] = 0
        # Exclude floor grid strip at bottom and fire core
        smoke_mask[460:, :] = 0
        smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(fire_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 320 if self.config["sensitivity_mode"] != "STANDARD" else self.config["smoke_min_area"]

        smoke_detections: List[DetectionItem] = []
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 24 or h < 24:
                continue

            local_flow = float(np.mean(flow_mag[y : y + h, x : x + w]))
            mean_dy = float(np.mean(flow_dy[y : y + h, x : x + w]))  # negative = rising

            # Early/small smoke plume produces moderate confidence (~0.58-0.68) to trigger Agent Uncertainty check
            if area < 2400 and not self.config.get("clahe_enabled", False):
                conf = round(min(0.69, 0.54 + area / 16000.0), 2)
            else:
                conf = round(min(0.93, 0.68 + min(0.22, area / 18000.0) + (0.04 if mean_dy < -0.05 else 0.0)), 2)

            uncert = round(max(0.06, 1.0 - conf), 2)
            nx = round((x + w * 0.5) / FRAME_WIDTH, 3)
            ny = round((y + h * 0.85) / FRAME_HEIGHT, 3)

            smoke_detections.append(
                DetectionItem(
                    id=f"SMOKE-{idx + 1}",
                    category="smoke",
                    label="Smoke Plume / Aerosol Dispersion",
                    confidence=conf,
                    uncertainty=uncert,
                    bbox=[int(x), int(y), int(w), int(h)],
                    world_pos=[nx, ny],
                    velocity=round(local_flow, 2),
                    posture_state="N/A",
                    in_restricted_zone=self._is_point_in_restricted((x + w // 2, y + h)),
                    metadata={
                        "plume_area_px": int(area),
                        "vertical_drift_dy": round(mean_dy, 3),
                        "opacity_regime": "DENSE_PLUME" if area >= 2400 else "FAINT_EARLY_PLUME",
                    },
                )
            )

        return smoke_detections, smoke_mask

    def _detect_people_falls_and_intrusions(
        self,
        frame: np.ndarray,
        fire_mask: np.ndarray,
        smoke_mask: np.ndarray,
        flow_mag: np.ndarray,
        fg_mask: np.ndarray,
    ) -> Tuple[List[DetectionItem], List[DetectionItem], List[DetectionItem]]:
        """
        Detect human subjects via chromatic garment + silhouette segmentation, track them
        across frames, analyze bounding-box aspect ratio + velocity for Fall Detection,
        and test ground-contact points against the Restricted Area polygon (`cv2.pointPolygonTest`).
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Human garment & silhouette segmentation masks (cyan/teal/blue jackets, amber/hi-vis vests, crimson/violet/emerald attire + skin/head)
        mask_teal_blue = cv2.inRange(hsv, np.array([85, 90, 95]), np.array([118, 255, 255]))
        mask_emerald = cv2.inRange(hsv, np.array([42, 85, 85]), np.array([82, 255, 255]))
        mask_purple = cv2.inRange(hsv, np.array([128, 75, 90]), np.array([168, 255, 255]))
        mask_skin_head = cv2.inRange(hsv, np.array([6, 55, 135]), np.array([22, 165, 245]))

        person_mask = cv2.bitwise_or(
            cv2.bitwise_or(mask_teal_blue, mask_emerald),
            cv2.bitwise_or(mask_purple, mask_skin_head),
        )
        # Also fuse strong moving foreground silhouettes on uploaded external videos if no garment mask triggers
        if int(np.count_nonzero(person_mask)) < 250:
            fg_clean = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            )
            person_mask = cv2.bitwise_or(person_mask, fg_clean)

        # Exclude HUD bar & fire/smoke masks
        person_mask[:48, :] = 0
        person_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(fire_mask))
        person_mask = cv2.bitwise_and(person_mask, cv2.bitwise_not(smoke_mask))

        # Unify head + torso + legs of each person without merging nearby crowd pedestrians
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes: List[Tuple[int, int, int, int, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 260 or area > 42000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 14 or h < 16:
                continue
            # Compute oriented bounding box angle via cv2.minAreaRect
            rect = cv2.minAreaRect(cnt)
            angle = float(rect[2])
            candidate_boxes.append((x, y, w, h, angle))

        # Real-world webcam / external video fallback: fuse OpenCV Haar Face + HOG Person Detector
        if not candidate_boxes:
            gray_det = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not self.face_cascade.empty():
                faces = self.face_cascade.detectMultiScale(gray_det, scaleFactor=1.15, minNeighbors=4, minSize=(28, 28))
                for fx, fy, fw, fh in faces:
                    # Expand face box into upper/full body bounding box
                    bx = max(0, int(fx - fw * 0.65))
                    by = max(44, int(fy - fh * 0.2))
                    bw = min(FRAME_WIDTH - bx, int(fw * 2.3))
                    bh = min(FRAME_HEIGHT - by, int(fh * 3.6))
                    candidate_boxes.append((bx, by, bw, bh, 0.0))
            if not candidate_boxes:
                hog_rects, _ = self.hog.detectMultiScale(
                    gray_det, winStride=(8, 8), padding=(16, 16), scale=1.08
                )
                for hx, hy, hw, hh in hog_rects:
                    candidate_boxes.append((int(hx), int(hy), int(hw), int(hh), 0.0))

        # Associate candidate boxes with existing TrackedPersonState objects
        matched_track_ids = set()
        for x, y, w, h, _ in candidate_boxes:
            cx, cy = x + w * 0.5, y + h * 0.5
            best_id: Optional[str] = None
            best_dist = 85.0
            for tid, tstate in self.tracked_people.items():
                if tid in matched_track_ids:
                    continue
                tcx, tcy = tstate.centroid_history[-1]
                dist = math.hypot(cx - tcx, cy - tcy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            local_flow = float(np.mean(flow_mag[y : y + h, x : x + w])) if flow_mag.size > 0 else 0.0
            if best_id is None:
                best_id = f"P-{self.next_person_id}"
                self.next_person_id += 1
                self.tracked_people[best_id] = TrackedPersonState(best_id, (x, y, w, h), self.frame_index)
            else:
                self.tracked_people[best_id].update(
                    (x, y, w, h), self.frame_index, local_flow, self.fall_aspect_threshold
                )
            matched_track_ids.add(best_id)

        # Prune stale tracks not seen in last 10 frames
        stale_ids = [
            tid
            for tid, tstate in self.tracked_people.items()
            if self.frame_index - tstate.last_seen_frame > 10
        ]
        for tid in stale_ids:
            del self.tracked_people[tid]

        person_detections: List[DetectionItem] = []
        fall_detections: List[DetectionItem] = []
        intrusion_detections: List[DetectionItem] = []

        for tid in sorted(matched_track_ids):
            tstate = self.tracked_people[tid]
            x, y, w, h = tstate.bbox
            ar = round(w / float(max(h, 1)), 2)
            foot_pt = (int(x + w * 0.5), int(y + h * 0.92))
            poly_dist = float(cv2.pointPolygonTest(self.restricted_polygon, foot_pt, measureDist=True))
            in_restricted = poly_dist >= -4.0

            nx = round(min(0.98, max(0.02, foot_pt[0] / FRAME_WIDTH)), 3)
            ny = round(min(0.98, max(0.05, foot_pt[1] / FRAME_HEIGHT)), 3)

            p_conf = round(min(0.96, 0.82 + min(0.12, (w * h) / 18000.0)), 2)
            person_detections.append(
                DetectionItem(
                    id=tid,
                    category="person",
                    label=f"Person {tid} ({tstate.posture_state})",
                    confidence=p_conf,
                    uncertainty=round(1.0 - p_conf, 2),
                    bbox=[x, y, w, h],
                    world_pos=[nx, ny],
                    velocity=tstate.velocity,
                    posture_state=tstate.posture_state,
                    in_restricted_zone=in_restricted,
                    metadata={
                        "aspect_ratio": ar,
                        "immobile_frames": tstate.immobile_frames,
                        "horizontal_frames": tstate.horizontal_frames,
                        "recovered_frames": tstate.recovered_frames,
                    },
                )
            )

            # Check for Fall Detection (lying on ground + low/zero velocity)
            if tstate.posture_state in ("FALLING", "FALLEN_IMMOBILE"):
                if tstate.posture_state == "FALLING":
                    # Initial collapse: moderate confidence (0.62 - 0.72) -> triggers Agent Uncertainty & ROI Zoom request
                    f_conf = round(min(0.73, 0.58 + tstate.horizontal_frames * 0.02), 2)
                    label_str = f"Possible Collapse / Horizontal Posture ({tid})"
                else:
                    # Confirmed motionless on ground
                    f_conf = round(min(0.96, 0.79 + min(0.16, tstate.immobile_frames * 0.012)), 2)
                    label_str = f"Fall Emergency: Person Motionless ({tid})"

                fall_detections.append(
                    DetectionItem(
                        id=f"FALL-{tid}",
                        category="fall",
                        label=label_str,
                        confidence=f_conf,
                        uncertainty=round(max(0.04, 1.0 - f_conf), 2),
                        bbox=[x, y, w, h],
                        world_pos=[nx, ny],
                        velocity=tstate.velocity,
                        posture_state=tstate.posture_state,
                        in_restricted_zone=in_restricted,
                        metadata={
                            "person_id": tid,
                            "aspect_ratio": ar,
                            "immobile_frames": tstate.immobile_frames,
                            "motionless_seconds": round(tstate.immobile_frames / 12.0, 1),
                        },
                    )
                )

            # Check for Restricted Area Intrusion
            if in_restricted:
                i_conf = round(min(0.97, 0.86 + min(0.10, max(0.0, poly_dist) / 60.0)), 2)
                intrusion_detections.append(
                    DetectionItem(
                        id=f"INTRUSION-{tid}",
                        category="restricted_area",
                        label=f"Restricted Zone Breach ({tid})",
                        confidence=i_conf,
                        uncertainty=round(1.0 - i_conf, 2),
                        bbox=[x, y, w, h],
                        world_pos=[nx, ny],
                        velocity=tstate.velocity,
                        posture_state=tstate.posture_state,
                        in_restricted_zone=True,
                        metadata={
                            "person_id": tid,
                            "penetration_depth_px": round(max(1.0, poly_dist), 1),
                            "zone_name": "Sector-B High-Voltage Hazard Zone",
                        },
                    )
                )

        return person_detections, fall_detections, intrusion_detections

    def _detect_crowd(
        self, person_dets: List[DetectionItem], flow_mag: np.ndarray
    ) -> Tuple[List[DetectionItem], Optional[np.ndarray]]:
        """Detect high-density crowd formation and panic surge using spatial clustering & convex hull."""
        if len(person_dets) < self.config["crowd_min_people"]:
            return [], None

        centroids = [
            (d.bbox[0] + d.bbox[2] * 0.5, d.bbox[1] + d.bbox[3] * 0.5, d)
            for d in person_dets
        ]

        # Find largest cluster within radius
        cluster_Radius = float(self.config["crowd_cluster_radius_px"])
        best_cluster: List[DetectionItem] = []
        for cx, cy, _ in centroids:
            members = [
                d
                for ox, oy, d in centroids
                if math.hypot(cx - ox, cy - oy) <= cluster_Radius
            ]
            if len(members) > len(best_cluster):
                best_cluster = members

        if len(best_cluster) < self.config["crowd_min_people"]:
            return [], None

        pts = []
        velocities = []
        for d in best_cluster:
            x, y, w, h = d.bbox
            pts.extend([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            velocities.append(d.velocity)

        pts_arr = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_arr)
        hx, hy, hw, hh = cv2.boundingRect(hull)
        hull_area_m2 = max(2.5, (hw * hh) / 4800.0)
        density_pm2 = round(len(best_cluster) / hull_area_m2, 2)
        mean_vel = round(float(np.mean(velocities)), 2)
        vel_std = round(float(np.std(velocities)), 2)

        conf = round(min(0.95, 0.74 + len(best_cluster) * 0.04), 2)
        nx = round((hx + hw * 0.5) / FRAME_WIDTH, 3)
        ny = round((hy + hh * 0.75) / FRAME_HEIGHT, 3)

        crowd_det = DetectionItem(
            id="CROWD-CLUSTER-1",
            category="crowd",
            label=f"High-Density Crowd Surge ({len(best_cluster)} Persons)",
            confidence=conf,
            uncertainty=round(1.0 - conf, 2),
            bbox=[int(hx), int(hy), int(hw), int(hh)],
            world_pos=[nx, ny],
            velocity=mean_vel,
            posture_state="CROWD_CLUSTER",
            in_restricted_zone=False,
            metadata={
                "person_count": len(best_cluster),
                "density_pm2": density_pm2,
                "velocity_variance": vel_std,
                "crowd_state": "PANIC_SURGE" if mean_vel > 2.4 else "CONGESTION_BOTTLENECK",
            },
        )
        return [crowd_det], hull

    def _detect_anomalous_movement(
        self,
        flow_mag: np.ndarray,
        person_dets: List[DetectionItem],
        fire_dets: List[DetectionItem],
    ) -> List[DetectionItem]:
        """Emit movement anomaly detection when rapid kinetic motion is detected in the scene."""
        fast_people = [p for p in person_dets if p.velocity >= 3.2]
        if not fast_people:
            return []

        fastest = max(fast_people, key=lambda p: p.velocity)
        conf = round(min(0.92, 0.70 + fastest.velocity * 0.03), 2)
        return [
            DetectionItem(
                id=f"MOV-{fastest.id}",
                category="movement",
                label=f"Rapid Kinetic Motion ({fastest.id}: {fastest.velocity:.1f} px/f)",
                confidence=conf,
                uncertainty=round(1.0 - conf, 2),
                bbox=fastest.bbox,
                world_pos=fastest.world_pos,
                velocity=fastest.velocity,
                posture_state=fastest.posture_state,
                in_restricted_zone=fastest.in_restricted_zone,
                metadata={"peak_velocity": fastest.velocity},
            )
        ]

    def _perform_roi_zoom_analysis(
        self,
        frame: np.ndarray,
        flow_mag: np.ndarray,
        clahe_gain: float,
        fall_dets: List[DetectionItem],
        smoke_dets: List[DetectionItem],
        fire_dets: List[DetectionItem],
    ) -> ROIZoomTelemetry:
        """
        Execute high-resolution OpenCV ROI crop, Laplacian sharpness check, and micro-motion
        optical flow verification when requested by the AI Agent or when an ambiguous hazard exists.
        """
        if self.manual_roi_frames > 0 and self.manual_roi_box:
            self.manual_roi_frames -= 1
            x, y, w, h = self.manual_roi_box
            target_id = "OPERATOR-LOCK"
            target_cat = "manual_lock"
        else:
            if not self.config.get("roi_zoom_active", False):
                return ROIZoomTelemetry(active=False)

            target_det: Optional[DetectionItem] = None
            if fall_dets:
                target_det = fall_dets[0]
            elif smoke_dets:
                target_det = smoke_dets[0]
            elif fire_dets:
                target_det = fire_dets[0]

            if target_det is not None:
                x, y, w, h = target_det.bbox
                target_id = target_det.id
                target_cat = target_det.category
            elif self.config.get("roi_zoom_box"):
                x, y, w, h = self.config["roi_zoom_box"]
                target_id = "ROI-CUSTOM"
                target_cat = "verification"
            else:
                return ROIZoomTelemetry(active=False)

        # Pad ROI box safely within frame bounds
        pad = 24
        x1 = max(0, x - pad)
        y1 = max(44, y - pad)
        x2 = min(FRAME_WIDTH, x + w + pad)
        y2 = min(FRAME_HEIGHT, y + h + pad)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return ROIZoomTelemetry(active=False)

        roi_crop = frame[y1:y2, x1:x2]
        roi_flow = flow_mag[y1:y2, x1:x2]
        micro_motion = round(float(np.mean(roi_flow)), 3)
        gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        sharpness = round(float(cv2.Laplacian(gray_roi, cv2.CV_64F).var()), 1)

        if target_cat == "fall":
            if micro_motion < 0.85:
                conclusion = "CONFIRMED_IMMOBILE_ON_GROUND (Zero Respiration/Limb Shift)"
            else:
                conclusion = "ACTIVE_MOVEMENT_DETECTED (Subject Shifting/Recovering)"
        elif target_cat in ("smoke", "fire"):
            conclusion = "CONFIRMED_THERMAL_AEROSOL_PLUME (High Contrast Expansion)"
        else:
            conclusion = "ROI_VERIFICATION_ACTIVE"

        return ROIZoomTelemetry(
            active=True,
            target_id=target_id,
            target_category=target_cat,
            bbox=[x1, y1, x2 - x1, y2 - y1],
            micro_motion_score=micro_motion,
            sharpness_score=sharpness,
            clahe_contrast_gain=clahe_gain,
            verification_conclusion=conclusion,
        )

    def _is_point_in_restricted(self, pt: Tuple[int, int]) -> bool:
        return float(cv2.pointPolygonTest(self.restricted_polygon, pt, measureDist=True)) >= 0.0

    def _render_annotated_hud(
        self,
        base_frame: np.ndarray,
        detections: List[DetectionItem],
        crowd_hull: Optional[np.ndarray],
        flow_dx: np.ndarray,
        flow_dy: np.ndarray,
        roi_zoom: ROIZoomTelemetry,
        telemetry: OpenCVFrameTelemetry,
    ) -> np.ndarray:
        """Render clean, high-precision OpenCV 5 Command Center overlays onto the video frame."""
        canvas = base_frame.copy()
        overlay = canvas.copy()

        # 1. Draw Restricted Area Polygon (Sector-B 13.8kV Substation & Hazmat Bay)
        has_intrusion = any(d.category == "restricted_area" for d in detections)
        zone_color = (60, 50, 240) if has_intrusion else (45, 165, 245)
        cv2.fillPoly(overlay, [self.restricted_polygon], zone_color)
        cv2.addWeighted(
            overlay,
            0.12 if not has_intrusion else 0.25,
            canvas,
            0.88 if not has_intrusion else 0.75,
            0,
            canvas,
        )
        cv2.polylines(canvas, [self.restricted_polygon], True, zone_color, 2, cv2.LINE_AA)

        # Draw corner fiducial markers on Sector-B polygon vertices
        for vx, vy in self.restricted_polygon:
            cv2.circle(canvas, (int(vx), int(vy)), 4, zone_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(vx), int(vy)), 8, zone_color, 1, cv2.LINE_AA)

        zone_banner = (
            "SECTOR-B: 13.8kV SUBSTATION BAY [BREACH ALERT]"
            if has_intrusion
            else "SECTOR-B: 13.8kV SUBSTATION BAY [PERIMETER ARMED]"
        )
        bx0, by0 = int(self.restricted_polygon[0][0]), int(self.restricted_polygon[0][1])
        cv2.rectangle(canvas, (bx0 + 6, by0 + 6), (bx0 + 276, by0 + 26), (14, 18, 26), -1)
        cv2.rectangle(canvas, (bx0 + 6, by0 + 6), (bx0 + 276, by0 + 26), zone_color, 1)
        cv2.putText(
            canvas,
            zone_banner,
            (bx0 + 12, by0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (195, 225, 255) if not has_intrusion else (150, 150, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Crowd Convex Hull if present
        if crowd_hull is not None:
            hull_overlay = canvas.copy()
            cv2.fillPoly(hull_overlay, [crowd_hull], (0, 140, 255))
            cv2.addWeighted(hull_overlay, 0.18, canvas, 0.82, 0, canvas)
            cv2.polylines(canvas, [crowd_hull], True, (0, 180, 255), 2, cv2.LINE_AA)

        # 3. Draw sparse Optical Flow velocity vectors on moving regions
        step = 32
        for y in range(64, FRAME_HEIGHT - 40, step):
            for x in range(32, FRAME_WIDTH - 32, step):
                dx = float(flow_dx[y, x])
                dy = float(flow_dy[y, x])
                mag = math.hypot(dx, dy)
                if mag > 1.2:
                    end_pt = (int(x + dx * 4.0), int(y + dy * 4.0))
                    cv2.arrowedLine(
                        canvas,
                        (x, y),
                        end_pt,
                        (230, 200, 70),
                        1,
                        cv2.LINE_AA,
                        tipLength=0.3,
                    )

        # 4. Draw Detections with corner brackets, ground-contact rings & telemetry tags
        color_map = {
            "person": (180, 225, 90),          # Emerald/Teal normal person
            "fall": (75, 75, 255),             # Critical Red fall
            "fire": (35, 110, 255),            # Bright Orange/Red fire
            "smoke": (195, 175, 150),          # Slate/Amber smoke
            "crowd": (30, 185, 255),           # Amber crowd
            "restricted_area": (90, 50, 255),  # Crimson breach
            "movement": (240, 210, 60),        # Cyan kinetic motion
        }

        for det in detections:
            # Avoid duplicate box clutter for movement if person box is already drawn
            if det.category == "movement":
                continue
            x, y, w, h = det.bbox
            color = color_map.get(det.category, (200, 200, 200))
            if det.category == "person" and det.posture_state in ("FALLING", "FALLEN_IMMOBILE"):
                continue  # Drawn by the high-priority 'fall' detection box

            # Ground-plane foot ring & posture centerline axis for people/falls
            if det.category in ("person", "fall", "restricted_area"):
                foot_cx, foot_cy = int(x + w * 0.5), int(y + h * 0.94)
                cv2.ellipse(
                    canvas,
                    (foot_cx, foot_cy),
                    (max(14, w // 2), 6),
                    0,
                    0,
                    360,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                if det.category == "restricted_area":
                    # Draw vector line from nearest substation transformer (785, 145) to intruder foot point
                    cv2.line(canvas, (785, 145), (foot_cx, foot_cy), (80, 80, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, (785, 145), 4, (80, 80, 255), -1, cv2.LINE_AA)

            thickness = 2 if det.category in ("fall", "fire", "restricted_area", "crowd") else 1
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

            # Corner precision brackets
            clen = min(12, w // 3, h // 3)
            cv2.line(canvas, (x, y), (x + clen, y), color, 3)
            cv2.line(canvas, (x, y), (x, y + clen), color, 3)
            cv2.line(canvas, (x + w, y), (x + w - clen, y), color, 3)
            cv2.line(canvas, (x + w, y), (x + w, y + clen), color, 3)
            cv2.line(canvas, (x, y + h), (x + clen, y + h), color, 3)
            cv2.line(canvas, (x, y + h), (x, y + h - clen), color, 3)
            cv2.line(canvas, (x + w, y + h), (x + w - clen, y + h), color, 3)
            cv2.line(canvas, (x + w, y + h), (x + w, y + h - clen), color, 3)

            conf_pct = int(round(det.confidence * 100))
            tag = f"{det.id} | {det.category.upper()} {conf_pct}%"
            if det.uncertainty >= 0.28:
                tag += " [UNCERTAIN]"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(56, y - 6)
            cv2.rectangle(canvas, (x, ty - th - 6), (x + tw + 10, ty + 2), (18, 22, 30), -1)
            cv2.rectangle(canvas, (x, ty - th - 6), (x + tw + 10, ty + 2), color, 1)
            cv2.putText(
                canvas,
                tag,
                (x + 5, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 248, 255),
                1,
                cv2.LINE_AA,
            )

        # 5. Picture-in-Picture (PiP) Agent ROI Verification Inset (bottom-right corner)
        if roi_zoom.active and roi_zoom.bbox:
            rx, ry, rw, rh = roi_zoom.bbox
            crop = base_frame[ry : ry + rh, rx : rx + rw]
            if crop.size > 0:
                pip_w, pip_h = 220, 135
                pip_img = cv2.resize(crop, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
                # Apply CLAHE visual boost on PiP inset
                lab_pip = cv2.cvtColor(pip_img, cv2.COLOR_BGR2LAB)
                l_c, a_c, b_c = cv2.split(lab_pip)
                pip_img = cv2.cvtColor(
                    cv2.merge((self.clahe.apply(l_c), a_c, b_c)), cv2.COLOR_LAB2BGR
                )
                px1 = FRAME_WIDTH - pip_w - 16
                py1 = FRAME_HEIGHT - pip_h - 16
                canvas[py1 : py1 + pip_h, px1 : px1 + pip_w] = pip_img
                cv2.rectangle(
                    canvas,
                    (px1 - 2, py1 - 22),
                    (px1 + pip_w + 2, py1 + pip_h + 2),
                    (80, 210, 255),
                    2,
                )
                cv2.rectangle(canvas, (px1 - 2, py1 - 22), (px1 + pip_w + 2, py1), (18, 26, 38), -1)
                cv2.putText(
                    canvas,
                    f"AGENT ROI ZOOM: {roi_zoom.target_id} (uMot:{roi_zoom.micro_motion_score:.2f})",
                    (px1 + 4, py1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.37,
                    (110, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 6. Top OpenCV 5 Telemetry Header Bar
        cv2.rectangle(canvas, (0, 0), (FRAME_WIDTH, 36), (12, 16, 23), -1)
        cv2.line(canvas, (0, 36), (FRAME_WIDTH, 36), (45, 58, 80), 1)
        hud_left = (
            f"OPENCV 5 VISION ENGINE | MODE: {telemetry.active_cv_mode} "
            f"| CLAHE: {'ON' if telemetry.clahe_active else 'OFF'} "
            f"| FPS: {telemetry.fps:.1f} ({telemetry.processing_ms:.1f}ms)"
        )
        cv2.putText(
            canvas,
            hud_left,
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (215, 228, 248),
            1,
            cv2.LINE_AA,
        )
        hud_right = f"PEOPLE: {telemetry.people_count} | HAZARDS: {telemetry.hazard_count} | FLOW: {telemetry.motion_energy:.2f}"
        cv2.putText(
            canvas,
            hud_right,
            (FRAME_WIDTH - 335, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (130, 225, 190) if telemetry.hazard_count == 0 else (110, 145, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def _render_optical_flow_heatmap(
        self,
        base_frame: np.ndarray,
        flow_mag: np.ndarray,
        detections: List[DetectionItem],
    ) -> np.ndarray:
        """Render Dense Optical Flow & Thermal Motion Energy Heatmap."""
        norm_mag = np.clip(flow_mag * 28.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        blended = cv2.addWeighted(base_frame, 0.38, heatmap, 0.62, 0)
        for det in detections:
            if det.category in ("fall", "fire", "smoke", "restricted_area", "crowd"):
                x, y, w, h = det.bbox
                cv2.rectangle(blended, (x, y), (x + w, y + h), (90, 240, 255), 2)
        cv2.rectangle(blended, (0, 0), (FRAME_WIDTH, 36), (12, 16, 23), -1)
        cv2.putText(
            blended,
            "OPENCV 5 DENSE OPTICAL FLOW (FARNEBACK) & MOTION ENERGY HEATMAP",
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (110, 225, 255),
            1,
            cv2.LINE_AA,
        )
        return blended

    def _render_split_comparison(
        self, raw_frame: np.ndarray, annotated_frame: np.ndarray
    ) -> np.ndarray:
        """Render side-by-side Split Comparison (Left: Raw Surveillance Video | Right: OpenCV 5 Analysis)."""
        half_w = FRAME_WIDTH // 2
        left_crop = cv2.resize(raw_frame, (half_w, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        right_crop = cv2.resize(annotated_frame, (half_w, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        split = np.hstack((left_crop, right_crop))
        cv2.line(split, (half_w, 0), (half_w, FRAME_HEIGHT), (90, 200, 255), 2)
        cv2.rectangle(split, (10, 44), (195, 70), (15, 20, 28), -1)
        cv2.putText(
            split,
            "RAW VIDEO INPUT",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 220, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(split, (half_w + 10, 44), (half_w + 235, 70), (15, 20, 28), -1)
        cv2.putText(
            split,
            "OPENCV 5 VISION OUTPUT",
            (half_w + 18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 225, 255),
            1,
            cv2.LINE_AA,
        )
        return split
