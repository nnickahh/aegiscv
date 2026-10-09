"""
Realistic Surveillance Scene Synthesizer & Sample MP4 Generator for OpenCV 5 Analysis.

Renders lifelike 3D-perspective CCTV camera frames of an industrial transit concourse and
Sector-B High-Voltage Substation Bay (without any detection overlays) that are then fed into
`OpenCV5VisionPipeline.process_frame()` so all detections (people, falls, fires, smoke, crowds,
movement, and restricted area breaches) are computed genuinely from pixel data by OpenCV.
"""
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    RESTRICTED_ZONE_POLYGON,
    SAMPLE_VIDEOS_DIR,
)


SCENARIO_CATALOG: Dict[str, Dict[str, str]] = {
    "fall_emergency": {
        "id": "fall_emergency",
        "name": "Scenario 1: Medical Fall & Immobility",
        "short_name": "Fall Emergency",
        "description": "Subject P-101 walks across Sector-A Concourse, stumbles and collapses onto the floor, remaining motionless. Triggers OpenCV aspect-ratio + zero-velocity detection and Agent ROI micro-motion verification.",
        "badge": "MEDICAL / FALL",
    },
    "fall_recovery": {
        "id": "fall_recovery",
        "name": "Scenario 2: Fall → Self-Recovery (Dynamic Decision Change)",
        "short_name": "Adaptive Recovery",
        "description": "Subject collapses briefly (Agent flags emergency & zooms ROI), then stands back up and walks away. Demonstrates the AI Agent dynamically changing its decision and cancelling dispatch.",
        "badge": "ADAPTIVE AGENT",
    },
    "fire_smoke": {
        "id": "fire_smoke",
        "name": "Scenario 3: Early Smoke Plume → Fire Outbreak",
        "short_name": "Fire & Smoke",
        "description": "Faint aerosol smoke plume emerges near the equipment bay first (low confidence triggers Agent CLAHE boost), followed by rapid thermal flame ignition and expansion.",
        "badge": "THERMAL / FIRE",
    },
    "crowd_surge": {
        "id": "crowd_surge",
        "name": "Scenario 4: Crowd Convergence & Bottleneck Surge",
        "short_name": "Crowd Surge",
        "description": "Six subjects converge rapidly into a dense bottleneck near Concourse Exit-1, exceeding safe crowd density thresholds.",
        "badge": "CROWD DENSITY",
    },
    "restricted_intrusion": {
        "id": "restricted_intrusion",
        "name": "Scenario 5: Restricted Hazard Zone Intrusion",
        "short_name": "Zone Breach",
        "description": "Unauthorized subject crosses the OSHA hazard perimeter into Sector-B 13.8kV High-Voltage Substation Bay (`cv2.pointPolygonTest`).",
        "badge": "PERIMETER BREACH",
    },
    "multi_crisis": {
        "id": "multi_crisis",
        "name": "Scenario 6: Multi-Emergency Compound Crisis",
        "short_name": "Multi-Crisis",
        "description": "Simultaneous transformer fire/smoke ignition in Sector-B, subject collapse from smoke inhalation near the perimeter, unauthorized zone entry, and bystander evacuation.",
        "badge": "COMPOUND CRISIS",
    },
    "normal_patrol": {
        "id": "normal_patrol",
        "name": "Baseline: Nominal Facility Monitoring",
        "short_name": "Nominal Patrol",
        "description": "Routine pedestrian flow in safe concourse lanes with zero active hazards.",
        "badge": "NOMINAL",
    },
}


class SurveillanceSceneSynthesizer:
    """Generates realistic raw BGR CCTV frames for interactive simulation and MP4 export."""

    def __init__(self) -> None:
        self.current_scenario: str = "fall_emergency"
        self.scenario_tick: int = 0
        self._bg_cache: Optional[np.ndarray] = None

    def set_scenario(self, scenario_id: str) -> None:
        if scenario_id in SCENARIO_CATALOG:
            self.current_scenario = scenario_id
            self.scenario_tick = 0

    def _build_static_background(self) -> np.ndarray:
        """
        Render a realistic 3D-perspective industrial transit concourse & Sector-B
        High-Voltage Substation Bay. Uses strictly neutral/low-saturation architectural
        tones (S < 12 for V >= 85) so background elements never trigger chromatic hazard masks.
        """
        if self._bg_cache is not None:
            return self._bg_cache.copy()

        bg = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        # 1. Vertical perspective floor luminance gradient (darker in far distance, lighter mid-floor)
        for y in range(FRAME_HEIGHT):
            if y < 92:
                # Upper industrial wall
                v = int(24 + (y / 92.0) * 12)
                bg[y, :] = (v, v, v)
            else:
                # Polished concrete floor gradient
                norm_y = (y - 92) / float(FRAME_HEIGHT - 92)
                v = int(34 + math.sin(norm_y * math.pi * 0.85) * 16)
                bg[y, :] = (v, v, v)

        # 2. Overhead ceiling light fixtures & soft floor reflection pools
        light_xs = [190, 480, 770]
        for lx in light_xs:
            # Ceiling light bar
            cv2.rectangle(bg, (lx - 38, 12), (lx + 38, 20), (82, 82, 82), -1)
            cv2.rectangle(bg, (lx - 32, 14), (lx + 32, 18), (115, 115, 115), -1)

        # 3. Back Wall Architectural Columns, Transit Doors & Exit Signage
        cv2.line(bg, (0, 92), (FRAME_WIDTH, 92), (62, 62, 62), 2)
        pillar_xs = [85, 340, 595]
        for px in pillar_xs:
            cv2.rectangle(bg, (px - 18, 0), (px + 18, 96), (29, 29, 29), -1)
            cv2.rectangle(bg, (px - 18, 0), (px - 12, 96), (44, 44, 44), -1)
            cv2.line(bg, (px + 18, 0), (px + 18, 96), (18, 18, 18), 2)

        # Emergency Egress Double Doors on Back Wall (Left-Center)
        cv2.rectangle(bg, (165, 28), (265, 92), (22, 22, 22), -1)
        cv2.rectangle(bg, (165, 28), (265, 92), (55, 55, 55), 1)
        cv2.line(bg, (215, 28), (215, 92), (48, 48, 48), 2)
        # Glass door window panels
        cv2.rectangle(bg, (176, 38), (206, 68), (38, 38, 38), -1)
        cv2.rectangle(bg, (224, 38), (254, 68), (38, 38, 38), -1)
        cv2.putText(
            bg,
            "EXIT-1 CONCOURSE",
            (172, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (84, 84, 84),
            1,
            cv2.LINE_AA,
        )

        # 4. Perspective Concrete Floor Expansion Joints & Tiles
        for y in range(118, FRAME_HEIGHT, 44):
            cv2.line(bg, (0, y), (FRAME_WIDTH, y), (43, 43, 43), 1)
        for x in range(-120, FRAME_WIDTH + 180, 90):
            pt_top = (int(FRAME_WIDTH * 0.5 + (x - FRAME_WIDTH * 0.5) * 0.62), 92)
            pt_bot = (x, FRAME_HEIGHT)
            cv2.line(bg, pt_top, pt_bot, (42, 42, 42), 1)

        # 5. Sector-A Pedestrian Walkway Lane Markings
        cv2.line(bg, (40, 305), (585, 305), (58, 58, 58), 2)
        cv2.line(bg, (40, 472), (920, 472), (58, 58, 58), 2)
        # Dashed center lane divider
        for dx in range(60, 560, 48):
            cv2.line(bg, (dx, 388), (dx + 24, 388), (50, 50, 50), 1)
        cv2.putText(
            bg,
            "SECTOR-A PEDESTRIAN TRANSIT CORRIDOR",
            (64, 492),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (74, 74, 74),
            1,
            cv2.LINE_AA,
        )

        # 6. Realistic Sector-B Restricted Hazard Zone (13.8kV High-Voltage Substation & Hazmat Bay)
        # Polygon: [(640, 95), (920, 95), (940, 315), (610, 315)]
        poly = np.array(RESTRICTED_ZONE_POLYGON, dtype=np.int32)

        # Dark insulated substation floor pad inside Sector-B
        cv2.fillPoly(bg, [poly], (28, 29, 30))

        # Diagonal OSHA hazard warning stripes along bottom & left perimeter of Sector-B
        for sx in range(614, 932, 18):
            cv2.line(bg, (sx, 315), (sx + 8, 305), (68, 68, 68), 3)
        for sy in range(105, 310, 18):
            lx = int(640 - ((sy - 95) / 220.0) * 30)
            cv2.line(bg, (lx, sy), (lx + 10, sy - 6), (68, 68, 68), 3)

        cv2.polylines(bg, [poly], True, (76, 76, 76), 2, cv2.LINE_AA)

        # Physical High-Voltage Transformer & Switchgear Cabinets inside Sector-B (top right)
        cabinet_boxes = [
            (665, 58, 72, 92, "HV-TX1"),
            (752, 58, 72, 92, "HV-TX2"),
            (839, 58, 68, 92, "SW-BUS"),
        ]
        for cx, cy, cw, ch, label in cabinet_boxes:
            # 3D side depth shadow
            cv2.rectangle(bg, (cx - 6, cy + 4), (cx, cy + ch), (18, 18, 18), -1)
            # Main metallic cabinet body
            cv2.rectangle(bg, (cx, cy), (cx + cw, cy + ch), (40, 41, 42), -1)
            cv2.rectangle(bg, (cx, cy), (cx + cw, cy + ch), (66, 66, 66), 1)
            # Cooling louver vents
            for vy in range(cy + 16, cy + 52, 6):
                cv2.line(bg, (cx + 10, vy), (cx + cw - 10, vy), (25, 25, 25), 2)
            # High-voltage stencil label
            cv2.putText(
                bg,
                label,
                (cx + 10, cy + 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (85, 85, 85),
                1,
                cv2.LINE_AA,
            )

        # Perimeter Safety Bollards (vertical posts at the corners of Sector-B)
        bollards = [(610, 315), (775, 315), (940, 315), (625, 205), (640, 95)]
        for bx, by in bollards:
            cv2.ellipse(bg, (bx, by), (6, 3), 0, 0, 360, (16, 16, 16), -1)
            cv2.rectangle(bg, (bx - 4, by - 32), (bx + 4, by), (58, 58, 58), -1)
            cv2.rectangle(bg, (bx - 4, by - 28), (bx + 4, by - 22), (82, 82, 82), -1)

        cv2.putText(
            bg,
            "DANGER: 13.8kV SUBSTATION BAY [AUTHORIZED ONLY]",
            (628, 296),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (82, 82, 82),
            1,
            cv2.LINE_AA,
        )

        self._bg_cache = bg
        return bg.copy()

    @staticmethod
    def _draw_human_subject(
        canvas: np.ndarray,
        center_x: int,
        foot_y: int,
        garment_bgr: Tuple[int, int, int],
        fallen_progress: float = 0.0,
        walk_phase: float = 0.0,
    ) -> None:
        """
        Render an articulated, shaded human subject on the surveillance floor.
        - `fallen_progress == 0.0`: Upright walking posture with animated stride (width ~36, height ~88 -> AR ~0.41)
        - `fallen_progress == 1.0`: Sprawled horizontal posture on floor (width ~96, height ~32 -> AR ~3.00)
        """
        skin_bgr = (162, 188, 224)  # Natural skin tone within mask_skin_head HSV
        hair_bgr = (32, 34, 38)
        pants_bgr = (
            max(35, int(garment_bgr[0] * 0.78)),
            max(35, int(garment_bgr[1] * 0.78)),
            max(35, int(garment_bgr[2] * 0.78)),
        )
        highlight_bgr = (
            min(255, int(garment_bgr[0] * 1.12)),
            min(255, int(garment_bgr[1] * 1.12)),
            min(255, int(garment_bgr[2] * 1.12)),
        )

        if fallen_progress < 0.5:
            # Animated walking leg swing (-5..+5 px)
            stride = int(math.sin(walk_phase) * 5)
            arm_swing = int(math.cos(walk_phase) * 4)

            # Ground contact shadow
            cv2.ellipse(canvas, (center_x, foot_y + 2), (20, 6), 0, 0, 360, (20, 20, 22), -1)

            # Left & Right Articulated Legs + Shoes
            l_foot_y = foot_y + stride // 2
            r_foot_y = foot_y - stride // 2
            cv2.line(canvas, (center_x - 6, foot_y - 35), (center_x - 7 - stride // 2, l_foot_y), pants_bgr, 7, cv2.LINE_AA)
            cv2.line(canvas, (center_x + 6, foot_y - 35), (center_x + 7 + stride // 2, r_foot_y), pants_bgr, 7, cv2.LINE_AA)
            # Shoes
            cv2.ellipse(canvas, (center_x - 7 - stride // 2, l_foot_y), (6, 3), 0, 0, 360, (28, 30, 32), -1)
            cv2.ellipse(canvas, (center_x + 7 + stride // 2, r_foot_y), (6, 3), 0, 0, 360, (28, 30, 32), -1)

            # Torso Jacket with rounded shoulders & center zipper/shading
            torso_pts = np.array(
                [
                    [center_x - 15, foot_y - 68],
                    [center_x + 15, foot_y - 68],
                    [center_x + 13, foot_y - 33],
                    [center_x - 13, foot_y - 33],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(canvas, [torso_pts], garment_bgr)
            # Shoulder highlight & jacket seam
            cv2.line(canvas, (center_x - 13, foot_y - 67), (center_x + 13, foot_y - 67), highlight_bgr, 2, cv2.LINE_AA)
            cv2.line(canvas, (center_x, foot_y - 67), (center_x, foot_y - 34), pants_bgr, 1, cv2.LINE_AA)

            # Articulated Arms + Hands
            cv2.line(
                canvas,
                (center_x - 15, foot_y - 64),
                (center_x - 17, foot_y - 42 + arm_swing),
                garment_bgr,
                5,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (center_x + 15, foot_y - 64),
                (center_x + 17, foot_y - 42 - arm_swing),
                garment_bgr,
                5,
                cv2.LINE_AA,
            )
            cv2.circle(canvas, (center_x - 17, foot_y - 40 + arm_swing), 3, skin_bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, (center_x + 17, foot_y - 40 - arm_swing), 3, skin_bgr, -1, cv2.LINE_AA)

            # Neck & Head with Hair Crown
            cv2.rectangle(canvas, (center_x - 3, foot_y - 73), (center_x + 3, foot_y - 68), skin_bgr, -1)
            cv2.circle(canvas, (center_x, foot_y - 79), 9, skin_bgr, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (center_x, foot_y - 82), (9, 6), 0, 180, 360, hair_bgr, -1, cv2.LINE_AA)
        else:
            # Sprawled horizontal subject lying motionless on concourse floor
            y_mid = foot_y - 14
            # Floor shadow under fallen subject
            cv2.ellipse(canvas, (center_x, y_mid + 8), (50, 11), 0, 0, 360, (19, 20, 22), -1)

            # Sprawled legs (right side of fallen figure)
            cv2.line(canvas, (center_x + 6, y_mid - 3), (center_x + 42, y_mid - 6), pants_bgr, 8, cv2.LINE_AA)
            cv2.line(canvas, (center_x + 6, y_mid + 5), (center_x + 44, y_mid + 8), pants_bgr, 8, cv2.LINE_AA)

            # Horizontal torso jacket
            torso_horiz = np.array(
                [
                    [center_x - 38, y_mid - 13],
                    [center_x + 12, y_mid - 12],
                    [center_x + 12, y_mid + 12],
                    [center_x - 38, y_mid + 13],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(canvas, [torso_horiz], garment_bgr)
            cv2.line(canvas, (center_x - 36, y_mid - 11), (center_x + 10, y_mid - 10), highlight_bgr, 2, cv2.LINE_AA)

            # Outstretched arm on floor
            cv2.line(canvas, (center_x - 28, y_mid + 10), (center_x - 42, y_mid + 16), garment_bgr, 5, cv2.LINE_AA)
            cv2.circle(canvas, (center_x - 44, y_mid + 16), 3, skin_bgr, -1, cv2.LINE_AA)

            # Head resting near floor on left side
            cv2.circle(canvas, (center_x - 46, y_mid - 1), 9, skin_bgr, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (center_x - 48, y_mid - 3), (6, 8), 0, 90, 270, hair_bgr, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_smoke_plume(
        canvas: np.ndarray,
        origin_x: int,
        origin_y: int,
        intensity: float,
        tick: int,
    ) -> None:
        """
        Render volumetric rising smoke plume in HSV range [95..125, 25..70, 105..205]
        so OpenCV's `_detect_smoke` segments the rising aerosol contour accurately.
        """
        overlay = canvas.copy()
        num_puffs = int(6 + intensity * 8)
        for i in range(num_puffs):
            drift_y = int((i * 15 + (tick * 2) % 18) * (0.72 + 0.38 * intensity))
            drift_x = int(math.sin(tick * 0.11 + i * 0.85) * (9 + i * 3.2))
            px = origin_x + drift_x
            py = max(56, origin_y - drift_y)
            rx = int((20 + i * 7.5) * intensity)
            ry = int((15 + i * 5.5) * intensity)
            # Layered slate-blue aerosol plume BGR ~(168, 146, 128) -> HSV ~(106, 60, 168)
            smoke_color = (168, 146, 128) if i % 2 == 0 else (158, 138, 122)
            cv2.ellipse(
                overlay,
                (px, py),
                (max(12, rx), max(10, ry)),
                int(math.sin(i + tick * 0.05) * 20),
                0,
                360,
                smoke_color,
                -1,
                cv2.LINE_AA,
            )

        alpha = min(0.76, max(0.46, intensity * 0.70))
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    @staticmethod
    def _draw_fire_flames(
        canvas: np.ndarray,
        origin_x: int,
        origin_y: int,
        scale: float,
        tick: int,
    ) -> None:
        """
        Render volumetric multi-layered thermal flame core with floor glow and rising embers
        in HSV [0..28, 170..255, 220..255] so OpenCV's `_detect_fire` detects the flame geometry.
        """
        # Warm thermal glow cast on concrete floor around base of fire
        glow_overlay = canvas.copy()
        cv2.ellipse(
            glow_overlay,
            (origin_x, origin_y + 6),
            (int(44 * scale), int(16 * scale)),
            0,
            0,
            360,
            (18, 118, 250),
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(glow_overlay, 0.32, canvas, 0.68, 0, canvas)

        num_tongues = 7
        for i in range(num_tongues):
            phase = tick * 0.36 + i * 1.05
            dx = int(math.sin(phase) * 13 * scale + (i - 3.0) * 8 * scale)
            dy = int(abs(math.cos(phase * 1.35)) * 24 * scale)
            rx = max(10, int(17 * scale))
            ry = max(14, int((28 + (i % 3) * 9) * scale))
            # Outer orange-red flame BGR (16, 115, 250) -> HSV (12, 239, 250)
            cv2.ellipse(
                canvas,
                (origin_x + dx, origin_y - dy),
                (rx, ry),
                int(math.sin(phase) * 12),
                0,
                360,
                (16, 115, 250),
                -1,
                cv2.LINE_AA,
            )
            # Inner bright yellow-orange flame BGR (28, 198, 255) -> HSV (22, 227, 255)
            cv2.ellipse(
                canvas,
                (origin_x + dx // 2, origin_y - dy // 2),
                (max(6, int(rx * 0.65)), max(8, int(ry * 0.65))),
                0,
                0,
                360,
                (28, 198, 255),
                -1,
                cv2.LINE_AA,
            )

        # White-hot combustion core
        cv2.ellipse(
            canvas,
            (origin_x, origin_y - int(8 * scale)),
            (max(8, int(15 * scale)), max(8, int(12 * scale))),
            0,
            0,
            360,
            (82, 240, 255),
            -1,
            cv2.LINE_AA,
        )

    def generate_next_raw_frame(self) -> np.ndarray:
        """Render the next raw surveillance frame for the active scenario."""
        self.scenario_tick += 1
        t = self.scenario_tick
        frame = self._build_static_background()

        # Garment color palette (all within OpenCV person segmentation HSV ranges)
        teal_jacket = (215, 170, 35)     # HSV ~(102, 213, 215)
        emerald_vest = (105, 195, 45)    # HSV ~(68, 196, 195)
        purple_coat = (205, 65, 175)     # HSV ~(156, 174, 205)

        scen = self.current_scenario

        if scen == "fall_emergency":
            # Loop every 160 ticks: walk for first 20 ticks, then collapse and remain motionless
            cycle = t % 160
            if cycle < 20:
                px = 260 + cycle * 4
                py = 342
                self._draw_human_subject(
                    frame, px, py, teal_jacket, fallen_progress=0.0, walk_phase=t * 0.45
                )
            else:
                # Motionless fallen subject on Concourse floor
                self._draw_human_subject(frame, 340, 345, teal_jacket, fallen_progress=1.0)

            # Second person walking normally along lower concourse lane
            p2_x = 760 - (t * 2) % 520
            self._draw_human_subject(
                frame, p2_x, 455, emerald_vest, fallen_progress=0.0, walk_phase=t * 0.35
            )

        elif scen == "fall_recovery":
            # Walk (0-18) -> Collapse & lie still (19-62) -> Stand up & walk away (63-130)
            cycle = t % 130
            if cycle < 18:
                px = 250 + cycle * 4
                self._draw_human_subject(
                    frame, px, 342, teal_jacket, fallen_progress=0.0, walk_phase=t * 0.45
                )
            elif cycle < 62:
                self._draw_human_subject(frame, 325, 345, teal_jacket, fallen_progress=1.0)
            else:
                px = 325 + (cycle - 62) * 5
                self._draw_human_subject(
                    frame, min(880, px), 342, teal_jacket, fallen_progress=0.0, walk_phase=t * 0.48
                )

        elif scen == "fire_smoke":
            cycle = t % 160
            # Early faint smoke first (0-15), then denser smoke + growing fire (16+)
            smoke_intensity = min(1.15, 0.45 + cycle * 0.015)
            self._draw_smoke_plume(frame, 520, 290, smoke_intensity, t)
            if cycle >= 16:
                fire_scale = min(1.55, 0.65 + (cycle - 16) * 0.018)
                self._draw_fire_flames(frame, 520, 325, fire_scale, t)

            # Evacuating bystander moving away from hazard toward Exit-1
            evac_x = max(110, 410 - (cycle * 3))
            self._draw_human_subject(
                frame, evac_x, 385, emerald_vest, fallen_progress=0.0, walk_phase=t * 0.55
            )

        elif scen == "crowd_surge":
            # 6 people converging into a tight cluster around (460, 345) with non-overlapping silhouettes
            base_cx, base_cy = 460, 345
            offsets = [
                (-130, -18, teal_jacket),
                (-78, 22, emerald_vest),
                (-26, -20, purple_coat),
                (26, 20, teal_jacket),
                (78, -18, emerald_vest),
                (130, 22, purple_coat),
            ]
            for idx, (ox, oy, col) in enumerate(offsets):
                jitter_x = int(math.sin(t * 0.28 + idx) * 4)
                jitter_y = int(math.cos(t * 0.24 + idx * 1.3) * 4)
                self._draw_human_subject(
                    frame,
                    base_cx + ox + jitter_x,
                    base_cy + oy + jitter_y,
                    col,
                    fallen_progress=0.0,
                    walk_phase=t * 0.5 + idx,
                )

        elif scen == "restricted_intrusion":
            # Subject walks directly across the OSHA hazard boundary into Sector-B Substation Bay
            cycle = t % 120
            intruder_x = min(820, 600 + cycle * 3)
            intruder_y = 248
            self._draw_human_subject(
                frame, intruder_x, intruder_y, purple_coat, fallen_progress=0.0, walk_phase=t * 0.42
            )
            # Normal worker outside restricted area in Sector-A
            worker_x = 220 + int(math.sin(t * 0.06) * 80)
            self._draw_human_subject(
                frame, worker_x, 415, teal_jacket, fallen_progress=0.0, walk_phase=t * 0.30
            )

        elif scen == "multi_crisis":
            # 1. Active Transformer Fire & Smoke inside Sector-B Substation Bay
            self._draw_smoke_plume(frame, 690, 235, 1.05, t)
            self._draw_fire_flames(frame, 690, 268, 1.25, t)
            # 2. Person collapsed motionless near the hazard zone perimeter
            self._draw_human_subject(frame, 535, 345, teal_jacket, fallen_progress=1.0)
            # 3. Unauthorized technician/intruder inside Sector-B restricted zone
            self._draw_human_subject(
                frame, 795, 228, purple_coat, fallen_progress=0.0, walk_phase=t * 0.40
            )
            # 4. Evacuating bystander moving rapidly on left toward Exit-1
            p3_x = max(95, 340 - (t * 4) % 230)
            self._draw_human_subject(
                frame, p3_x, 430, emerald_vest, fallen_progress=0.0, walk_phase=t * 0.60
            )

        else:  # normal_patrol
            p1_x = 180 + (t * 2) % 620
            p2_x = 780 - (t * 2) % 580
            self._draw_human_subject(
                frame, p1_x, 345, teal_jacket, fallen_progress=0.0, walk_phase=t * 0.35
            )
            self._draw_human_subject(
                frame, p2_x, 445, emerald_vest, fallen_progress=0.0, walk_phase=t * 0.35
            )

        # Realistic CCTV OSD Timestamp & Camera Metadata Bar at bottom-left
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.rectangle(frame, (10, FRAME_HEIGHT - 28), (370, FRAME_HEIGHT - 8), (18, 18, 18), -1)
        cv2.putText(
            frame,
            f"CAM-04 CONCOURSE-NORTH [1080p24] | {ts}",
            (16, FRAME_HEIGHT - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (130, 130, 130),
            1,
            cv2.LINE_AA,
        )

        return frame


def generate_sample_mp4_files(force_regenerate: bool = False) -> List[Path]:
    """
    Generate real `.mp4` emergency video files on disk using `cv2.VideoWriter`
    so judges can also test uploading video files directly from `data/sample_videos/`.
    """
    synth = SurveillanceSceneSynthesizer()
    demos = [
        ("fall_emergency_sample.mp4", "fall_emergency", 72),
        ("fire_and_smoke_sample.mp4", "fire_smoke", 72),
        ("multi_crisis_sample.mp4", "multi_crisis", 72),
    ]
    created_paths: List[Path] = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for filename, scenario_id, num_frames in demos:
        out_path = SAMPLE_VIDEOS_DIR / filename
        if not force_regenerate and out_path.exists() and out_path.stat().st_size > 10000:
            created_paths.append(out_path)
            continue

        synth.set_scenario(scenario_id)
        writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
        if writer.isOpened():
            for _ in range(num_frames):
                raw_frame = synth.generate_next_raw_frame()
                writer.write(raw_frame)
            writer.release()
            created_paths.append(out_path)

    return created_paths
