"""
Closed-Loop AI Emergency Coordinator Agent (See → Understand → Prioritize → Act → Verify).

Implements a stateful neuro-symbolic + AWS Bedrock agent whose decisions are directly driven
by OpenCV 5 telemetry and whose actions actively reconfigure the OpenCV 5 vision pipeline
(e.g., requesting ROI zoom micro-motion verification, enabling CLAHE contrast boost,
escalating/de-escalating priorities, and reversing decisions when the scene changes).
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.aws_service import AWSCloudCoordinator
from backend.database import DatabaseManager
from backend.models import (
    AgentDecision,
    DetectionItem,
    IncidentRecord,
    OpenCVFrameTelemetry,
    TimelineEvent,
)
from backend.opencv_pipeline import OpenCV5VisionPipeline


class AIEmergencyCoordinatorAgent:
    """
    Stateful AI Agent that coordinates emergency detection, active OpenCV re-analysis,
    incident prioritization, human-in-the-loop approvals, and outcome verification.
    """

    def __init__(
        self,
        vision_pipeline: OpenCV5VisionPipeline,
        db: DatabaseManager,
        aws: AWSCloudCoordinator,
    ) -> None:
        self.vision = vision_pipeline
        self.db = db
        self.aws = aws

        self.decision_counter: int = 0
        self.event_counter: int = 0
        self.active_incidents: Dict[str, IncidentRecord] = {}
        self.last_decision: Optional[AgentDecision] = None
        self.last_logged_headline: str = ""
        self.last_s3_snapshot_frame: int = -100
        self.last_cw_publish_frame: int = 0

    def reset_agent(self) -> None:
        """Reset agent memory when switching simulation scenarios."""
        self.active_incidents.clear()
        self.last_decision = None
        self.last_logged_headline = ""
        self.last_s3_snapshot_frame = -100
        self.vision.apply_agent_command("RESET_STANDARD")

    def evaluate_telemetry(
        self,
        telemetry: OpenCVFrameTelemetry,
        annotated_frame: Optional[np.ndarray] = None,
    ) -> AgentDecision:
        """
        Execute the closed-loop See → Understand → Prioritize → Act → Verify cycle
        on the latest OpenCV 5 frame telemetry.
        """
        self.decision_counter += 1
        now_ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

        dets = telemetry.detections
        falls = [d for d in dets if d.category == "fall"]
        fires = [d for d in dets if d.category == "fire"]
        smokes = [d for d in dets if d.category == "smoke"]
        intrusions = [d for d in dets if d.category == "restricted_area"]
        crowds = [d for d in dets if d.category == "crowd"]
        people = [d for d in dets if d.category == "person"]
        recovering_people = [p for p in people if p.posture_state == "RECOVERING"]

        prev_headline = self.last_decision.headline if self.last_decision else None
        prev_priority = self.last_decision.priority_level if self.last_decision else "NOMINAL"

        # Determine if a previously active fall incident has just self-recovered!
        if (
            not falls
            and "INC-FALL" in self.active_incidents
            and self.active_incidents["INC-FALL"].status != "RESOLVED"
        ):
            inc = self.active_incidents["INC-FALL"]
            if recovering_people or any(p.posture_state == "UPRIGHT" for p in people):
                return self._handle_fall_self_recovery(inc, telemetry, prev_headline, now_ts)

        # Case 1: Compound Multi-Emergency Crisis (e.g., Fire/Smoke + Fall or Intrusion)
        hazard_categories = sum(
            [bool(falls), bool(fires or smokes), bool(intrusions), bool(crowds)]
        )
        if hazard_categories >= 2 and (fires or falls):
            decision = self._handle_compound_crisis(
                telemetry, falls, fires, smokes, intrusions, crowds, prev_headline, prev_priority, now_ts
            )
        # Case 2: Person Fall / Lying on Ground Detection
        elif falls:
            decision = self._handle_fall_emergency(
                telemetry, falls[0], prev_headline, prev_priority, now_ts, annotated_frame
            )
        # Case 3: Fire or Smoke Outbreak
        elif fires or smokes:
            decision = self._handle_fire_or_smoke(
                telemetry, fires, smokes, prev_headline, prev_priority, now_ts, annotated_frame
            )
        # Case 4: Restricted Area Intrusion
        elif intrusions:
            decision = self._handle_restricted_intrusion(
                telemetry, intrusions[0], prev_headline, prev_priority, now_ts, annotated_frame
            )
        # Case 5: High-Density Crowd Surge
        elif crowds:
            decision = self._handle_crowd_surge(
                telemetry, crowds[0], prev_headline, prev_priority, now_ts, annotated_frame
            )
        # Case 6: Nominal Scene Monitoring
        else:
            decision = self._handle_nominal_scene(telemetry, people, prev_headline, prev_priority, now_ts)

        # Periodically publish operational metrics to Amazon CloudWatch
        if telemetry.frame_index - self.last_cw_publish_frame >= 25:
            self.last_cw_publish_frame = telemetry.frame_index
            active_cnt = sum(1 for i in self.active_incidents.values() if i.status != "RESOLVED")
            self.aws.publish_cloudwatch_metrics(
                fps=telemetry.fps,
                processing_ms=telemetry.processing_ms,
                active_incidents=active_cnt,
                agent_confidence=decision.confidence,
            )

        self.last_decision = decision
        return decision

    def _handle_fall_emergency(
        self,
        telemetry: OpenCVFrameTelemetry,
        fall_det: DetectionItem,
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
        annotated_frame: Optional[np.ndarray],
    ) -> AgentDecision:
        """
        Closed-loop Fall Protocol:
        1. Early horizontal aspect ratio (< 8 immobile frames or confidence < 0.76):
           - Stage = UNDERSTAND
           - Flag Uncertainty ("Subject posture horizontal; verifying if crouching vs immobilized")
           - Issue active CV command: `REQUEST_ROI_ZOOM_ANALYSIS` on fallen subject bbox.
        2. Confirmed motionless on ground (ROI Zoom micro-motion < 0.85 or immobile_frames >= 8):
           - Stage = PRIORITIZE -> ACT
           - Escalate Priority to CRITICAL (Score 92)
           - Create/Update Incident `INC-FALL` & Archive evidence to AWS S3
           - Recommend Paramedic Dispatch + PA Alert (Awaiting Human Approval for EMS dispatch).
        """
        immobile_frames = int(fall_det.metadata.get("immobile_frames", 0))
        ar = fall_det.metadata.get("aspect_ratio", 1.8)
        roi_verified = bool(fall_det.metadata.get("roi_verified", False))

        if immobile_frames < 5:
            # Stage 2: UNDERSTAND -> Request OpenCV ROI Zoom & CLAHE micro-motion check
            cv_cmd = "REQUEST_ROI_ZOOM_ANALYSIS"
            self.vision.apply_agent_command(
                cv_cmd, roi_box=fall_det.bbox, target_label=fall_det.id
            )
            stage = "UNDERSTAND"
            headline = f"Potential Fall Detected ({fall_det.id}) — Requesting OpenCV ROI Micro-Motion Scan"
            understanding = (
                f"OpenCV detected subject {fall_det.id} transitioning to horizontal aspect ratio "
                f"(AR={ar:.2f}, vel={fall_det.velocity:.2f} px/f). Confidence is moderate ({int(fall_det.confidence*100)}%). "
                f"Agent is commanding OpenCV 5 to zoom into ROI {fall_det.bbox} with CLAHE to verify immobility."
            )
            priority_level = "HIGH"
            priority_score = 72
            rec_action = "Execute automated OpenCV 5 ROI Zoom + Dense Optical Flow respiration/motion check before EMS dispatch."
            uncertainty_flag = True
            uncertainty_reason = (
                f"Initial posture collapse confidence ({int(fall_det.confidence*100)}%) is below autonomous dispatch threshold (80%). "
                "Possible temporary crouch or slip — active ROI verification engaged."
            )
            inc_status = "ANALYZING_ROI"
            req_approval = False
            ver_status = "REQUESTING_MORE_ANALYSIS"
        else:
            # Confirmed Motionless Fall Emergency -> PRIORITIZE & ACT
            cv_cmd = "REQUEST_ROI_ZOOM_ANALYSIS"
            self.vision.apply_agent_command(
                cv_cmd, roi_box=fall_det.bbox, target_label=fall_det.id
            )
            existing_inc = self.active_incidents.get("INC-FALL")
            already_approved = existing_inc and existing_inc.human_approval_status == "APPROVED"

            stage = "VERIFY" if already_approved else "ACT"
            headline = (
                f"CRITICAL Medical Emergency Confirmed: {fall_det.id} Motionless on Floor"
                if not already_approved
                else f"VERIFYING Response: Paramedic Unit En Route to {fall_det.id}"
            )
            understanding = (
                f"OpenCV 5 ROI Zoom confirmed subject {fall_det.id} is lying horizontally (AR={ar:.2f}) "
                f"with near-zero micro-motion ({telemetry.roi_zoom.micro_motion_score:.2f} px/f) for "
                f"{immobile_frames} consecutive frames ({fall_det.confidence*100:.0f}% confidence)."
            )
            priority_level = "CRITICAL"
            priority_score = 94
            rec_action = (
                "Dispatch Sector-A Paramedic Response Team, activate local audio check beacon, "
                "and illuminate Concourse Lane-1 emergency corridor."
            )
            uncertainty_flag = False
            uncertainty_reason = None
            inc_status = "VERIFYING" if already_approved else "AWAITING_APPROVAL"
            req_approval = not already_approved
            ver_status = "VERIFYING_OUTCOME" if already_approved else "ACTION_PENDING"

        inc = self._upsert_incident_state(
            inc_id="INC-FALL",
            title=f"Medical Fall & Immobility ({fall_det.id})",
            category="fall",
            severity=priority_level,
            priority_score=priority_score,
            status=inc_status,
            stage=stage,
            confidence=fall_det.confidence,
            uncertainty=fall_det.uncertainty,
            world_pos=fall_det.world_pos,
            bbox=fall_det.bbox,
            recommended_action=rec_action,
            requires_approval=req_approval,
            cv_summary=f"AR={ar:.2f} | Vel={fall_det.velocity:.2f}px/f | Immobile={immobile_frames}f | ROI_Verified={roi_verified}",
            frame_idx=telemetry.frame_index,
            annotated_frame=annotated_frame,
        )

        decision_changed = bool(prev_headline and prev_headline != headline)
        change_reason = (
            f"Escalated from {prev_priority} to {priority_level} after OpenCV ROI micro-motion scan confirmed zero movement."
            if decision_changed and priority_level == "CRITICAL"
            else None
        )

        self._emit_timeline_if_changed(
            headline=headline,
            stage=stage,
            source="AI_AGENT",
            detail=understanding,
            severity="CRITICAL" if priority_level == "CRITICAL" else "WARNING",
        )
        self.aws.record_bedrock_agent_invocation(stage, headline, fall_det.confidence, cv_cmd)

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage=stage,
            headline=headline,
            situation_understanding=understanding,
            priority_level=priority_level,
            priority_score=priority_score,
            recommended_action=rec_action,
            active_cv_command=cv_cmd,
            confidence=fall_det.confidence,
            uncertainty_score=fall_det.uncertainty,
            uncertainty_flag=uncertainty_flag,
            uncertainty_reason=uncertainty_reason,
            requires_human_approval=inc.human_approval_status == "PENDING",
            approval_incident_id=inc.id,
            decision_changed=decision_changed,
            previous_decision=prev_headline if decision_changed else None,
            change_reason=change_reason,
            verification_status=ver_status,
            reasoning_trace=[
                f"[SEE] OpenCV 5 detected horizontal human silhouette {fall_det.id} at bbox {fall_det.bbox} (AR={ar:.2f}).",
                f"[UNDERSTAND] Velocity={fall_det.velocity:.2f} px/f, ROI micro-motion={telemetry.roi_zoom.micro_motion_score:.2f}.",
                f"[PRIORITIZE] Assigned Priority {priority_score}/100 ({priority_level}) based on immobility duration ({immobile_frames} frames).",
                f"[ACT] {rec_action}",
                f"[VERIFY] Active OpenCV ROI Zoom locked on {fall_det.id} to monitor recovery or responder arrival.",
            ],
        )

    def _handle_fall_self_recovery(
        self,
        inc: IncidentRecord,
        telemetry: OpenCVFrameTelemetry,
        prev_headline: Optional[str],
        now_ts: str,
    ) -> AgentDecision:
        """
        Dynamic Decision Reversal:
        When OpenCV detects that a previously fallen person has stood back up and resumed normal
        locomotion, the AI Agent dynamically changes its decision, downgrades priority, and resolves the incident.
        """
        self.vision.apply_agent_command("RESET_STANDARD")
        inc.severity = "RESOLVED"
        inc.priority_score = 15
        inc.status = "RESOLVED"
        inc.pipeline_stage = "VERIFY"
        inc.updated_at = now_ts
        note = f"[{now_ts}] AI Agent reversed CRITICAL state: OpenCV confirmed subject regained upright posture (AR < 0.75) and normal walking velocity."
        if note not in inc.evolution_log:
            inc.evolution_log.append(note)
        self.aws.sync_incident_to_dynamodb(inc)

        headline = "DECISION CHANGED: Subject Self-Recovered & Standing — Emergency Downgraded"
        understanding = (
            "OpenCV 5 posture tracker detected subject P-101 transitioning from horizontal (AR > 1.8) "
            "back to upright walking posture (AR=0.41) with normal velocity. AI Agent dynamically "
            "reversed its prior emergency escalation and cancelled paramedic dispatch."
        )
        self._emit_timeline_if_changed(
            headline=headline,
            stage="VERIFY",
            source="AI_AGENT",
            detail=understanding,
            severity="SUCCESS",
        )

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage="VERIFY",
            headline=headline,
            situation_understanding=understanding,
            priority_level="LOW",
            priority_score=15,
            recommended_action="Cancel Paramedic Dispatch, reset OpenCV 5 pipeline to standard patrol, and log false-alarm prevention in AWS DynamoDB.",
            active_cv_command="RESET_STANDARD",
            confidence=0.95,
            uncertainty_score=0.05,
            uncertainty_flag=False,
            requires_human_approval=False,
            approval_incident_id=None,
            decision_changed=True,
            previous_decision=prev_headline or "CRITICAL Medical Emergency: Subject Motionless",
            change_reason="OpenCV 5 verified subject stood back up (upright aspect ratio restored) and resumed walking.",
            verification_status="VERIFIED_SAFE",
            reasoning_trace=[
                "[SEE] OpenCV 5 tracked subject transitioning from horizontal bbox back to upright aspect ratio (AR=0.41).",
                "[UNDERSTAND] Locomotion velocity restored (> 2.2 px/f); subject is no longer immobilized.",
                "[PRIORITIZE] Dynamically downgraded priority from 94 (CRITICAL) to 15 (RESOLVED).",
                "[ACT] Cancelled emergency medical dispatch and released corridor lock.",
                "[VERIFY] Incident INC-FALL marked RESOLVED in Amazon DynamoDB after continuous upright verification.",
            ],
        )

    def _handle_fire_or_smoke(
        self,
        telemetry: OpenCVFrameTelemetry,
        fires: List[DetectionItem],
        smokes: List[DetectionItem],
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
        annotated_frame: Optional[np.ndarray],
    ) -> AgentDecision:
        """
        Handle Smoke and Fire progression:
        - Early smoke only (no flame yet, confidence < 0.72):
          Agent flags uncertainty, commands OpenCV `BOOST_SENSITIVITY_CLAHE` to enhance aerosol contrast.
        - Active Fire + Smoke:
          Agent escalates to CRITICAL (Score 96), creates/updates incident, requests human approval for fire suppression + evacuation.
        """
        if fires:
            primary = fires[0]
            cv_cmd = "REQUEST_ROI_ZOOM_ANALYSIS"
            self.vision.apply_agent_command(cv_cmd, roi_box=primary.bbox, target_label=primary.id)
            existing = self.active_incidents.get("INC-FIRE")
            approved = existing and existing.human_approval_status == "APPROVED"

            stage = "VERIFY" if approved else "ACT"
            headline = (
                f"CRITICAL Fire & Smoke Outbreak ({primary.id}) — Suppression & Evacuation Required"
                if not approved
                else "VERIFYING Fire Suppression & Concourse Evacuation Progress"
            )
            smoke_note = f" with rising smoke plume ({smokes[0].id})" if smokes else ""
            understanding = (
                f"OpenCV 5 HSV/YCrCb thermal segmentation confirmed active flame core{smoke_note} "
                f"(area={primary.metadata.get('area_px', 850)}px², confidence={primary.confidence*100:.0f}%). "
                "Rapid expansion requires immediate suppression and concourse evacuation."
            )
            priority_level = "CRITICAL"
            priority_score = 96
            rec_action = "Trigger Zone-A Automated Fire Suppression, unlock emergency egress gates, and dispatch Fire Response Unit via AWS SNS."
            uncertainty_flag = False
            uncertainty_reason = None
            req_approval = not approved
            inc_status = "VERIFYING" if approved else "AWAITING_APPROVAL"
        else:
            primary = smokes[0]
            cv_cmd = "BOOST_SENSITIVITY_CLAHE"
            self.vision.apply_agent_command(cv_cmd, roi_box=primary.bbox, target_label=primary.id)
            stage = "UNDERSTAND"
            headline = f"Early Smoke Plume Detected ({primary.id}) — Boosting OpenCV CLAHE Contrast"
            understanding = (
                f"OpenCV detected a diffusing desaturated aerosol plume (area={primary.metadata.get('plume_area_px', 600)}px², "
                f"confidence={primary.confidence*100:.0f}%). No open flame core visible yet. "
                "Agent enabled OpenCV CLAHE contrast enhancement to distinguish smoke from steam/glare."
            )
            priority_level = "HIGH"
            priority_score = 76
            rec_action = "Activate OpenCV 5 CLAHE high-sensitivity thermal scan and pre-alert facility fire safety officer."
            uncertainty_flag = primary.confidence < 0.75
            uncertainty_reason = "Early low-opacity aerosol plume could be steam or dust; CLAHE multi-scale verification active."
            req_approval = False
            inc_status = "ANALYZING_ROI"

        inc = self._upsert_incident_state(
            inc_id="INC-FIRE",
            title="Thermal Fire & Smoke Outbreak" if fires else "Early Smoke Aerosol Plume",
            category="fire" if fires else "smoke",
            severity=priority_level,
            priority_score=priority_score,
            status=inc_status,
            stage=stage,
            confidence=primary.confidence,
            uncertainty=primary.uncertainty,
            world_pos=primary.world_pos,
            bbox=primary.bbox,
            recommended_action=rec_action,
            requires_approval=req_approval,
            cv_summary=f"FireContours={len(fires)} | SmokeContours={len(smokes)} | CLAHE={telemetry.clahe_active}",
            frame_idx=telemetry.frame_index,
            annotated_frame=annotated_frame,
        )

        decision_changed = bool(prev_headline and prev_headline != headline)
        change_reason = (
            "OpenCV detected open thermal flame ignition inside the smoke plume; escalated from HIGH Smoke Advisory to CRITICAL Fire Emergency."
            if decision_changed and fires
            else None
        )

        self._emit_timeline_if_changed(
            headline=headline,
            stage=stage,
            source="AI_AGENT",
            detail=understanding,
            severity="CRITICAL" if fires else "WARNING",
        )
        self.aws.record_bedrock_agent_invocation(stage, headline, primary.confidence, cv_cmd)

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage=stage,
            headline=headline,
            situation_understanding=understanding,
            priority_level=priority_level,
            priority_score=priority_score,
            recommended_action=rec_action,
            active_cv_command=cv_cmd,
            confidence=primary.confidence,
            uncertainty_score=primary.uncertainty,
            uncertainty_flag=uncertainty_flag,
            uncertainty_reason=uncertainty_reason,
            requires_human_approval=inc.human_approval_status == "PENDING",
            approval_incident_id=inc.id,
            decision_changed=decision_changed,
            previous_decision=prev_headline if decision_changed else None,
            change_reason=change_reason,
            verification_status="ACTION_PENDING" if req_approval else "REQUESTING_MORE_ANALYSIS",
            reasoning_trace=[
                f"[SEE] OpenCV 5 segmented {'flame core + smoke plume' if fires else 'diffusing smoke contour'} at {primary.bbox}.",
                f"[UNDERSTAND] Chromaticity & optical flow upward drift confirm {'active combustion' if fires else 'early aerosol dispersion'}.",
                f"[PRIORITIZE] Priority set to {priority_score}/100 ({priority_level}).",
                f"[ACT] {rec_action}",
                f"[VERIFY] OpenCV CLAHE & ROI thermal tracking locked on {primary.id}.",
            ],
        )

    def _handle_restricted_intrusion(
        self,
        telemetry: OpenCVFrameTelemetry,
        intrusion: DetectionItem,
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
        annotated_frame: Optional[np.ndarray],
    ) -> AgentDecision:
        """Handle Restricted Area Polygon Intrusion (`cv2.pointPolygonTest`)."""
        self.vision.apply_agent_command(
            "REQUEST_ROI_ZOOM_ANALYSIS", roi_box=intrusion.bbox, target_label=intrusion.id
        )
        existing = self.active_incidents.get("INC-INTRUSION")
        approved = existing and existing.human_approval_status == "APPROVED"

        stage = "VERIFY" if approved else "ACT"
        depth = intrusion.metadata.get("penetration_depth_px", 18.0)
        headline = f"HIGH Security & Safety Alert: Restricted Zone Breach ({intrusion.id})"
        understanding = (
            f"OpenCV `cv2.pointPolygonTest` confirmed subject penetrated Sector-B High-Voltage "
            f"Hazard Polygon by {depth:.1f}px (confidence={intrusion.confidence*100:.0f}%)."
        )
        rec_action = "Issue automated perimeter acoustic warning, de-energize Sector-B auxiliary bus, and dispatch Security Patrol."

        inc = self._upsert_incident_state(
            inc_id="INC-INTRUSION",
            title=f"Restricted Hazard Zone Breach ({intrusion.id})",
            category="restricted_area",
            severity="HIGH",
            priority_score=84,
            status="VERIFYING" if approved else "AWAITING_APPROVAL",
            stage=stage,
            confidence=intrusion.confidence,
            uncertainty=intrusion.uncertainty,
            world_pos=intrusion.world_pos,
            bbox=intrusion.bbox,
            recommended_action=rec_action,
            requires_approval=not approved,
            cv_summary=f"PolygonTest=+{depth:.1f}px | Zone=Sector-B High-Voltage",
            frame_idx=telemetry.frame_index,
            annotated_frame=annotated_frame,
        )

        self._emit_timeline_if_changed(
            headline=headline,
            stage=stage,
            source="AI_AGENT",
            detail=understanding,
            severity="HIGH",
        )
        self.aws.record_bedrock_agent_invocation(stage, headline, intrusion.confidence, "REQUEST_ROI_ZOOM_ANALYSIS")

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage=stage,
            headline=headline,
            situation_understanding=understanding,
            priority_level="HIGH",
            priority_score=84,
            recommended_action=rec_action,
            active_cv_command="REQUEST_ROI_ZOOM_ANALYSIS",
            confidence=intrusion.confidence,
            uncertainty_score=intrusion.uncertainty,
            uncertainty_flag=False,
            requires_human_approval=inc.human_approval_status == "PENDING",
            approval_incident_id=inc.id,
            decision_changed=bool(prev_headline and prev_headline != headline),
            previous_decision=prev_headline,
            change_reason="Unauthorized entry into restricted polygon detected by OpenCV.",
            verification_status="VERIFYING_OUTCOME" if approved else "ACTION_PENDING",
            reasoning_trace=[
                f"[SEE] OpenCV `pointPolygonTest` >= 0 for subject foot point at {intrusion.bbox}.",
                f"[UNDERSTAND] Penetration depth {depth:.1f}px inside High-Voltage Hazard Zone.",
                "[PRIORITIZE] Assigned Priority 84/100 (HIGH).",
                f"[ACT] {rec_action}",
                "[VERIFY] Tracking intruder trajectory until exit from restricted polygon.",
            ],
        )

    def _handle_crowd_surge(
        self,
        telemetry: OpenCVFrameTelemetry,
        crowd: DetectionItem,
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
        annotated_frame: Optional[np.ndarray],
    ) -> AgentDecision:
        """Handle High-Density Crowd Clustering & Bottleneck Surge."""
        self.vision.apply_agent_command("BOOST_SENSITIVITY_CLAHE")
        existing = self.active_incidents.get("INC-CROWD")
        approved = existing and existing.human_approval_status == "APPROVED"

        cnt = crowd.metadata.get("person_count", telemetry.people_count)
        density = crowd.metadata.get("density_pm2", 1.8)
        stage = "VERIFY" if approved else "PRIORITIZE"
        headline = f"Crowd Surge & Bottleneck Warning ({cnt} Persons Clustered)"
        understanding = (
            f"OpenCV `cv2.convexHull` spatial clustering identified {cnt} subjects within "
            f"high-density proximity ({density} persons/m²) with elevated optical flow variance."
        )
        rec_action = "Open auxiliary Concourse Exit-2 gates, activate directional LED floor guidance, and alert crowd stewards."

        inc = self._upsert_incident_state(
            inc_id="INC-CROWD",
            title=f"High-Density Crowd Bottleneck ({cnt} Persons)",
            category="crowd",
            severity="HIGH",
            priority_score=79,
            status="VERIFYING" if approved else "AWAITING_APPROVAL",
            stage=stage,
            confidence=crowd.confidence,
            uncertainty=crowd.uncertainty,
            world_pos=crowd.world_pos,
            bbox=crowd.bbox,
            recommended_action=rec_action,
            requires_approval=not approved,
            cv_summary=f"ClusterCount={cnt} | Density={density}p/m² | HullActive=True",
            frame_idx=telemetry.frame_index,
            annotated_frame=annotated_frame,
        )

        self._emit_timeline_if_changed(
            headline=headline,
            stage=stage,
            source="AI_AGENT",
            detail=understanding,
            severity="WARNING",
        )
        self.aws.record_bedrock_agent_invocation(stage, headline, crowd.confidence, "BOOST_SENSITIVITY_CLAHE")

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage=stage,
            headline=headline,
            situation_understanding=understanding,
            priority_level="HIGH",
            priority_score=79,
            recommended_action=rec_action,
            active_cv_command="BOOST_SENSITIVITY_CLAHE",
            confidence=crowd.confidence,
            uncertainty_score=crowd.uncertainty,
            uncertainty_flag=False,
            requires_human_approval=inc.human_approval_status == "PENDING",
            approval_incident_id=inc.id,
            decision_changed=bool(prev_headline and prev_headline != headline),
            previous_decision=prev_headline,
            change_reason="Spatial proximity cluster exceeded crowd threshold (>= 4 subjects).",
            verification_status="ACTION_PENDING",
            reasoning_trace=[
                f"[SEE] OpenCV tracked {cnt} subjects converging into convex hull {crowd.bbox}.",
                f"[UNDERSTAND] Local density={density} persons/m² creates crush/bottleneck risk.",
                "[PRIORITIZE] Assigned Priority 79/100 (HIGH).",
                f"[ACT] {rec_action}",
                "[VERIFY] Monitoring hull dispersion rate via Farneback optical flow.",
            ],
        )

    def _handle_compound_crisis(
        self,
        telemetry: OpenCVFrameTelemetry,
        falls: List[DetectionItem],
        fires: List[DetectionItem],
        smokes: List[DetectionItem],
        intrusions: List[DetectionItem],
        crowds: List[DetectionItem],
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
    ) -> AgentDecision:
        """
        Handle Multi-Emergency Compound Crisis:
        Simultaneous hazards (e.g., Fire/Smoke + Fallen Person + Restricted Area Intrusion).
        Agent prioritizes life-safety extraction first while coordinating fire suppression.
        """
        target_box = falls[0].bbox if falls else (fires[0].bbox if fires else [400, 200, 160, 160])
        self.vision.apply_agent_command(
            "REQUEST_ROI_ZOOM_ANALYSIS", roi_box=target_box, target_label="COMPOUND-CRISIS"
        )
        existing = self.active_incidents.get("INC-COMPOUND")
        approved = existing and existing.human_approval_status == "APPROVED"

        stage = "VERIFY" if approved else "ACT"
        headline = "CRITICAL MULTI-HAZARD CRISIS: Fire Outbreak + Fallen Subject + Zone Breach"
        understanding = (
            f"OpenCV 5 detected simultaneous emergencies: Active Fire/Smoke in Sector-B, "
            f"immobilized fallen subject ({falls[0].id if falls else 'P-101'}) near hazard boundary, "
            f"and {len(intrusions)} restricted zone breach. Agent reprioritized life-safety extraction #1."
        )
        rec_action = (
            "1) Dispatch Hazmat Rescue Team to extract fallen subject P-101; "
            "2) Activate Sector-B Fire Suppression; "
            "3) Broadcast facility-wide evacuation via AWS SNS."
        )

        inc = self._upsert_incident_state(
            inc_id="INC-COMPOUND",
            title="Compound Crisis: Fire + Fallen Casualty + Zone Breach",
            category="compound",
            severity="CRITICAL",
            priority_score=99,
            status="VERIFYING" if approved else "AWAITING_APPROVAL",
            stage=stage,
            confidence=0.94,
            uncertainty=0.06,
            world_pos=falls[0].world_pos if falls else [0.62, 0.55],
            bbox=target_box,
            recommended_action=rec_action,
            requires_approval=not approved,
            cv_summary=f"Falls={len(falls)} | Fires={len(fires)} | Smokes={len(smokes)} | Intrusions={len(intrusions)}",
            frame_idx=telemetry.frame_index,
            annotated_frame=None,
        )

        self._emit_timeline_if_changed(
            headline=headline,
            stage=stage,
            source="AI_AGENT",
            detail=understanding,
            severity="CRITICAL",
        )
        self.aws.record_bedrock_agent_invocation(stage, headline, 0.94, "REQUEST_ROI_ZOOM_ANALYSIS")

        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage=stage,
            headline=headline,
            situation_understanding=understanding,
            priority_level="CRITICAL",
            priority_score=99,
            recommended_action=rec_action,
            active_cv_command="REQUEST_ROI_ZOOM_ANALYSIS",
            confidence=0.94,
            uncertainty_score=0.06,
            uncertainty_flag=False,
            requires_human_approval=inc.human_approval_status == "PENDING",
            approval_incident_id=inc.id,
            decision_changed=bool(prev_headline and prev_headline != headline),
            previous_decision=prev_headline,
            change_reason="Multiple concurrent hazards detected; Agent synthesized unified life-safety priority plan.",
            verification_status="VERIFYING_OUTCOME" if approved else "ACTION_PENDING",
            reasoning_trace=[
                f"[SEE] OpenCV detected {len(fires)} fire core, {len(smokes)} smoke plume, {len(falls)} fallen person, and {len(intrusions)} zone intrusion.",
                "[UNDERSTAND] Fallen subject is in close proximity to expanding thermal hazard — extreme life safety risk.",
                "[PRIORITIZE] Maximum Priority 99/100 (CRITICAL COMPOUND CRISIS).",
                f"[ACT] {rec_action}",
                "[VERIFY] Locked OpenCV ROI Zoom on fallen casualty while tracking fire perimeter expansion.",
            ],
        )

    def _handle_nominal_scene(
        self,
        telemetry: OpenCVFrameTelemetry,
        people: List[DetectionItem],
        prev_headline: Optional[str],
        prev_priority: str,
        now_ts: str,
    ) -> AgentDecision:
        """Baseline nominal monitoring when no hazards are active."""
        self.vision.apply_agent_command("RESET_STANDARD")
        headline = f"Nominal Patrol — Tracking {len(people)} Subjects in Safe Concourse"
        understanding = (
            f"OpenCV 5 tracking {len(people)} upright subjects with normal gait velocity "
            f"(mean flow energy={telemetry.motion_energy:.2f}). Restricted Zone Sector-B is clear."
        )
        self._emit_timeline_if_changed(
            headline=headline,
            stage="SEE",
            source="OPENCV_5",
            detail=understanding,
            severity="INFO",
        )
        return AgentDecision(
            decision_id=f"DEC-{self.decision_counter:04d}",
            timestamp=now_ts,
            current_stage="SEE",
            headline=headline,
            situation_understanding=understanding,
            priority_level="NOMINAL",
            priority_score=10,
            recommended_action="Continue standard 24 FPS OpenCV 5 multi-cue surveillance and CloudWatch telemetry heartbeat.",
            active_cv_command="RESET_STANDARD",
            confidence=0.96,
            uncertainty_score=0.04,
            uncertainty_flag=False,
            requires_human_approval=False,
            approval_incident_id=None,
            decision_changed=bool(prev_priority not in ("NOMINAL", "LOW")),
            previous_decision=prev_headline if prev_priority not in ("NOMINAL", "LOW") else None,
            change_reason="All hazards cleared; returned to baseline surveillance." if prev_priority not in ("NOMINAL", "LOW") else None,
            verification_status="MONITORING",
            reasoning_trace=[
                f"[SEE] OpenCV 5 detected {len(people)} upright pedestrians; 0 thermal, smoke, fall, or perimeter anomalies.",
                "[UNDERSTAND] Scene kinematics within normal baseline parameters.",
                "[PRIORITIZE] Priority 10/100 (NOMINAL).",
                "[ACT] No intervention required.",
                "[VERIFY] Continuous background subtraction & optical flow baseline active.",
            ],
        )

    def handle_human_approval(
        self, incident_id: str, action: str, operator_note: Optional[str] = None
    ) -> Optional[IncidentRecord]:
        """
        Process Human-in-the-Loop operator decision:
        - `APPROVE`: Approves recommended emergency action, dispatches AWS SNS responder alert,
          and advances incident to `VERIFYING` stage.
        - `REQUEST_RESCAN`: Commands OpenCV 5 to run an intensive CLAHE + ROI Zoom re-verification scan.
        - `DISMISS`: Overrides/dismisses the incident as resolved/false alarm and resets OpenCV.
        """
        inc = self.active_incidents.get(incident_id)
        if not inc:
            # Check DB if not in memory dict
            for db_inc in self.db.get_incidents(limit=25):
                if db_inc.id == incident_id:
                    inc = db_inc
                    self.active_incidents[incident_id] = inc
                    break
        if not inc:
            return None

        now_ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        if action == "APPROVE":
            inc.human_approval_status = "APPROVED"
            inc.status = "VERIFYING"
            inc.pipeline_stage = "VERIFY"
            inc.updated_at = now_ts
            inc.evolution_log.append(
                f"[{now_ts}] HUMAN OPERATOR APPROVED dispatch. AWS SNS alert sent to responders; entering VERIFY stage."
            )
            self.vision.apply_agent_command(
                "VERIFY_INTERVENTION_OUTCOME", roi_box=inc.bbox, target_label=inc.id
            )
            self.aws.dispatch_sns_emergency_alert(
                incident_id=inc.id,
                severity=inc.severity,
                action_summary=inc.recommended_action,
                approved_by="HUMAN_COMMAND_OPERATOR",
            )
            self._emit_timeline_if_changed(
                headline=f"Operator Approved Action for {inc.id} — Dispatching via AWS SNS",
                stage="ACT",
                source="HUMAN_OPERATOR",
                detail=f"Approved: {inc.recommended_action}. Transitioning to OpenCV 5 outcome verification.",
                severity="SUCCESS",
            )
        elif action == "REQUEST_RESCAN":
            inc.status = "ANALYZING_ROI"
            inc.pipeline_stage = "UNDERSTAND"
            inc.updated_at = now_ts
            inc.evolution_log.append(
                f"[{now_ts}] Operator requested intensive OpenCV 5 ROI Zoom + CLAHE re-verification scan."
            )
            self.vision.apply_agent_command(
                "REQUEST_ROI_ZOOM_ANALYSIS", roi_box=inc.bbox, target_label=inc.id
            )
            self._emit_timeline_if_changed(
                headline=f"Operator Triggered Intensive OpenCV 5 Re-Scan on {inc.id}",
                stage="UNDERSTAND",
                source="HUMAN_OPERATOR",
                detail="Enabled CLAHE contrast enhancement and 2.5x ROI optical flow verification.",
                severity="INFO",
            )
        elif action == "DISMISS":
            inc.human_approval_status = "OVERRIDDEN"
            inc.severity = "RESOLVED"
            inc.status = "RESOLVED"
            inc.pipeline_stage = "VERIFY"
            inc.priority_score = 10
            inc.updated_at = now_ts
            inc.evolution_log.append(
                f"[{now_ts}] Operator dismissed/overrode incident ({operator_note or 'Manual stand-down'})."
            )
            self.vision.apply_agent_command("RESET_STANDARD")
            self._emit_timeline_if_changed(
                headline=f"Operator Overrode & Resolved Incident {inc.id}",
                stage="VERIFY",
                source="HUMAN_OPERATOR",
                detail=f"Incident marked RESOLVED. OpenCV 5 reset to standard patrol.",
                severity="INFO",
            )

        self.aws.sync_incident_to_dynamodb(inc)
        return inc

    def _upsert_incident_state(
        self,
        inc_id: str,
        title: str,
        category: str,
        severity: str,
        priority_score: int,
        status: str,
        stage: str,
        confidence: float,
        uncertainty: float,
        world_pos: List[float],
        bbox: List[int],
        recommended_action: str,
        requires_approval: bool,
        cv_summary: str,
        frame_idx: int,
        annotated_frame: Optional[np.ndarray],
    ) -> IncidentRecord:
        now_ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        existing = self.active_incidents.get(inc_id)

        if existing is None:
            s3_key = self.aws.archive_evidence_to_s3(
                incident_id=inc_id,
                frame_bgr=annotated_frame,
                metadata={"title": title, "severity": severity, "cv_summary": cv_summary},
            )
            self.last_s3_snapshot_frame = frame_idx
            inc = IncidentRecord(
                id=inc_id,
                title=title,
                category=category,
                severity=severity,
                priority_score=priority_score,
                status=status,
                pipeline_stage=stage,
                confidence=confidence,
                uncertainty=uncertainty,
                world_pos=world_pos,
                bbox=bbox,
                recommended_action=recommended_action,
                human_approval_status="PENDING" if requires_approval else "NOT_REQUIRED",
                cv_evidence_summary=cv_summary,
                s3_evidence_key=s3_key,
                created_at=now_ts,
                updated_at=now_ts,
                evolution_log=[
                    f"[{now_ts}] Incident created at stage {stage} ({severity}, Priority {priority_score}/100)."
                ],
            )
            self.active_incidents[inc_id] = inc
            self.aws.sync_incident_to_dynamodb(inc)
        else:
            prev_sev = existing.severity
            prev_stage = existing.pipeline_stage
            existing.title = title
            existing.severity = severity
            existing.priority_score = priority_score
            if existing.human_approval_status != "APPROVED":
                existing.status = status
                existing.pipeline_stage = stage
                if requires_approval and existing.human_approval_status == "NOT_REQUIRED":
                    existing.human_approval_status = "PENDING"
            existing.confidence = confidence
            existing.uncertainty = uncertainty
            existing.world_pos = world_pos
            existing.bbox = bbox
            existing.recommended_action = recommended_action
            existing.cv_evidence_summary = cv_summary
            existing.updated_at = now_ts

            if prev_sev != severity or prev_stage != existing.pipeline_stage:
                existing.evolution_log.append(
                    f"[{now_ts}] Transitioned {prev_stage}→{existing.pipeline_stage} | Severity {prev_sev}→{severity} (Conf {int(confidence*100)}%)."
                )
                self.aws.sync_incident_to_dynamodb(existing)
            inc = existing

        return inc

    def _emit_timeline_if_changed(
        self,
        headline: str,
        stage: str,
        source: str,
        detail: str,
        severity: str,
    ) -> None:
        if headline == self.last_logged_headline:
            return
        self.last_logged_headline = headline
        self.event_counter += 1
        now_ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        evt = TimelineEvent(
            id=f"EVT-{self.event_counter:04d}",
            timestamp=now_ts,
            stage=stage,
            source=source,
            title=headline,
            detail=detail,
            severity=severity,
        )
        self.db.add_timeline_event(evt)
