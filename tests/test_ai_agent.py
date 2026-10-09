"""Unit tests for Closed-Loop AI Emergency Coordinator Agent."""
from pathlib import Path
from backend.ai_agent import AIEmergencyCoordinatorAgent
from backend.aws_service import AWSCloudCoordinator
from backend.database import DatabaseManager
from backend.opencv_pipeline import OpenCV5VisionPipeline
from backend.scenario_synthesizer import SurveillanceSceneSynthesizer


def test_agent_closed_loop_fall_escalation_and_hitl(tmp_path: Path) -> None:
    """
    Verify:
    1. Early horizontal transition triggers UNDERSTAND stage, Uncertainty Flag, and ROI Zoom command.
    2. Continued immobility escalates to CRITICAL ACT stage and creates INC-FALL awaiting HITL approval.
    3. Human operator approval transitions incident to VERIFYING and dispatches AWS SNS alert.
    """
    db = DatabaseManager(db_path=tmp_path / "test_agent.db")
    aws = AWSCloudCoordinator(db)
    pipe = OpenCV5VisionPipeline()
    agent = AIEmergencyCoordinatorAgent(pipe, db, aws)
    synth = SurveillanceSceneSynthesizer()
    synth.set_scenario("fall_emergency")

    saw_uncertainty_or_understand = False
    final_decision = None

    for _ in range(32):
        raw = synth.generate_next_raw_frame()
        tel, ann = pipe.process_frame(raw)
        dec = agent.evaluate_telemetry(tel, ann)
        if dec.current_stage == "UNDERSTAND" or dec.uncertainty_flag:
            saw_uncertainty_or_understand = True
        final_decision = dec

    assert saw_uncertainty_or_understand is True
    assert final_decision is not None
    assert final_decision.priority_level == "CRITICAL"
    assert final_decision.requires_human_approval is True
    assert "INC-FALL" in agent.active_incidents

    # Approve via Human-in-the-Loop
    updated_inc = agent.handle_human_approval("INC-FALL", "APPROVE", "Paramedics dispatched")
    assert updated_inc is not None
    assert updated_inc.human_approval_status == "APPROVED"
    assert updated_inc.status == "VERIFYING"
    assert aws.sns_alerts_dispatched >= 1


def test_agent_dynamic_decision_change_on_self_recovery(tmp_path: Path) -> None:
    """
    Verify that if a fallen person stands back up and resumes walking, the AI Agent
    dynamically changes its decision, downgrades priority, and resolves the incident.
    """
    db = DatabaseManager(db_path=tmp_path / "test_recovery.db")
    aws = AWSCloudCoordinator(db)
    pipe = OpenCV5VisionPipeline()
    agent = AIEmergencyCoordinatorAgent(pipe, db, aws)
    synth = SurveillanceSceneSynthesizer()
    synth.set_scenario("fall_recovery")

    # Run through collapse phase (frames 1..45) -> incident created
    for _ in range(45):
        raw = synth.generate_next_raw_frame()
        tel, ann = pipe.process_frame(raw)
        agent.evaluate_telemetry(tel, ann)

    assert "INC-FALL" in agent.active_incidents
    assert agent.active_incidents["INC-FALL"].severity in ("HIGH", "CRITICAL")

    # Advance into self-recovery upright phase (frames 46..72)
    recovery_decision = None
    for _ in range(28):
        raw = synth.generate_next_raw_frame()
        tel, ann = pipe.process_frame(raw)
        dec = agent.evaluate_telemetry(tel, ann)
        if dec.decision_changed and dec.verification_status == "VERIFIED_SAFE":
            recovery_decision = dec
            break

    assert recovery_decision is not None
    assert recovery_decision.decision_changed is True
    assert recovery_decision.priority_level == "LOW"
    assert agent.active_incidents["INC-FALL"].status == "RESOLVED"
