"""Unit tests for OpenCV 5 Vision Pipeline across all emergency detection modalities."""
import numpy as np
from backend.opencv_pipeline import OpenCV5VisionPipeline
from backend.scenario_synthesizer import SurveillanceSceneSynthesizer


def test_opencv_person_and_fall_detection() -> None:
    """Verify OpenCV detects walking people and then detects a motionless horizontal fall."""
    pipe = OpenCV5VisionPipeline()
    synth = SurveillanceSceneSynthesizer()
    synth.set_scenario("fall_emergency")

    # First 5 frames: subjects are walking upright
    for _ in range(5):
        raw = synth.generate_next_raw_frame()
        tel, ann = pipe.process_frame(raw)
        assert ann.shape == (540, 960, 3)
        assert tel.people_count >= 1

    # Advance into motionless fall phase (frames 22..35)
    last_tel = None
    for _ in range(28):
        raw = synth.generate_next_raw_frame()
        last_tel, _ = pipe.process_frame(raw)

    assert last_tel is not None
    categories = {d.category for d in last_tel.detections}
    assert "person" in categories
    assert "fall" in categories
    fall_det = [d for d in last_tel.detections if d.category == "fall"][0]
    assert fall_det.posture_state in ("FALLING", "FALLEN_IMMOBILE")
    assert fall_det.confidence >= 0.60


def test_opencv_fire_and_smoke_detection() -> None:
    """Verify OpenCV detects early smoke plumes and subsequent thermal fire cores."""
    pipe = OpenCV5VisionPipeline()
    synth = SurveillanceSceneSynthesizer()
    synth.set_scenario("fire_smoke")

    # Early phase: smoke plume only
    early_tel = None
    for _ in range(8):
        raw = synth.generate_next_raw_frame()
        early_tel, _ = pipe.process_frame(raw)

    assert early_tel is not None
    early_cats = {d.category for d in early_tel.detections}
    assert "smoke" in early_cats

    # Later phase: active fire + smoke
    late_tel = None
    for _ in range(20):
        raw = synth.generate_next_raw_frame()
        late_tel, _ = pipe.process_frame(raw)

    assert late_tel is not None
    late_cats = {d.category for d in late_tel.detections}
    assert "fire" in late_cats


def test_opencv_crowd_and_restricted_zone_detection() -> None:
    """Verify OpenCV convexHull crowd surge detection and pointPolygonTest restricted zone breach."""
    pipe = OpenCV5VisionPipeline()
    synth = SurveillanceSceneSynthesizer()

    # Test Crowd Surge
    synth.set_scenario("crowd_surge")
    crowd_tel = None
    for _ in range(6):
        crowd_tel, _ = pipe.process_frame(synth.generate_next_raw_frame())
    assert crowd_tel is not None
    assert "crowd" in {d.category for d in crowd_tel.detections}
    assert crowd_tel.people_count >= 4

    # Test Restricted Area Intrusion
    pipe.reset_state()
    synth.set_scenario("restricted_intrusion")
    intr_tel = None
    for _ in range(15):
        intr_tel, _ = pipe.process_frame(synth.generate_next_raw_frame())
    assert intr_tel is not None
    assert "restricted_area" in {d.category for d in intr_tel.detections}


def test_opencv_agent_roi_zoom_and_clahe_feedback() -> None:
    """Verify active OpenCV reconfiguration when commanded by the AI Agent."""
    pipe = OpenCV5VisionPipeline()
    synth = SurveillanceSceneSynthesizer()
    synth.set_scenario("fall_emergency")

    pipe.apply_agent_command(
        "REQUEST_ROI_ZOOM_ANALYSIS", roi_box=[280, 290, 120, 70], target_label="FALL-P-101"
    )
    tel, _ = pipe.process_frame(synth.generate_next_raw_frame())
    assert tel.clahe_active is True
    assert tel.active_cv_mode == "VERIFICATION_SCAN"
    assert tel.roi_zoom.active is True
