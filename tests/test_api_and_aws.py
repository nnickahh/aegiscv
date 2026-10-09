"""Integration tests for FastAPI endpoints, AWS service status, and live evaluation benchmarks."""
from fastapi.testclient import TestClient
from backend.main import app


client = TestClient(app)


def test_health_and_landing_page() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["project"] == "AI Emergency Coordinator"

    html_res = client.get("/")
    assert html_res.status_code == 200
    assert "AI Emergency Coordinator" in html_res.text
    assert "See the situation. Understand the risk. Coordinate the response." in html_res.text


def test_scenario_switch_and_state_telemetry() -> None:
    res = client.post("/api/scenario/select", json={"scenario_id": "multi_crisis"})
    assert res.status_code == 200
    assert res.json()["scenario_id"] == "multi_crisis"

    state_res = client.get("/api/state?advance=true")
    assert state_res.status_code == 200
    state = state_res.json()
    assert "telemetry" in state
    assert "agent_decision" in state
    assert "aws_status" in state
    assert state["telemetry"]["frame_index"] >= 1


def test_video_frame_and_view_modes() -> None:
    for mode in ["opencv_annotated", "raw_video", "optical_flow", "split_comparison"]:
        m_res = client.post("/api/video/view-mode", json={"view_mode": mode})
        assert m_res.status_code == 200

        f_res = client.get("/api/video/frame")
        assert f_res.status_code == 200
        assert f_res.headers["content-type"] == "image/jpeg"
        assert len(f_res.content) > 2000


def test_evaluation_and_live_benchmark_endpoint() -> None:
    eval_res = client.get("/api/evaluation")
    assert eval_res.status_code == 200
    report = eval_res.json()
    assert "summary_metrics" in report
    assert "failure_cases" in report
    assert "limitations" in report

    bench_res = client.post("/api/evaluation/run-benchmark")
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert len(bench_data["live_benchmark_runs"]) == 6
    assert all(r["passed"] for r in bench_data["live_benchmark_runs"])


def test_custom_restricted_zone_and_audit_report_export() -> None:
    custom_poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    poly_res = client.post("/api/video/restricted-zone", json={"vertices": custom_poly})
    assert poly_res.status_code == 200
    assert len(poly_res.json()["polygon"]) == 4

    reset_res = client.post("/api/video/restricted-zone", json={"reset": True})
    assert reset_res.status_code == 200
    assert reset_res.json()["mode"] == "default"

    report_res = client.get("/api/incidents/export-report")
    assert report_res.status_code == 200
    assert "AI Emergency Coordinator — Post-Incident Audit Report" in report_res.text

