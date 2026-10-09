"""
Main entrypoint for AI Emergency Coordinator (OpenCV 5 + AWS Hackathon).

Usage:
    python app.py
"""
import uvicorn
from backend.scenario_synthesizer import generate_sample_mp4_files


def main() -> None:
    print("=" * 72)
    print("  AI EMERGENCY COORDINATOR — OPENCV 5 + AWS COMMAND CENTER")
    print("  Tagline: See the situation. Understand the risk. Coordinate the response.")
    print("=" * 72)
    sample_paths = generate_sample_mp4_files()
    print(f"[Init] Verified {len(sample_paths)} sample emergency MP4 videos in data/sample_videos/")
    print("[Init] Launching FastAPI + OpenCV 5 Server at http://127.0.0.1:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
