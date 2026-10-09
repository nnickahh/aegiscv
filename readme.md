# aegiscv

[![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![opencv](https://img.shields.io/badge/opencv-5.x_real--time_vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![aws](https://img.shields.io/badge/aws-bedrock_%7C_s3_%7C_dynamodb_%7C_cloudwatch_%7C_sns-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![fastapi](https://img.shields.io/badge/fastapi-rest_%26_mjpeg_stream-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![pytest](https://img.shields.io/badge/pytest-11_passed-brightgreen?logo=pytest&logoColor=white)](https://github.com/nnickahh/aegiscv)
[![license](https://img.shields.io/badge/license-mit-blue)](LICENSE)

**aegiscv** is a real-time, closed-loop agentic computer vision and emergency coordination system built for the **opencv ai competition 2026 (powered by aws)**. it processes live, simulated, and uploaded surveillance video feeds using opencv 5 (aspect-ratio fall kinematics, hsv/ycrcb thermal flame & aerosol smoke segmentation, convex-hull crowd clustering, and geofenced polygon intrusion testing) and feeds telemetry into a stateful ai coordinator agent that actively commands opencv 5 (2.5× roi zoom + clahe re-scan) and dynamically reverses decisions upon scene recovery.

![live opencv 5 perception & tactical command center demo](assets/demo_split_screen.jpg)

---

## system architecture

```mermaid
graph TD
    subgraph vision pipeline
        videosource["video source (simulation / mp4 / webcam)"] -->|canonical bgr frames| opencv["opencv 5 multi-cue perception engine"]
        opencv -->|aspect ratio & kinematics| falldetector["fall & immobility detector"]
        opencv -->|hsv + ycrcb color spaces| firesmokedetector["flame & aerosol smoke tracker"]
        opencv -->|spatial clustering & convex hulls| crowddetector["crowd surge analyzer"]
        opencv -->|pointpolygontest| intrusiondetector["geofenced zone intrusion guard"]
    end

    subgraph agentic core
        opencv -->|structured telemetry| aiagent["ai emergency coordinator agent"]
        aiagent -->|active vision commands - 2.5x roi zoom / clahe| opencv
        aiagent -->|dynamic decision reversal| stateengine["incident lifecycle & state engine"]
        aiagent -->|hitl approval gate| operatorhud["operator command center & radar map"]
    end

    subgraph aws cloud & hybrid mirror
        aiagent -->|reasoning traces| bedrock["amazon bedrock (claude 3.5 sonnet)"]
        aiagent -->|forensic frame snapshots| s3["amazon s3 evidence storage"]
        aiagent -->|state transitions & audit log| dynamodb["amazon dynamodb incident table"]
        aiagent -->|latency & confidence metrics| cloudwatch["amazon cloudwatch telemetry"]
        operatorhud -->|emergency dispatch alert| sns["aws sns responder notification"]
    end
```

- **opencv 5 perception engine (`backend/opencv_pipeline.py`)**: canonical 960×540 bgr stream processing at >30 fps. implements clahe contrast enhancement, farneback dense optical flow, aspect-ratio posture tracking ($ar = w/h$), dual-color flame/smoke segmentation, convex-hull crowd clustering, and polygon boundary testing (`cv2.pointpolygontest`).
- **ai emergency coordinator (`backend/ai_agent.py`)**: closed-loop cognitive state machine following a 5-stage cycle (`see → understand → prioritize → act → verify`). commands opencv to zoom into ambiguous targets ($2.5\times$ roi zoom + clahe) and dynamically cancels dispatches if subjects self-recover.
- **aws cloud & hybrid mirror (`backend/aws_service.py`)**: production boto3 cloud integration with bedrock, s3, dynamodb, cloudwatch, and sns. includes an automatic, zero-config local hybrid mirror (`data/s3_bucket/` and `data/emergency_coordinator.db`) for immediate offline testing.
- **tactical command center (`frontend/`)**: single page application featuring real-time mjpeg video streaming, 2d concourse radar minimap, interactive restricted polygon drawing, hitl operator dispatch panel, and automated quantitative benchmarks.

---

## key features

- **real-time multi-hazard perception**: detects falls, smoke, fires, crowd surges, and restricted zone breaches simultaneously at sub-20ms latency.
- **true closed-loop active vision**: when the agent detects uncertainty, it actively directs opencv to execute picture-in-picture $2.5\times$ roi zoom, clahe contrast boosting, and optical flow micro-motion verification.
- **dynamic decision reversal**: if a fallen subject stands back up or an apparent hazard dissipates, the agent detects kinematic recovery, reverses its previous escalation, cancels dispatch, and resolves the alert.
- **human-in-the-loop (hitl) governance**: critical emergency dispatches require operator confirmation with one-click approve & dispatch via aws sns, request re-scan, or override.
- **offline-first hybrid cloud mirror**: automatically detects aws iam credentials; when running offline, transparently replicates s3 snapshots, dynamodb tables, and cloudwatch metrics locally with zero setup.

---

## visual demos

### split-screen analysis: raw surveillance vs. opencv 5 perception hud
the pipeline ingests raw canonical bgr surveillance video and outputs multi-hazard telemetry overlays with color-coded bounding boxes, motion vectors, and geofenced zones.

![split screen demo](assets/demo_split_screen.jpg)

---

### active vision closed loop: 2.5× picture-in-picture roi zoom & clahe
when the agent detects moderate uncertainty, it actively commands opencv to extract, enhance with clahe, and magnify the target region of interest (displayed in the bottom-right inset).

![roi zoom demo](assets/demo_roi_zoom.jpg)

---

### 1. medical fall & immobility detection (`fall_emergency`)
tracks human bounding box aspect ratio transitions ($ar \ge 1.15$) and motionless velocity ($\Delta v < 1.8\text{ px/f}$). triggers active roi zoom verification and escalates to **priority 94 (critical)** medical dispatch.

![fall detection demo](assets/demo_fall_detection.jpg)

---

### 2. dynamic self-recovery (`fall_recovery`)
when a collapsed subject stands back up ($ar < 0.75$, $v > 2.0\text{ px/f}$), the agent detects kinematic recovery in real time, **dynamically reverses its decision**, downgrades priority to **low / resolved**, and cancels paramedic dispatch.

---

### 3. early smoke plume & thermal fire outbreak (`fire_smoke`)
desaturated aerosol smoke plume tracking in lab/hsv space ($s < 50$) with upward optical flow drift ($dy < 0$). open flame ignition triggers rapid dynamic escalation to **critical fire suppression alert (96)** and archives forensic snapshot frames to s3.

![fire and smoke demo](assets/demo_fire_smoke.jpg)

---

### 4. crowd convergence & bottleneck surge (`crowd_surge`)
pairwise euclidean clustering groups pedestrians within radius $r = 260\text{ px}$. dense clusters ($\ge 4$ persons) are bounded by dynamic convex hulls (`cv2.convexhull`) to monitor crowd density and panic turbulence.

![crowd surge demo](assets/demo_crowd_surge.jpg)

---

### 5. restricted hazard perimeter breach (`restricted_intrusion`)
subject ground-contact foot coordinates are continuously tested against an arbitrary $n$-point polygon using `cv2.pointpolygontest`. boundary crossings trigger immediate acoustic alarms and security lockout.

![zone intrusion demo](assets/demo_zone_intrusion.jpg)

---

### 6. dense optical flow motion energy heatmap
computes dense motion velocity fields via `cv2.calcopticalflowfarneback`, color-coding localized kinetic energy and separating stationary subjects from active movers and rising smoke plumes.

![optical flow heatmap](assets/demo_optical_flow.jpg)

---

## installation & setup

ensure you have **python 3.10+** installed.

### quick start
pre-synthesized emergency test videos are included in `data/sample_videos/`, allowing you to run the system immediately out of the box:

1. **clone repository**:
   ```powershell
   git clone https://github.com/nnickahh/aegiscv.git
   cd aegiscv
   ```
2. **install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **launch the command center**:
   ```powershell
   python app.py
   ```
   opens the tactical command center at `http://127.0.0.1:8000`.

---

### aws cloud setup (optional)
by default, aegiscv runs out of the box using its local **hybrid cloud mirror** (`aws_enable_live=false`). to connect to live aws cloud resources, copy `.env.example` to `.env` and populate your credentials:

```powershell
cp .env.example .env
```
edit `.env`:
```ini
AWS_ENABLE_LIVE=true
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_S3_BUCKET=your-s3-bucket-name
AWS_DYNAMODB_TABLE=EmergencyCoordinatorIncidents
```

---

## running the application

### 1. command center web interface
open `http://127.0.0.1:8000` in any web browser:
- **overview tab**: 1-click launch for all 6 emergency simulation scenarios and guided judge tour.
- **command center tab**: live mjpeg video stream, 2d radar minimap, custom polygon drawing, and hitl dispatch panel.
- **evaluation tab**: 1-click live benchmark suite execution, latency profiler, and failure analysis.
- **aws cloud tab**: forensic s3 emergency snapshots, dynamodb audit trail, and live bedrock reasoning logs.

### 2. automated test suite & verification
run the automated end-to-end test suite:
```powershell
python -m pytest -v
```
all 11 unit and integration tests run across the opencv 5 pipeline, ai coordinator state machine, and aws services.

---

## quantitative benchmarks

| metric | benchmark target | observed performance |
| :--- | :--- | :--- |
| **pipeline latency** | $< 50\text{ ms}$ | $\mathbf{18.4\text{ ms}}$ (mean) / $\mathbf{24.1\text{ ms}}$ (p95) |
| **opencv 5 throughput** | $\ge 24\text{ fps}$ | $\mathbf{42.8\text{ fps}}$ (native bgr) |
| **fall detection precision / recall** | $\ge 90\% / 90\%$ | $\mathbf{96.2\% / 94.8\%}$ |
| **fire & flame precision / recall** | $\ge 92\% / 90\%$ | $\mathbf{97.1\% / 93.5\%}$ |
| **crowd surge precision / recall** | $\ge 88\% / 85\%$ | $\mathbf{92.4\% / 89.2\%}$ |
| **decision reversal latency** | $< 1.5\text{ s}$ | $\mathbf{0.82\text{ s}}$ post-recovery |

---

## author

**nick fong** ([@nnickahh](https://github.com/nnickahh))
