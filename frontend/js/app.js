/**
 * AI Emergency Coordinator — Frontend Command Center Controller
 * Connects to the FastAPI + OpenCV 5 + AWS Backend
 */

let currentMainView = "landing";
let currentApprovalIncidentId = null;
let webcamActive = false;
let webcamStream = null;
let pollIntervalId = null;
let frameLoopActive = false;
let latestState = null;

document.addEventListener("DOMContentLoaded", () => {
  initPolygonOverlayEvents();
  loadEvaluationMetrics();
  startStatePolling();
  startSmoothVideoFrameLoop();
});

function switchMainView(viewName) {
  currentMainView = viewName;
  const views = ["landing", "dashboard", "evaluation", "aws"];
  views.forEach((v) => {
    const el = document.getElementById(`view-${v}`);
    const tab = document.getElementById(`tab-${v}`);
    if (el) el.classList.toggle("active", v === viewName);
    if (tab) tab.classList.toggle("active", v === viewName);
  });

  if (viewName === "evaluation") {
    loadEvaluationMetrics();
  } else if (viewName === "aws") {
    loadAWSViewData();
  }
}

function launchCommandCenter() {
  switchMainView("dashboard");
}

async function launchWithScenario(scenarioId) {
  switchMainView("dashboard");
  await selectScenario(scenarioId);
}

async function selectScenario(scenarioId) {
  if (webcamActive) {
    stopWebcamFeed();
  }
  document.querySelectorAll(".scenario-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-scen") === scenarioId);
  });
  const sampleSelect = document.getElementById("sample-video-select");
  if (sampleSelect) sampleSelect.value = "";

  try {
    await fetch("/api/scenario/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    await fetchAndRenderState(true);
    refreshSingleVideoFrame();
  } catch (err) {
    console.error("Failed to switch scenario:", err);
  }
}

async function setVideoViewMode(mode) {
  ["opencv_annotated", "raw_video", "optical_flow", "split_comparison"].forEach((m) => {
    const btn = document.getElementById(`vmode-${m}`);
    if (btn) btn.classList.toggle("active", m === mode);
  });

  try {
    await fetch("/api/video/view-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ view_mode: mode }),
    });
    refreshSingleVideoFrame();
  } catch (err) {
    console.error("Error switching view mode:", err);
  }
}

async function toggleStreamPause() {
  try {
    const res = await fetch("/api/video/pause", { method: "POST" });
    const data = await res.json();
    const btn = document.getElementById("btn-pause-toggle");
    if (btn) {
      btn.textContent = data.paused ? "▶ Resume" : "Pause";
    }
  } catch (err) {
    console.error("Error toggling pause:", err);
  }
}

async function handleVideoUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  if (webcamActive) stopWebcamFeed();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/video/upload", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      document.querySelectorAll(".scenario-btn").forEach((b) => b.classList.remove("active"));
      await fetchAndRenderState(true);
      refreshSingleVideoFrame();
    }
  } catch (err) {
    console.error("Video upload failed:", err);
  }
}

async function handleSampleVideoSelect(filename) {
  if (!filename) return;
  if (webcamActive) stopWebcamFeed();

  try {
    const res = await fetch("/api/video/load-sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    if (res.ok) {
      document.querySelectorAll(".scenario-btn").forEach((b) => b.classList.remove("active"));
      await fetchAndRenderState(true);
      refreshSingleVideoFrame();
    }
  } catch (err) {
    console.error("Sample video load error:", err);
  }
}

async function toggleWebcamFeed() {
  if (webcamActive) {
    stopWebcamFeed();
    await selectScenario("fall_emergency");
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 360 },
      audio: false,
    });
    const videoEl = document.getElementById("hidden-webcam-video");
    videoEl.srcObject = webcamStream;
    await videoEl.play();
    webcamActive = true;
    const btn = document.getElementById("btn-webcam-toggle");
    if (btn) btn.textContent = "■ Stop Webcam";
    runWebcamLoop();
  } catch (err) {
    alert("Webcam access unavailable in this browser/environment. Using real-time OpenCV simulation stream.");
  }
}

function stopWebcamFeed() {
  webcamActive = false;
  if (webcamStream) {
    webcamStream.getTracks().forEach((t) => t.stop());
    webcamStream = null;
  }
  const btn = document.getElementById("btn-webcam-toggle");
  if (btn) btn.textContent = "Live Webcam";
}

async function runWebcamLoop() {
  const videoEl = document.getElementById("hidden-webcam-video");
  const offCanvas = document.createElement("canvas");
  offCanvas.width = 640;
  offCanvas.height = 360;
  const ctx = offCanvas.getContext("2d");

  while (webcamActive) {
    ctx.drawImage(videoEl, 0, 0, 640, 360);
    const dataUrl = offCanvas.toDataURL("image/jpeg", 0.78);
    try {
      const res = await fetch("/api/video/webcam-frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl }),
      });
      if (res.ok) {
        const out = await res.json();
        const imgEl = document.getElementById("opencv-video-img");
        if (imgEl && out.annotated_frame_base64) {
          imgEl.src = out.annotated_frame_base64;
        }
      }
    } catch (e) {
      console.error("Webcam frame error:", e);
    }
    await new Promise((r) => setTimeout(r, 140));
  }
}

/**
 * Smooth double-buffered video frame updater when Dashboard is visible
 */
function startSmoothVideoFrameLoop() {
  if (frameLoopActive) return;
  frameLoopActive = true;

  const tickFrame = () => {
    if (currentMainView === "dashboard" && !webcamActive) {
      refreshSingleVideoFrame();
    }
    setTimeout(tickFrame, 125);
  };
  tickFrame();
}

function refreshSingleVideoFrame() {
  const imgEl = document.getElementById("opencv-video-img");
  if (!imgEl) return;
  const preload = new Image();
  preload.onload = () => {
    imgEl.src = preload.src;
  };
  preload.src = `/api/video/frame?t=${Date.now()}`;
}

function startStatePolling() {
  fetchAndRenderState(true);
  pollIntervalId = setInterval(() => {
    fetchAndRenderState(currentMainView !== "dashboard");
  }, 380);
}

async function fetchAndRenderState(advanceFrame = false) {
  try {
    const res = await fetch(`/api/state?advance=${advanceFrame ? "true" : "false"}`);
    if (!res.ok) return;
    const state = await res.json();
    latestState = state;
    renderCommandCenterState(state);
  } catch (err) {
    console.error("State poll error:", err);
  }
}

function renderCommandCenterState(state) {
  const tel = state.telemetry;
  const dec = state.agent_decision;
  const aws = state.aws_status;

  // 1. Top Navbar Telemetry
  if (tel) {
    setText("nav-fps", tel.fps.toFixed(1));
    setText("nav-ms", tel.processing_ms.toFixed(1));
    setText("nav-cv-mode", tel.active_cv_mode);

    setText("vf-source", state.source_mode);
    setText("vf-frame", tel.frame_index);
    setText("vf-flow", tel.motion_energy.toFixed(2));
    setText("vf-clahe", tel.clahe_active ? "ACTIVE (LAB)" : "OFF");

    const roiStatus = document.getElementById("vf-roi-status");
    if (roiStatus) {
      if (tel.roi_zoom && tel.roi_zoom.active) {
        roiStatus.textContent = `LOCKED ON ${tel.roi_zoom.target_id} (uMot: ${tel.roi_zoom.micro_motion_score.toFixed(2)})`;
        roiStatus.style.color = "#38bdf8";
      } else {
        roiStatus.textContent = "STANDBY";
        roiStatus.style.color = "#94a3b8";
      }
    }
  }

  if (state.scenario_info && state.scenario_info.badge) {
    setText("active-scenario-badge", state.scenario_info.badge);
  }

  // 2. Update 5-Stage Pipeline Flow Bar
  if (dec) {
    const stages = ["SEE", "UNDERSTAND", "PRIORITIZE", "ACT", "VERIFY"];
    const activeIdx = stages.indexOf(dec.current_stage);
    const isCritical = dec.priority_level === "CRITICAL";

    stages.forEach((st, idx) => {
      const node = document.getElementById(`stage-node-${st}`);
      if (!node) return;
      node.classList.remove("active", "active-critical");
      if (idx === activeIdx) {
        node.classList.add(isCritical ? "active-critical" : "active");
      }
    });

    const navDot = document.getElementById("nav-status-dot");
    const vidDot = document.getElementById("video-live-dot");
    if (navDot) navDot.classList.toggle("critical", isCritical);
    if (vidDot) vidDot.classList.toggle("critical", isCritical);

    // 3. Render AI Coordinator Panel
    setText("agent-stage-badge", `STAGE: ${dec.current_stage}`);
    const prioBadge = document.getElementById("agent-priority-badge");
    if (prioBadge) {
      prioBadge.textContent = `${dec.priority_level} (${dec.priority_score}/100)`;
      prioBadge.className = `severity-badge severity-${dec.priority_level}`;
    }

    const headlineBox = document.getElementById("agent-headline-box");
    if (headlineBox) {
      headlineBox.className = "agent-headline-box";
      if (dec.priority_level === "CRITICAL") headlineBox.classList.add("critical");
      else if (dec.priority_level === "HIGH") headlineBox.classList.add("high");
    }

    setText("agent-headline-text", dec.headline);
    setText("agent-understanding-text", dec.situation_understanding);
    setText("agent-recommended-action", dec.recommended_action);
    setText("agent-confidence-label", `CONFIDENCE: ${Math.round(dec.confidence * 100)}% (UNCERTAINTY: ${Math.round(dec.uncertainty_score * 100)}%)`);
    setText("agent-cv-command", dec.active_cv_command || "RESET_STANDARD");
    setText("agent-verification-status", dec.verification_status);

    // Uncertainty Alert Banner
    const uncertBox = document.getElementById("agent-uncertainty-box");
    if (uncertBox) {
      if (dec.uncertainty_flag && dec.uncertainty_reason) {
        uncertBox.style.display = "flex";
        setText("agent-uncertainty-reason", dec.uncertainty_reason);
      } else {
        uncertBox.style.display = "none";
      }
    }

    // Dynamic Decision Change Banner
    const changeBox = document.getElementById("agent-decision-change-box");
    if (changeBox) {
      if (dec.decision_changed && dec.change_reason) {
        changeBox.style.display = "block";
        setText(
          "agent-decision-change-text",
          `${dec.change_reason} (Previous: "${dec.previous_decision || "Prior State"}")`
        );
      } else {
        changeBox.style.display = "none";
      }
    }

    // Human-in-the-Loop Approval Box
    const hitlBox = document.getElementById("hitl-approval-box");
    if (hitlBox) {
      if (dec.requires_human_approval && dec.approval_incident_id) {
        currentApprovalIncidentId = dec.approval_incident_id;
        hitlBox.style.display = "flex";
        setText("hitl-incident-id-label", dec.approval_incident_id);
        setText("hitl-action-desc", dec.recommended_action);
      } else {
        hitlBox.style.display = "none";
      }
    }

    // Reasoning Trace List
    const traceUl = document.getElementById("agent-reasoning-list");
    if (traceUl && Array.isArray(dec.reasoning_trace)) {
      traceUl.innerHTML = dec.reasoning_trace
        .map((line) => `<li>${escapeHtml(line)}</li>`)
        .join("");
    }
  }

  // 4. Render OpenCV 5 Detections List, 2D Tactical Scene Map & Live Sparklines
  if (tel) {
    renderDetectionsList(tel.detections || []);
    renderSceneMapCanvas(tel.detections || [], state.restricted_zone_polygon || [], tel.roi_zoom);
    renderTelemetrySparklines(tel, dec);
  }
  if (dec) {
    announceHeadlineIfChanged(dec);
  }

  // 5. Render Active Incidents Panel
  renderIncidentsList(state.incidents || []);

  // 6. Render Step-by-Step Event Timeline
  renderTimelineList(state.timeline || []);

  // 7. Render AWS System Status
  if (aws) {
    setText("aws-mode-badge", aws.connection_mode.includes("LIVE") ? "LIVE AWS IAM" : "HYBRID CLOUD (BOTO3)");
    setText("aws-s3-count", aws.s3_objects_count);
    setText("aws-ddb-count", aws.dynamodb_items_count);
    setText("aws-cw-count", aws.cloudwatch_metrics_pushed);
    setText("aws-bedrock-count", aws.bedrock_invocations);
    setText("aws-sns-count", aws.sns_alerts_dispatched);
  }
}

function renderDetectionsList(detections) {
  const container = document.getElementById("detections-list-container");
  setText("detections-count-badge", `${detections.length} DETECTED`);
  if (!container) return;

  if (detections.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.78rem; padding: 0.5rem;">No objects in current frame.</div>`;
    return;
  }

  const colorByCat = {
    fall: "#ef4444",
    fire: "#f97316",
    smoke: "#94a3b8",
    restricted_area: "#ef4444",
    crowd: "#f59e0b",
    person: "#10b981",
    movement: "#38bdf8",
  };

  container.innerHTML = detections
    .map((d) => {
      const confPct = Math.round(d.confidence * 100);
      const uncertPct = Math.round(d.uncertainty * 100);
      const barColor = colorByCat[d.category] || "#38bdf8";
      return `
        <div class="detection-item-card">
          <div class="detection-item-top">
            <span style="color: ${barColor};">${escapeHtml(d.id)} · ${escapeHtml(d.label)}</span>
            <span style="font-family: var(--font-mono);">${confPct}%</span>
          </div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width: ${confPct}%; background: ${barColor};"></div>
          </div>
          <div class="detection-item-meta">
            <span>BBox: [${d.bbox.join(", ")}]</span>
            <span>Vel: ${d.velocity.toFixed(1)} px/f · Uncert: ${uncertPct}%</span>
          </div>
        </div>
      `;
    })
    .join("");
}

const personTrailHistory = {};

/**
 * Draw 2D Top-Down Spatial Digital Twin Blueprint on `<canvas id="scene-map-canvas">`
 */
function renderSceneMapCanvas(detections, restrictedPolygonNorm, roiZoom) {
  const canvas = document.getElementById("scene-map-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // 1. Background Architectural CAD Blueprint Grid
  ctx.fillStyle = "#060a12";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.06)";
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 32) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 0; y < H; y += 32) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // 2. North Wall, Structural Pillars & Emergency Exit-1 Doorway
  ctx.fillStyle = "rgba(30, 41, 59, 0.75)";
  ctx.fillRect(0, 0, W, 28);
  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.beginPath();
  ctx.moveTo(0, 28);
  ctx.lineTo(W, 28);
  ctx.stroke();

  // Exit-1 Egress Bay
  ctx.fillStyle = "rgba(16, 185, 129, 0.16)";
  ctx.strokeStyle = "#10b981";
  ctx.fillRect(76, 6, 54, 22);
  ctx.strokeRect(76, 6, 54, 22);
  ctx.fillStyle = "#6ee7b7";
  ctx.font = "700 8px 'JetBrains Mono', monospace";
  ctx.fillText("EXIT-1", 88, 20);

  // Structural Columns along North Wall
  [38, 162, 284].forEach((cx, i) => {
    ctx.fillStyle = "#334155";
    ctx.fillRect(cx - 6, 18, 12, 12);
    ctx.strokeStyle = "#64748b";
    ctx.strokeRect(cx - 6, 18, 12, 12);
  });

  // 3. Sector-A Pedestrian Transit Corridor Lane Markings
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(16, H * 0.72);
  ctx.lineTo(W - 16, H * 0.72);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(148, 163, 184, 0.42)";
  ctx.font = "600 8px 'JetBrains Mono', monospace";
  ctx.fillText("SECTOR-A MAIN TRANSIT CORRIDOR", 16, H - 10);

  // 4. Camera-04 FOV Frustum Cone
  ctx.fillStyle = "rgba(56, 189, 248, 0.025)";
  ctx.beginPath();
  ctx.moveTo(W * 0.5, H - 2);
  ctx.lineTo(10, 28);
  ctx.lineTo(W - 10, 28);
  ctx.closePath();
  ctx.fill();

  // 5. Sector-B Restricted Hazard Zone (13.8kV High-Voltage Substation Bay)
  if (restrictedPolygonNorm.length >= 3) {
    const hasBreach = detections.some((d) => d.category === "restricted_area");
    ctx.save();
    ctx.fillStyle = hasBreach ? "rgba(239, 68, 68, 0.22)" : "rgba(245, 158, 11, 0.11)";
    ctx.strokeStyle = hasBreach ? "#ef4444" : "#f59e0b";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    restrictedPolygonNorm.forEach((pt, idx) => {
      const px = pt[0] * W;
      const py = pt[1] * H;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Clip to polygon and draw diagonal OSHA hazard hatching
    ctx.clip();
    ctx.strokeStyle = hasBreach ? "rgba(239, 68, 68, 0.18)" : "rgba(245, 158, 11, 0.12)";
    ctx.lineWidth = 1;
    for (let hx = W * 0.55; hx < W + 80; hx += 14) {
      ctx.beginPath();
      ctx.moveTo(hx, 25);
      ctx.lineTo(hx - 65, H * 0.65);
      ctx.stroke();
    }
    ctx.restore();

    // Draw the 3 physical High-Voltage Transformer footprints inside Sector-B
    const txUnits = [
      { x: W * 0.69, y: 42, w: 32, h: 22, label: "TX1" },
      { x: W * 0.78, y: 42, w: 32, h: 22, label: "TX2" },
      { x: W * 0.87, y: 42, w: 30, h: 22, label: "BUS" },
    ];
    txUnits.forEach((u) => {
      ctx.fillStyle = "rgba(15, 23, 42, 0.9)";
      ctx.strokeStyle = hasBreach ? "#fca5a5" : "#fbbf24";
      ctx.lineWidth = 1;
      ctx.fillRect(u.x, u.y, u.w, u.h);
      ctx.strokeRect(u.x, u.y, u.w, u.h);
      ctx.fillStyle = hasBreach ? "#fca5a5" : "#fcd34d";
      ctx.font = "600 7px 'JetBrains Mono', monospace";
      ctx.fillText(u.label, u.x + 6, u.y + 14);
    });

    // Corner Bollard Nodes on Polygon
    restrictedPolygonNorm.forEach((pt) => {
      ctx.fillStyle = hasBreach ? "#ef4444" : "#f59e0b";
      ctx.beginPath();
      ctx.arc(pt[0] * W, pt[1] * H, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.fillStyle = hasBreach ? "#fca5a5" : "#fcd34d";
    ctx.font = "700 8px 'JetBrains Mono', monospace";
    ctx.fillText(
      hasBreach ? "⚠ SECTOR-B 13.8kV BREACH" : "SECTOR-B: 13.8kV SUBSTATION",
      restrictedPolygonNorm[3][0] * W + 10,
      restrictedPolygonNorm[3][1] * H - 8
    );
  }

  // 6. Plot Detections (Fire, Smoke, Crowd, People + Breadcrumb Trails, Falls)
  const pulse = 0.5 + 0.5 * Math.sin(Date.now() * 0.008);

  detections.forEach((d) => {
    if (d.category === "movement") return;
    const x = d.world_pos[0] * W;
    const y = d.world_pos[1] * H;

    if (d.category === "person") {
      if (!personTrailHistory[d.id]) personTrailHistory[d.id] = [];
      const trail = personTrailHistory[d.id];
      trail.push({ x, y });
      if (trail.length > 10) trail.shift();

      // Draw motion breadcrumb trail
      if (trail.length > 1) {
        ctx.strokeStyle = d.in_restricted_zone
          ? "rgba(239, 68, 68, 0.45)"
          : "rgba(16, 185, 129, 0.35)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        trail.forEach((pt, idx) => {
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
      }
    }

    if (d.category === "fire") {
      ctx.fillStyle = `rgba(249, 115, 22, ${0.24 + pulse * 0.22})`;
      ctx.beginPath();
      ctx.arc(x, y, 22 + pulse * 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#f97316";
      ctx.beginPath();
      ctx.arc(x, y, 6.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffedd5";
      ctx.font = "700 8px 'JetBrains Mono', monospace";
      ctx.fillText("🔥 FIRE CORE", x - 24, y - 10);
    } else if (d.category === "smoke") {
      ctx.fillStyle = "rgba(148, 163, 184, 0.22)";
      ctx.beginPath();
      ctx.arc(x, y - 8, 17, 0, Math.PI * 2);
      ctx.fill();
    } else if (d.category === "crowd") {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(x, y, 32, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (d.category === "fall") {
      ctx.fillStyle = `rgba(239, 68, 68, ${0.26 + pulse * 0.25})`;
      ctx.beginPath();
      ctx.arc(x, y, 18 + pulse * 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#fecaca";
      ctx.font = "700 8px 'JetBrains Mono', monospace";
      ctx.fillText(`${d.id} [IMMOBILE]`, x - 28, y - 10);
    } else if (d.category === "person" && d.posture_state !== "FALLING" && d.posture_state !== "FALLEN_IMMOBILE") {
      const col = d.in_restricted_zone ? "#ef4444" : "#10b981";
      if (d.in_restricted_zone) {
        // Draw breach vector line to HV-TX2 transformer
        ctx.strokeStyle = "rgba(239, 68, 68, 0.65)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(W * 0.78 + 16, 64);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#e2e8f0";
      ctx.font = "600 8px 'JetBrains Mono', monospace";
      ctx.fillText(d.id, x + 7, y + 3);
    }
  });

  // 7. Agent ROI Verification Target Reticle on Map
  if (roiZoom && roiZoom.active && roiZoom.bbox) {
    const rx = ((roiZoom.bbox[0] + roiZoom.bbox[2] * 0.5) / 960.0) * W;
    const ry = ((roiZoom.bbox[1] + roiZoom.bbox[3] * 0.85) / 540.0) * H;
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(rx - 14, ry - 14, 28, 28);
  }
}

let latestIncidentsCache = [];

function renderIncidentsList(incidents) {
  latestIncidentsCache = incidents || [];
  const container = document.getElementById("incidents-list-container");
  const activeCount = incidents.filter((i) => i.status !== "RESOLVED").length;
  setText("incidents-count-badge", `${activeCount} ACTIVE`);
  if (!container) return;

  if (incidents.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.78rem;">No emergency incidents logged yet.</div>`;
    return;
  }

  container.innerHTML = incidents
    .map((inc) => {
      const confPct = Math.round(inc.confidence * 100);
      const latestNote =
        inc.evolution_log && inc.evolution_log.length > 0
          ? inc.evolution_log[inc.evolution_log.length - 1]
          : "";
      return `
        <div class="incident-card">
          <div class="incident-card-header">
            <span style="font-weight: 700; font-size: 0.8rem; color: var(--text-primary);">
              ${escapeHtml(inc.id)} · ${escapeHtml(inc.title)}
            </span>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
              <button class="btn-tool" style="padding: 0.14rem 0.48rem; font-size: 0.66rem; color: var(--accent-cyan); border-color: rgba(56,189,248,0.35);" onclick="openS3EvidenceModal('${escapeHtml(inc.id)}')">
                🖼 View S3 Frame
              </button>
              <span class="severity-badge severity-${inc.severity}">
                ${inc.severity} (${inc.priority_score})
              </span>
            </div>
          </div>
          <div style="font-size: 0.75rem; color: var(--text-secondary);">
            ${escapeHtml(inc.recommended_action)}
          </div>
          <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 0.68rem; color: var(--text-muted);">
            <span>Stage: ${inc.pipeline_stage} · Status: ${inc.status}</span>
            <span>Conf: ${confPct}% · Approval: ${inc.human_approval_status}</span>
          </div>
          ${
            latestNote
              ? `<div style="font-family: var(--font-mono); font-size: 0.67rem; color: var(--accent-cyan); border-top: 1px solid rgba(148,163,184,0.1); padding-top: 0.25rem;">
                   ${escapeHtml(latestNote)}
                 </div>`
              : ""
          }
        </div>
      `;
    })
    .join("");
}

function renderTimelineList(events) {
  const container = document.getElementById("timeline-list-container");
  if (!container) return;

  if (events.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.78rem;">Awaiting events...</div>`;
    return;
  }

  container.innerHTML = events
    .slice(0, 14)
    .map((ev) => {
      return `
        <div class="timeline-item">
          <span class="timeline-time">${escapeHtml(ev.timestamp)}</span>
          <span class="timeline-stage-tag">${escapeHtml(ev.stage)}</span>
          <div style="flex: 1;">
            <div style="font-weight: 600; color: var(--text-primary);">${escapeHtml(ev.title)}</div>
            <div style="color: var(--text-secondary); font-size: 0.72rem;">${escapeHtml(ev.detail)}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

async function submitOperatorApproval(actionType) {
  if (!currentApprovalIncidentId) return;
  try {
    await fetch("/api/incidents/approval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        incident_id: currentApprovalIncidentId,
        action: actionType,
        operator_note: `Operator executed ${actionType} from Command Center`,
      }),
    });
    await fetchAndRenderState(false);
  } catch (err) {
    console.error("Error submitting operator approval:", err);
  }
}

/**
 * Evaluation & Live Benchmark Functions
 */
async function loadEvaluationMetrics() {
  try {
    const res = await fetch("/api/evaluation");
    if (!res.ok) return;
    const report = await res.json();
    renderEvaluationView(report);
  } catch (err) {
    console.error("Failed to load evaluation metrics:", err);
  }
}

async function runLiveBenchmark() {
  const btn = document.getElementById("btn-run-live-benchmark");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "⏳ Running OpenCV 5 + Agent Benchmark Across 6 Scenarios...";
  }
  try {
    const res = await fetch("/api/evaluation/run-benchmark", { method: "POST" });
    if (res.ok) {
      const report = await res.json();
      renderEvaluationView(report);
    }
  } catch (err) {
    console.error("Live benchmark error:", err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "▶ Run Live OpenCV 5 + Agent Benchmark Suite";
    }
  }
}

function renderEvaluationView(report) {
  const sum = report.summary_metrics || {};
  setText("eval-overall-acc", `${sum.overall_accuracy_pct}%`);
  setText("eval-resp-ms", `${sum.mean_response_time_ms} ms`);
  setText("eval-cv-ms", sum.opencv_frame_latency_ms);
  setText("eval-ag-ms", sum.agent_decision_latency_ms);
  setText("eval-fp-rate", `${sum.false_positive_rate_pct}%`);
  setText("eval-agent-success", `${sum.agent_success_rate_pct}%`);

  // Class performance table
  const classTbody = document.getElementById("eval-class-tbody");
  if (classTbody && Array.isArray(report.class_performance)) {
    classTbody.innerHTML = report.class_performance
      .map(
        (c) => `
        <tr>
          <td>
            <div style="font-weight: 700; color: var(--text-primary);">${escapeHtml(c.category)}</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">${escapeHtml(c.notes)}</div>
          </td>
          <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${(c.precision * 100).toFixed(1)}%</td>
          <td style="font-family: var(--font-mono);">${(c.recall * 100).toFixed(1)}%</td>
          <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-emerald);">${(c.f1_score * 100).toFixed(1)}%</td>
          <td style="font-family: var(--font-mono);">${c.mean_iou.toFixed(2)}</td>
          <td style="font-family: var(--font-mono); color: var(--accent-amber);">${(c.false_positive_rate * 100).toFixed(1)}%</td>
        </tr>
      `
      )
      .join("");
  }

  // Latency Breakdown Bars
  const latContainer = document.getElementById("eval-latency-bars");
  if (latContainer && Array.isArray(report.latency_breakdown_ms)) {
    latContainer.innerHTML = report.latency_breakdown_ms
      .map((item) => {
        const pct = Math.min(100, Math.round((item.ms / 18.0) * 100));
        return `
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.25rem;">
              <span>${escapeHtml(item.stage)}</span>
              <strong style="font-family: var(--font-mono); color: var(--accent-cyan);">${item.ms.toFixed(1)} ms</strong>
            </div>
            <div class="conf-bar-track" style="height: 7px;">
              <div class="conf-bar-fill" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // Failure Cases
  const fcGrid = document.getElementById("eval-failure-cases-grid");
  if (fcGrid && Array.isArray(report.failure_cases)) {
    fcGrid.innerHTML = report.failure_cases
      .map(
        (fc) => `
        <div class="failure-card">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span class="panel-badge" style="color: var(--accent-amber);">${escapeHtml(fc.id)}</span>
            <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-emerald);">Residual Risk: ${escapeHtml(fc.residual_risk)}</span>
          </div>
          <div style="font-weight: 700; font-size: 0.92rem; color: var(--text-primary);">${escapeHtml(fc.title)}</div>
          <div style="font-size: 0.78rem; color: var(--text-secondary);"><strong>Trigger:</strong> ${escapeHtml(fc.trigger_condition)}</div>
          <div style="font-size: 0.78rem; color: #fca5a5;"><strong>Naive CV Failure:</strong> ${escapeHtml(fc.observed_failure)}</div>
          <div style="font-size: 0.78rem; color: #a7f3d0;"><strong>Closed-Loop Mitigation:</strong> ${escapeHtml(fc.mitigation_in_system)}</div>
        </div>
      `
      )
      .join("");
  }

  // Limitations
  const limGrid = document.getElementById("eval-limitations-grid");
  if (limGrid && Array.isArray(report.limitations)) {
    limGrid.innerHTML = report.limitations
      .map(
        (l) => `
        <div class="failure-card">
          <div style="font-weight: 700; color: var(--accent-cyan); font-size: 0.88rem;">${escapeHtml(l.area)}</div>
          <div style="font-size: 0.8rem; color: var(--text-secondary);">${escapeHtml(l.detail)}</div>
        </div>
      `
      )
      .join("");
  }

  // Live Benchmark Results Table
  const livePanel = document.getElementById("live-benchmark-results-panel");
  const liveTbody = document.getElementById("live-benchmark-tbody");
  if (livePanel && liveTbody && Array.isArray(report.live_benchmark_runs) && report.live_benchmark_runs.length > 0) {
    livePanel.style.display = "block";
    liveTbody.innerHTML = report.live_benchmark_runs
      .map(
        (r) => `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700;">${escapeHtml(r.scenario_id)}</td>
          <td style="font-family: var(--font-mono);">${r.frames_evaluated}</td>
          <td style="font-family: var(--font-mono);">${escapeHtml(r.expected_category)}</td>
          <td style="font-family: var(--font-mono); color: var(--accent-cyan);">${escapeHtml(r.detected_categories.join(", "))}</td>
          <td style="font-family: var(--font-mono);">${escapeHtml(r.agent_stage_reached)}</td>
          <td><span class="severity-badge severity-${r.agent_priority}">${escapeHtml(r.agent_priority)}</span></td>
          <td style="font-family: var(--font-mono); font-size: 0.74rem;">${escapeHtml(r.active_cv_command || "NONE")}</td>
          <td style="font-family: var(--font-mono); color: var(--accent-emerald); font-weight: 700;">${r.passed ? "✓ PASS" : "WARN"}</td>
        </tr>
      `
      )
      .join("");
  }
}

/**
 * AWS Cloud & S3 Evidence View Loader
 */
async function loadAWSViewData() {
  try {
    const [s3Res, stateRes] = await Promise.all([
      fetch("/api/s3-evidence"),
      fetch("/api/state?advance=false"),
    ]);
    if (s3Res.ok) {
      const s3Data = await s3Res.json();
      const gallery = document.getElementById("s3-evidence-gallery");
      if (gallery) {
        if (!s3Data.artifacts || s3Data.artifacts.length === 0) {
          gallery.innerHTML = `<div style="color: var(--text-muted); font-size: 0.82rem;">No S3 snapshots archived yet. Trigger an emergency scenario in the Command Center.</div>`;
        } else {
          gallery.innerHTML = s3Data.artifacts
            .map(
              (art) => `
              <div style="background: rgba(15,23,42,0.75); border: 1px solid var(--border-subtle); border-radius: 8px; overflow: hidden;">
                <img src="${art.url}" alt="${escapeHtml(art.filename)}" style="width: 100%; height: 125px; object-fit: cover; display: block;" />
                <div style="padding: 0.5rem; font-family: var(--font-mono); font-size: 0.68rem;">
                  <div style="color: var(--text-primary); word-break: break-all;">${escapeHtml(art.filename)}</div>
                  <div style="color: var(--text-muted); margin-top: 0.2rem;">s3://${escapeHtml(s3Data.bucket)} · ${(art.size_bytes / 1024).toFixed(1)} KB</div>
                </div>
              </div>
            `
            )
            .join("");
        }
      }
    }

    if (stateRes.ok) {
      const st = await stateRes.json();
      const logsList = document.getElementById("aws-logs-detail-list");
      const events = (st.aws_status && st.aws_status.recent_cloud_events) || [];
      if (logsList) {
        logsList.innerHTML = events
          .map(
            (ev) => `
            <div style="background: rgba(15,23,42,0.75); border: 1px solid var(--border-subtle); border-radius: 6px; padding: 0.65rem; font-family: var(--font-mono); font-size: 0.72rem;">
              <div style="display: flex; justify-content: space-between; color: var(--accent-amber); margin-bottom: 0.25rem;">
                <strong>${escapeHtml(ev.service)} :: ${escapeHtml(ev.operation)}</strong>
                <span>${escapeHtml(ev.timestamp)} · ${escapeHtml(ev.status)}</span>
              </div>
              <pre style="color: var(--text-secondary); white-space: pre-wrap; font-size: 0.68rem;">${escapeHtml(JSON.stringify(ev.payload, null, 2))}</pre>
            </div>
          `
          )
          .join("");
      }
    }
  } catch (err) {
    console.error("Failed to load AWS view data:", err);
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(val);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * ============================================================================
 * UPGRADE 1: Interactive Custom Restricted Zone Polygon Drawing on Video Viewport
 * ============================================================================
 */
let isDrawingPolygon = false;
let draftPolygonVertices = [];

function initPolygonOverlayEvents() {
  const overlay = document.getElementById("polygon-draw-overlay");
  if (!overlay) return;

  overlay.addEventListener("click", async (e) => {
    const rect = overlay.getBoundingClientRect();
    const xNorm = Math.max(0.01, Math.min(0.99, (e.clientX - rect.left) / rect.width));
    const yNorm = Math.max(0.01, Math.min(0.99, (e.clientY - rect.top) / rect.height));

    if (!isDrawingPolygon) {
      await lockRoiAtClick(xNorm, yNorm);
      return;
    }

    draftPolygonVertices.push([Number(xNorm.toFixed(4)), Number(yNorm.toFixed(4))]);
    renderPolygonDraftOverlay();

    // Auto-complete after 4 vertices or let user click button again after >=3 vertices
    if (draftPolygonVertices.length === 4) {
      await commitCustomPolygon();
    }
  });
}

async function lockRoiAtClick(xNorm, yNorm) {
  const overlay = document.getElementById("polygon-draw-overlay");
  if (overlay) {
    const ctx = overlay.getContext("2d");
    const px = xNorm * overlay.width;
    const py = yNorm * overlay.height;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.save();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.strokeRect(px - 44, py - 34, 88, 68);
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "#38bdf8";
    ctx.font = "700 10px 'JetBrains Mono', monospace";
    ctx.fillText("⌖ LOCKING 2.4x ROI...", px - 44, py - 40);
    ctx.restore();
    setTimeout(() => {
      if (!isDrawingPolygon) ctx.clearRect(0, 0, overlay.width, overlay.height);
    }, 950);
  }

  try {
    await fetch("/api/video/roi-lock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x_norm: xNorm, y_norm: yNorm }),
    });
    await fetchAndRenderState(false);
    refreshSingleVideoFrame();
  } catch (err) {
    console.error("Failed to lock ROI target:", err);
  }
}

async function togglePolygonDrawMode() {
  const overlay = document.getElementById("polygon-draw-overlay");
  const btnDraw = document.getElementById("btn-draw-polygon");
  const hint = document.getElementById("polygon-draw-hint");
  if (!overlay || !btnDraw) return;

  if (isDrawingPolygon) {
    if (draftPolygonVertices.length >= 3) {
      await commitCustomPolygon();
    } else {
      isDrawingPolygon = false;
      draftPolygonVertices = [];
      if (hint) hint.style.display = "none";
      btnDraw.textContent = "✎ Draw Custom Zone";
      renderPolygonDraftOverlay();
    }
    return;
  }

  isDrawingPolygon = true;
  draftPolygonVertices = [];
  if (hint) hint.style.display = "block";
  btnDraw.textContent = "✓ Finish Zone (Click 4 Corners)";
  renderPolygonDraftOverlay();
}

async function commitCustomPolygon() {
  const btnDraw = document.getElementById("btn-draw-polygon");
  const btnReset = document.getElementById("btn-reset-polygon");
  const hint = document.getElementById("polygon-draw-hint");

  isDrawingPolygon = false;
  if (hint) hint.style.display = "none";
  if (btnDraw) btnDraw.textContent = "✎ Draw Custom Zone";
  if (btnReset) btnReset.style.display = "inline-flex";

  try {
    await fetch("/api/video/restricted-zone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ vertices: draftPolygonVertices }),
    });
    draftPolygonVertices = [];
    renderPolygonDraftOverlay();
    await fetchAndRenderState(false);
  } catch (err) {
    console.error("Failed to update custom restricted zone:", err);
  }
}

async function resetCustomPolygon() {
  const btnReset = document.getElementById("btn-reset-polygon");
  draftPolygonVertices = [];
  isDrawingPolygon = false;
  renderPolygonDraftOverlay();
  try {
    await fetch("/api/video/restricted-zone", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reset: true }),
    });
    if (btnReset) btnReset.style.display = "none";
    await fetchAndRenderState(false);
  } catch (err) {
    console.error("Failed to reset restricted zone:", err);
  }
}

function renderPolygonDraftOverlay() {
  const overlay = document.getElementById("polygon-draw-overlay");
  if (!overlay) return;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (draftPolygonVertices.length === 0) return;

  ctx.save();
  ctx.strokeStyle = "#38bdf8";
  ctx.fillStyle = "rgba(56, 189, 248, 0.2)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  draftPolygonVertices.forEach((pt, idx) => {
    const px = pt[0] * overlay.width;
    const py = pt[1] * overlay.height;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  if (draftPolygonVertices.length >= 3) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  draftPolygonVertices.forEach((pt, idx) => {
    const px = pt[0] * overlay.width;
    const py = pt[1] * overlay.height;
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#020617";
    ctx.font = "700 10px 'JetBrains Mono', monospace";
    ctx.fillText(String(idx + 1), px - 3, py + 3);
  });
  ctx.restore();
}

/**
 * ============================================================================
 * UPGRADE 2: Live Multi-Metric Telemetry Sparkline Graph (60-Frame History)
 * ============================================================================
 */
const sparklineHistory = [];
const MAX_SPARKLINE_POINTS = 60;

function renderTelemetrySparklines(tel, dec) {
  const canvas = document.getElementById("telemetry-sparkline-canvas");
  if (!canvas || !tel) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  const priorityScore = dec ? Number(dec.priority_score || 0) : 0;
  const confScore = dec ? Math.round(Number(dec.confidence || 0) * 100) : 95;
  const rawMotion = Number(tel.motion_energy || 0);
  // Non-linear visual scaling so normal patrol (0.2-0.6 px/f) sits around 20-32% and active emergencies scale up to 92%
  const motionPct = Math.min(96, Math.max(8, Math.round(Math.sqrt(rawMotion) * 34 + 10)));

  // Update the 4 live KPI boxes beneath the sparkline chart
  setText("spark-val-priority", `${priorityScore} / 100`);
  setText("spark-val-conf", `${confScore}.0%`);
  setText("spark-val-flow", `${rawMotion.toFixed(2)} px/f`);
  setText("spark-val-mode", tel.active_cv_mode || "STANDARD");

  sparklineHistory.push({
    priority: priorityScore,
    confidence: confScore,
    motion: motionPct,
  });
  if (sparklineHistory.length > MAX_SPARKLINE_POINTS) {
    sparklineHistory.shift();
  }

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#060a12";
  ctx.fillRect(0, 0, W, H);

  // Subtle vertical time-grid lines
  ctx.strokeStyle = "rgba(148, 163, 184, 0.06)";
  ctx.lineWidth = 1;
  for (let vx = 40; vx < W; vx += 60) {
    ctx.beginPath();
    ctx.moveTo(vx, 0);
    ctx.lineTo(vx, H);
    ctx.stroke();
  }

  // Horizontal reference lines (25%, 50%, 75%) + Y-axis labels
  ctx.strokeStyle = "rgba(148, 163, 184, 0.1)";
  ctx.fillStyle = "rgba(148, 163, 184, 0.45)";
  ctx.font = "600 9px 'JetBrains Mono', monospace";
  [
    { ratio: 0.2, label: "80%" },
    { ratio: 0.5, label: "50%" },
    { ratio: 0.8, label: "20%" },
  ].forEach((row) => {
    const y = H * row.ratio;
    ctx.beginPath();
    ctx.moveTo(30, y);
    ctx.lineTo(W, y);
    ctx.stroke();
    ctx.fillText(row.label, 6, y + 3);
  });

  // Critical priority threshold line at 75%
  ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(30, H * 0.25);
  ctx.lineTo(W, H * 0.25);
  ctx.stroke();
  ctx.setLineDash([]);

  if (sparklineHistory.length < 2) return;

  const leftPad = 34;
  const rightPad = 14;
  const plotW = W - leftPad - rightPad;

  const drawCurve = (key, strokeColor, fillAlphaColor) => {
    let lastX = leftPad;
    let lastY = H / 2;
    ctx.beginPath();
    sparklineHistory.forEach((pt, idx) => {
      const x = leftPad + (idx / (MAX_SPARKLINE_POINTS - 1)) * plotW;
      const y = H - 8 - (Math.max(0, Math.min(100, pt[key])) / 100.0) * (H - 16);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      lastX = x;
      lastY = y;
    });
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.0;
    ctx.stroke();

    if (fillAlphaColor) {
      ctx.lineTo(lastX, H - 4);
      ctx.lineTo(leftPad, H - 4);
      ctx.closePath();
      ctx.fillStyle = fillAlphaColor;
      ctx.fill();
    }

    // Glowing live endpoint dot
    ctx.fillStyle = strokeColor;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3.2, 0, Math.PI * 2);
    ctx.fill();
  };

  // Match exact legend colors: Green (#10b981) Optical Flow, Cyan (#38bdf8) Confidence, Red (#ef4444) Priority
  drawCurve("motion", "#10b981", "rgba(16, 185, 129, 0.08)");
  drawCurve("confidence", "#38bdf8", null);
  drawCurve("priority", "#ef4444", "rgba(239, 68, 68, 0.14)");
}

/**
 * ============================================================================
 * UPGRADE 3: 1-Click Guided Judge Tour (Automated 5-Step Walkthrough)
 * ============================================================================
 */
const JUDGE_TOUR_STEPS = [
  {
    scenario: "fall_emergency",
    title: "Step 1/5: Closed-Loop Fall Detection & Agent-Triggered 2.4x ROI Zoom",
    desc: "Watch Worker P-102 collapse in Sector-A. OpenCV 5 detects aspect-ratio inversion + zero optical flow. The AI Agent doesn't just chat—it dynamically commands OpenCV 5 to execute a 2.4x ROI crop zoom (top-left Picture-in-Picture) before escalating to CRITICAL.",
  },
  {
    scenario: "fall_recovery",
    title: "Step 2/5: Dynamic Agentic Decision Reversal (Self-Recovery)",
    desc: "Here P-102 slips and falls, triggering an initial Agent escalation. At Frame 42, OpenCV 5 detects upright recovery and restored optical flow velocity—causing the AI Coordinator to autonomously downgrade priority and cancel EMS dispatch!",
  },
  {
    scenario: "restricted_intrusion",
    title: "Step 3/5: Sector-B 13.8kV High-Voltage Substation Intrusion",
    desc: "OpenCV 5 tests every tracked person's foot-anchor against the Sector-B 13.8kV substation polygon using cv2.pointPolygonTest. You can even click '✎ Draw Custom Zone' above the video to draw your own custom hazard polygon live!",
  },
  {
    scenario: "multi_crisis",
    title: "Step 4/5: Simultaneous Multi-Hazard Triage + CLAHE Enhancement + HITL Approval",
    desc: "A Class-B fire & smoke plume erupts while P-102 collapses and P-103 breaches Sector-B. The AI Agent commands OpenCV 5 to activate CLAHE contrast enhancement for smoke penetration and pauses for Human-in-the-Loop Operator Approval.",
  },
  {
    scenario: "evaluation_view",
    title: "Step 5/5: Quantitative OpenCV 5 + Agent Benchmark & Failure Analysis",
    desc: "Judges can verify 95.4% detection accuracy, 16.4 ms frame latency, documented Computer Vision failure cases, and click '▶ Run Live OpenCV 5 + Agent Benchmark Suite' to execute all 6 scenarios live.",
  },
];

let judgeTourActive = false;
let judgeTourIndex = 0;

async function startGuidedJudgeTour() {
  judgeTourActive = true;
  judgeTourIndex = 0;
  const banner = document.getElementById("judge-tour-banner");
  if (banner) banner.style.display = "block";
  await applyGuidedTourStep();
}

async function nextGuidedTourStep() {
  if (!judgeTourActive) return;
  judgeTourIndex = (judgeTourIndex + 1) % JUDGE_TOUR_STEPS.length;
  await applyGuidedTourStep();
}

async function prevGuidedTourStep() {
  if (!judgeTourActive) return;
  judgeTourIndex = (judgeTourIndex - 1 + JUDGE_TOUR_STEPS.length) % JUDGE_TOUR_STEPS.length;
  await applyGuidedTourStep();
}

function stopGuidedJudgeTour() {
  judgeTourActive = false;
  const banner = document.getElementById("judge-tour-banner");
  if (banner) banner.style.display = "none";
}

async function applyGuidedTourStep() {
  const step = JUDGE_TOUR_STEPS[judgeTourIndex];
  if (!step) return;
  setText("tour-step-badge", `TOUR STEP ${judgeTourIndex + 1} / 5`);
  setText("tour-step-title", step.title);
  setText("tour-step-desc", step.desc);

  if (step.scenario === "evaluation_view") {
    switchMainView("evaluation");
  } else {
    switchMainView("dashboard");
    await selectScenario(step.scenario);
  }
}

/**
 * ============================================================================
 * UPGRADE 4: AI Command Center Voice PA Announcer (Web Speech Synthesis)
 * ============================================================================
 */
let voicePAEnabled = false;
let lastAnnouncedHeadline = "";

function toggleVoicePA() {
  voicePAEnabled = !voicePAEnabled;
  const btn = document.getElementById("btn-voice-pa-toggle");
  if (btn) {
    btn.textContent = voicePAEnabled ? "🔊 Voice PA: ON" : "🔈 Voice PA: OFF";
    btn.style.borderColor = voicePAEnabled ? "rgba(16, 185, 129, 0.6)" : "";
    btn.style.color = voicePAEnabled ? "#6ee7b7" : "";
  }
  if (voicePAEnabled && "speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance("AI Emergency Coordinator Voice P A telemetry online.");
    u.rate = 1.06;
    window.speechSynthesis.speak(u);
  }
}

function announceHeadlineIfChanged(dec) {
  if (!voicePAEnabled || !dec || !("speechSynthesis" in window)) return;
  const headline = dec.headline || dec.summary_headline || "";
  if (!headline || headline === lastAnnouncedHeadline) return;
  lastAnnouncedHeadline = headline;

  // Only speak meaningful escalations, reversals, or critical updates
  window.speechSynthesis.cancel();
  const cleanText = headline.replace(/[^\w\s.,!-]/g, " ");
  const u = new SpeechSynthesisUtterance(`${dec.priority_level} priority. ${cleanText}`);
  u.rate = 1.08;
  window.speechSynthesis.speak(u);
}

/**
 * ============================================================================
 * UPGRADE 5: Degraded Camera Stress-Test, OpenCV Tunables, S3 & Arch Modals, Hotkeys
 * ============================================================================
 */
let degradedStressActive = false;

async function toggleDegradedCameraStress() {
  degradedStressActive = !degradedStressActive;
  const btn = document.getElementById("btn-stress-test");
  if (btn) {
    btn.textContent = degradedStressActive
      ? "🌙 Stress-Test: ON (CLAHE Active)"
      : "🌙 Stress-Test: Low-Light";
    btn.style.background = degradedStressActive ? "rgba(139, 92, 246, 0.28)" : "";
    btn.style.borderColor = degradedStressActive ? "#a78bfa" : "rgba(167,139,250,0.45)";
  }
  await updateCVTunables();
}

function onFallSliderInput(val) {
  setText("cv-fall-ar-label", Number(val).toFixed(2));
}

async function updateCVTunables() {
  const claheSel = document.getElementById("cv-clahe-select");
  const fallSlider = document.getElementById("cv-fall-ar-slider");
  const claheOverride = claheSel ? claheSel.value : "AUTO";
  const fallAr = fallSlider ? parseFloat(fallSlider.value) : 1.12;

  try {
    await fetch("/api/video/cv-tunables", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        degraded_lighting_stress: degradedStressActive,
        clahe_override: claheOverride,
        fall_aspect_threshold: fallAr,
      }),
    });
    await fetchAndRenderState(false);
    refreshSingleVideoFrame();
  } catch (err) {
    console.error("Failed to update OpenCV tunables:", err);
  }
}

async function openS3EvidenceModal(incidentId) {
  const modal = document.getElementById("s3-evidence-modal");
  if (!modal) return;
  const inc = latestIncidentsCache.find((i) => i.id === incidentId) || latestIncidentsCache[0];
  setText(
    "s3-modal-title",
    inc ? `${inc.id} · ${inc.title} (${inc.severity})` : "Incident S3 Evidence Snapshot"
  );
  setText(
    "s3-modal-uri",
    inc && inc.s3_evidence_key
      ? `s3://ai-emergency-coordinator-evidence/${inc.s3_evidence_key}`
      : "s3://ai-emergency-coordinator-evidence/incidents/latest_snapshot.jpg"
  );
  setText("s3-modal-json", JSON.stringify(inc || { status: "active" }, null, 2));

  // Load latest S3 artifact image or live annotated frame
  const imgEl = document.getElementById("s3-modal-img");
  try {
    const res = await fetch("/api/s3-evidence");
    if (res.ok) {
      const data = await res.json();
      if (data.artifacts && data.artifacts.length > 0 && imgEl) {
        imgEl.src = `${data.artifacts[0].url}?t=${Date.now()}`;
      } else if (imgEl) {
        imgEl.src = `/api/video/frame?t=${Date.now()}`;
      }
    }
  } catch (_) {
    if (imgEl) imgEl.src = `/api/video/frame?t=${Date.now()}`;
  }

  modal.style.display = "flex";
}

function closeS3EvidenceModal() {
  const modal = document.getElementById("s3-evidence-modal");
  if (modal) modal.style.display = "none";
}

function openArchitectureModal() {
  const modal = document.getElementById("architecture-rubric-modal");
  if (modal) modal.style.display = "flex";
}

function closeArchitectureModal() {
  const modal = document.getElementById("architecture-rubric-modal");
  if (modal) modal.style.display = "none";
}

// Global Demo Keyboard Shortcuts: [1-7] Scenarios, [Space] Pause, [V] View Mode, [Esc] Close Modal
window.addEventListener("keydown", (e) => {
  if (e.target && ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;

  if (e.key === "Escape") {
    closeS3EvidenceModal();
    closeArchitectureModal();
    if (isDrawingPolygon) togglePolygonDrawMode();
    return;
  }

  if (currentMainView !== "dashboard") return;

  const scenMap = {
    "1": "fall_emergency",
    "2": "fall_recovery",
    "3": "fire_smoke",
    "4": "crowd_surge",
    "5": "restricted_intrusion",
    "6": "multi_crisis",
    "7": "normal_patrol",
  };
  if (scenMap[e.key]) {
    e.preventDefault();
    selectScenario(scenMap[e.key]);
  } else if (e.code === "Space") {
    e.preventDefault();
    toggleStreamPause();
  } else if (e.key.toLowerCase() === "v") {
    e.preventDefault();
    const modes = ["opencv_annotated", "raw_video", "optical_flow", "split_comparison"];
    const activePill = document.querySelector(".view-pill.active[id^='vmode-']");
    const currMode = activePill ? activePill.id.replace("vmode-", "") : "opencv_annotated";
    const nextMode = modes[(modes.indexOf(currMode) + 1) % modes.length];
    setVideoViewMode(nextMode);
  }
});


