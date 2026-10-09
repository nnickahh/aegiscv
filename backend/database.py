"""SQLite database persistence for Incidents, Timeline Events, and System Audit Logs."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.config import DB_PATH
from backend.models import IncidentRecord, TimelineEvent


class DatabaseManager:
    """Thread-safe persistent SQLite storage for emergency incidents, timeline events, and AWS sync logs."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        except sqlite3.Error:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=MEMORY;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    priority_score INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    pipeline_stage TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    uncertainty REAL NOT NULL,
                    world_pos TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    recommended_action TEXT NOT NULL,
                    human_approval_status TEXT NOT NULL,
                    cv_evidence_summary TEXT NOT NULL,
                    s3_evidence_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    evolution_log TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    severity TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS aws_cloud_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    service TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def upsert_incident(self, incident: IncidentRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO incidents (
                    id, title, category, severity, priority_score, status,
                    pipeline_stage, confidence, uncertainty, world_pos, bbox,
                    recommended_action, human_approval_status, cv_evidence_summary,
                    s3_evidence_key, created_at, updated_at, evolution_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    category=excluded.category,
                    severity=excluded.severity,
                    priority_score=excluded.priority_score,
                    status=excluded.status,
                    pipeline_stage=excluded.pipeline_stage,
                    confidence=excluded.confidence,
                    uncertainty=excluded.uncertainty,
                    world_pos=excluded.world_pos,
                    bbox=excluded.bbox,
                    recommended_action=excluded.recommended_action,
                    human_approval_status=excluded.human_approval_status,
                    cv_evidence_summary=excluded.cv_evidence_summary,
                    s3_evidence_key=excluded.s3_evidence_key,
                    updated_at=excluded.updated_at,
                    evolution_log=excluded.evolution_log
                """,
                (
                    incident.id,
                    incident.title,
                    incident.category,
                    incident.severity,
                    incident.priority_score,
                    incident.status,
                    incident.pipeline_stage,
                    incident.confidence,
                    incident.uncertainty,
                    json.dumps(incident.world_pos),
                    json.dumps(incident.bbox),
                    incident.recommended_action,
                    incident.human_approval_status,
                    incident.cv_evidence_summary,
                    incident.s3_evidence_key,
                    incident.created_at,
                    incident.updated_at,
                    json.dumps(incident.evolution_log),
                ),
            )
            self._conn.commit()

    def get_incidents(self, limit: int = 25) -> List[IncidentRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM incidents ORDER BY priority_score DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            results: List[IncidentRecord] = []
            for r in rows:
                results.append(
                    IncidentRecord(
                        id=r["id"],
                        title=r["title"],
                        category=r["category"],
                        severity=r["severity"],
                        priority_score=r["priority_score"],
                        status=r["status"],
                        pipeline_stage=r["pipeline_stage"],
                        confidence=r["confidence"],
                        uncertainty=r["uncertainty"],
                        world_pos=json.loads(r["world_pos"]),
                        bbox=json.loads(r["bbox"]),
                        recommended_action=r["recommended_action"],
                        human_approval_status=r["human_approval_status"],
                        cv_evidence_summary=r["cv_evidence_summary"],
                        s3_evidence_key=r["s3_evidence_key"],
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                        evolution_log=json.loads(r["evolution_log"]),
                    )
                )
            return results

    def add_timeline_event(self, event: TimelineEvent) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO timeline_events
                (id, timestamp, stage, source, title, detail, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.timestamp,
                    event.stage,
                    event.source,
                    event.title,
                    event.detail,
                    event.severity,
                ),
            )
            self._conn.commit()

    def get_timeline_events(self, limit: int = 40) -> List[TimelineEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM timeline_events ORDER BY rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                TimelineEvent(
                    id=r["id"],
                    timestamp=r["timestamp"],
                    stage=r["stage"],
                    source=r["source"],
                    title=r["title"],
                    detail=r["detail"],
                    severity=r["severity"],
                )
                for r in rows
            ]

    def log_aws_operation(self, service: str, operation: str, status: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            self._conn.execute(
                """
                INSERT INTO aws_cloud_logs (timestamp, service, operation, status, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now_str, service, operation, status, json.dumps(payload)),
            )
            self._conn.commit()

    def get_aws_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM aws_cloud_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "service": r["service"],
                    "operation": r["operation"],
                    "status": r["status"],
                    "payload": json.loads(r["payload_json"]),
                }
                for r in rows
            ]

    def clear_session_data(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM incidents")
            self._conn.execute("DELETE FROM timeline_events")
            self._conn.commit()
