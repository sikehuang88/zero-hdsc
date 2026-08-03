"""SQLite persistence for layered emotion and vocal-performance plans."""

from __future__ import annotations

import json
import sqlite3

from ssa.domain.emotion_frame import EmotionFrame


class EmotionFrameRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def insert(self, frame: EmotionFrame) -> EmotionFrame:
        self._connection.execute("SAVEPOINT insert_emotion_frame")
        try:
            self._connection.execute(
                """
                INSERT INTO emotion_frames (
                    id, conversation_id, correlation_id, query_event_id,
                    primary_emotion, primary_intensity, valence, arousal,
                    inhibition, certainty, action_tendency, frame_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    frame.id,
                    frame.conversation_id,
                    frame.correlation_id,
                    frame.query_event_id,
                    frame.primary.name,
                    frame.primary.intensity,
                    frame.valence,
                    frame.arousal,
                    frame.inhibition,
                    frame.certainty,
                    frame.action_tendency,
                    frame.model_dump_json(),
                    frame.created_at_ms,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO voice_performance_segments (
                    id, emotion_frame_id, conversation_id, correlation_id,
                    sequence, recipe, intensity, rate, performance_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        segment.id,
                        frame.id,
                        frame.conversation_id,
                        frame.correlation_id,
                        segment.sequence,
                        segment.recipe,
                        segment.intensity,
                        segment.rate,
                        segment.model_dump_json(),
                        frame.created_at_ms,
                    )
                    for segment in frame.voice_segments
                ],
            )
            self._connection.execute("RELEASE insert_emotion_frame")
        except Exception:
            self._connection.execute("ROLLBACK TO insert_emotion_frame")
            self._connection.execute("RELEASE insert_emotion_frame")
            raise
        return frame

    def find_by_correlation(self, correlation_id: str) -> EmotionFrame | None:
        row = self._connection.execute(
            "SELECT frame_json FROM emotion_frames WHERE correlation_id = ?",
            (correlation_id,),
        ).fetchone()
        if row is None:
            return None
        return EmotionFrame.model_validate(json.loads(str(row["frame_json"])))

    def recent(self, conversation_id: str, *, limit: int = 20) -> list[EmotionFrame]:
        rows = self._connection.execute(
            """
            SELECT frame_json FROM emotion_frames
            WHERE conversation_id = ?
            ORDER BY created_at_ms DESC, rowid DESC
            LIMIT ?
            """,
            (conversation_id, max(1, min(limit, 100))),
        ).fetchall()
        return [
            EmotionFrame.model_validate(json.loads(str(row["frame_json"])))
            for row in rows
        ]


__all__ = ["EmotionFrameRepository"]
