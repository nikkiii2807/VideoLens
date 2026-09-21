import math
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.models.database import get_db
from app.utils.logger import logger

class TemporalReasoningService:
    def __init__(self, default_window_sec: float = 4.0):
        self.default_window_sec = default_window_sec

    def merge_temporal_windows(
        self,
        seed_timestamps: List[float],
        window_sec: float,
        max_duration: float = 60.0
    ) -> List[Tuple[float, float]]:
        """
        Expands each seed timestamp into [t - window_sec, t + window_sec]
        and merges overlapping or adjacent intervals into coherent episodes.
        """
        if not seed_timestamps:
            return []

        raw_windows = []
        for t in seed_timestamps:
            start = max(0.0, round(t - window_sec, 2))
            end = min(max_duration, round(t + window_sec, 2))
            raw_windows.append((start, end))

        # Sort by start time
        raw_windows.sort(key=lambda w: w[0])

        merged = [raw_windows[0]]
        for curr_start, curr_end in raw_windows[1:]:
            last_start, last_end = merged[-1]
            # Merge if overlapping or within 1.0s gap
            if curr_start <= last_end + 1.0:
                merged[-1] = (last_start, max(last_end, curr_end))
            else:
                merged.append((curr_start, curr_end))

        return merged

    def compute_temporal_proximity(self, t: float, ref_t: float, window_sec: float) -> float:
        """
        Gaussian proximity score centered at seed peak ref_t.
        Returns value in (0.0, 1.0].
        """
        sigma = max(1.0, window_sec / 2.0)
        dist_sq = (t - ref_t) ** 2
        return math.exp(-dist_sq / (2 * (sigma ** 2)))

    def gather_temporal_context(
        self,
        video_id: str,
        episodes: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Retrieves all contextual frames, transcript segments, and OCR chunks
        occurring within the merged temporal episodes.
        """
        all_frames = []
        all_transcripts = []
        all_ocr = []

        with get_db() as conn:
            cursor = conn.cursor()

            for start_t, end_t in episodes:
                # Frames in window
                cursor.execute("""
                    SELECT frame_id, frame_idx, timestamp, image_path, is_keyframe, difference_score
                    FROM frames
                    WHERE video_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, (video_id, start_t, end_t))
                for r in cursor.fetchall():
                    all_frames.append(dict(r))

                # Transcripts in window
                cursor.execute("""
                    SELECT chunk_id, start_time, end_time, text
                    FROM transcripts
                    WHERE video_id = ? AND (
                        (start_time <= ? AND end_time >= ?) OR
                        (start_time >= ? AND start_time <= ?)
                    )
                    ORDER BY start_time ASC
                """, (video_id, end_t, start_t, start_t, end_t))
                for r in cursor.fetchall():
                    all_transcripts.append(dict(r))

                # OCR in window
                cursor.execute("""
                    SELECT ocr_id, frame_id, timestamp, text, confidence
                    FROM ocr_chunks
                    WHERE video_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                """, (video_id, start_t, end_t))
                for r in cursor.fetchall():
                    all_ocr.append(dict(r))

        # Deduplicate while preserving order
        unique_frames = {f["frame_id"]: f for f in all_frames}
        unique_transcripts = {t["chunk_id"]: t for t in all_transcripts}
        unique_ocr = {o["ocr_id"]: o for o in all_ocr}

        return {
            "episodes": [{"start": round(s, 2), "end": round(e, 2)} for s, e in episodes],
            "context_frames": list(unique_frames.values()),
            "context_transcripts": list(unique_transcripts.values()),
            "context_ocr": list(unique_ocr.values())
        }

    def get_temporal_event_sequence(
        self,
        video_id: str,
        center_timestamp: float,
        window_sec: float = 3.5,
        max_frames: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieves a coherent chronological sequence of consecutive frames and
        corresponding OCR/transcripts around a key timestamp to support
        temporal and action reasoning (e.g. before -> event -> after).
        """
        start_t = max(0.0, round(center_timestamp - window_sec, 2))
        end_t = round(center_timestamp + window_sec, 2)

        with get_db() as conn:
            cursor = conn.cursor()
            # Consecutive frames in window
            cursor.execute("""
                SELECT frame_id, frame_idx, timestamp, image_path, is_keyframe, difference_score
                FROM frames
                WHERE video_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (video_id, start_t, end_t))
            frames = [dict(r) for r in cursor.fetchall()]

            # OCR in window
            cursor.execute("""
                SELECT frame_id, timestamp, text, confidence
                FROM ocr_chunks
                WHERE video_id = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """, (video_id, start_t, end_t))
            ocr_items = [dict(r) for r in cursor.fetchall()]

            # Transcripts in window
            cursor.execute("""
                SELECT start_time, end_time, text
                FROM transcripts
                WHERE video_id = ? AND (
                    (start_time <= ? AND end_time >= ?) OR
                    (start_time >= ? AND start_time <= ?)
                )
                ORDER BY start_time ASC
            """, (video_id, end_t, start_t, start_t, end_t))
            transcripts = [dict(r) for r in cursor.fetchall()]

        # Subsample frames if too many to keep chronological event progression
        if len(frames) > max_frames:
            step = len(frames) / max_frames
            sampled_indices = [int(i * step) for i in range(max_frames)]
            frames = [frames[i] for i in sampled_indices]

        # Build chronological event items
        sequence_items = []
        text_lines = []
        for f in frames:
            t = f["timestamp"]
            # Find relevant OCR for this frame or within 0.5s
            f_ocr = [o["text"] for o in ocr_items if o["frame_id"] == f["frame_id"] or abs(o["timestamp"] - t) <= 0.6]
            # Find relevant transcripts active at this timestamp
            f_tx = [tx["text"] for tx in transcripts if tx["start_time"] <= t + 0.5 and tx["end_time"] >= t - 0.5]

            ocr_str = f" [OCR: '{', '.join(f_ocr)}']" if f_ocr else ""
            tx_str = f" [Speech: '{' '.join(f_tx)}']" if f_tx else ""

            mins = int(t // 60)
            secs = t % 60
            ts_label = f"{mins:02d}:{secs:04.1f}"
            line = f"{ts_label} (t={t:.1f}s) - Frame {f['frame_id']}{ocr_str}{tx_str}"
            text_lines.append(line)

            sequence_items.append({
                "timestamp": t,
                "frame_id": f["frame_id"],
                "image_path": f["image_path"],
                "difference_score": f.get("difference_score", 0.0),
                "ocr_texts": f_ocr,
                "transcript_texts": f_tx,
                "label": line
            })

        return {
            "window": (start_t, end_t),
            "frames": sequence_items,
            "text_progression": "\n".join(text_lines)
        }

temporal_reasoning_service = TemporalReasoningService(default_window_sec=settings.TEMPORAL_WINDOW_SECONDS)
