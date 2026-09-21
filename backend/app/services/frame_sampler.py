import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from app.config import settings
from app.models.database import get_db
from app.utils.logger import logger

class FrameSampler:
    def __init__(self):
        self.sampling_fps = settings.SAMPLING_FPS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD # e.g. 0.92
        # A difference score >= (1.0 - threshold) is considered significant change
        self.min_diff_threshold = max(0.02, round(1.0 - self.similarity_threshold, 4))
        self.heartbeat_interval_sec = 2.0 # Force a keyframe at least every 2 seconds

    def _compute_frame_difference(self, prev_thumb: np.ndarray, curr_thumb: np.ndarray) -> float:
        """Computes normalized mean absolute difference between 64x64 grayscale thumbnails."""
        diff = np.mean(np.abs(prev_thumb.astype(np.float32) - curr_thumb.astype(np.float32))) / 255.0
        return float(diff)

    def extract_frames(self, video_id: str, video_path: Path) -> List[Dict[str, Any]]:
        """
        Samples video at configurable fps (e.g. 2.0 fps) and performs
        intelligent redundancy filtering while preserving scene transitions.
        """
        frames_dir = settings.DATA_DIR / "videos" / video_id / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video at {video_path}")

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_native_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # Frame step in native video units
        step = max(1, int(round(native_fps / self.sampling_fps)))
        
        sampled_frames = []
        prev_thumb = None
        last_saved_timestamp = -999.0
        
        frame_idx = 0
        sample_idx = 0

        logger.info(
            f"Sampling video {video_id}: native_fps={native_fps:.1f}, "
            f"target_fps={self.sampling_fps}, step={step} frames"
        )

        with get_db() as conn:
            cursor = conn.cursor()
            # Clear any existing frames for this video_id
            cursor.execute("DELETE FROM frames WHERE video_id = ?", (video_id,))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    timestamp = round(frame_idx / native_fps, 2)
                    
                    # Create normalized 64x64 grayscale thumbnail for fast diff comparison
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    curr_thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

                    is_keyframe = False
                    diff_score = 0.0

                    if prev_thumb is None:
                        # Always keep the first frame
                        is_keyframe = True
                        diff_score = 1.0
                    else:
                        diff_score = self._compute_frame_difference(prev_thumb, curr_thumb)
                        time_since_last = timestamp - last_saved_timestamp
                        
                        # Conditions to keep:
                        # 1. Visual change above threshold (motion, new object, scene cut)
                        # 2. Heartbeat interval exceeded (ensure temporal continuity)
                        if diff_score >= self.min_diff_threshold or time_since_last >= self.heartbeat_interval_sec:
                            is_keyframe = True

                    if is_keyframe:
                        frame_id = f"{video_id}_f{sample_idx:04d}"
                        frame_filename = f"frame_{sample_idx:04d}_{timestamp:.2f}s.jpg"
                        frame_save_path = frames_dir / frame_filename

                        # Save high-quality frame for display and embeddings
                        cv2.imwrite(str(frame_save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        # Record in database
                        cursor.execute("""
                            INSERT INTO frames (frame_id, video_id, frame_idx, timestamp, image_path, is_keyframe, difference_score)
                            VALUES (?, ?, ?, ?, ?, 1, ?)
                        """, (
                            frame_id,
                            video_id,
                            sample_idx,
                            timestamp,
                            str(frame_save_path),
                            round(diff_score, 4)
                        ))

                        sampled_frames.append({
                            "frame_id": frame_id,
                            "frame_idx": sample_idx,
                            "timestamp": timestamp,
                            "image_path": str(frame_save_path),
                            "difference_score": round(diff_score, 4),
                            "is_keyframe": True
                        })

                        last_saved_timestamp = timestamp
                        sample_idx += 1

                    prev_thumb = curr_thumb

                frame_idx += 1

            conn.commit()

        cap.release()
        logger.info(
            f"Frame extraction completed for {video_id}: extracted {len(sampled_frames)} keyframes "
            f"from {frame_idx} raw frames."
        )
        return sampled_frames

frame_sampler = FrameSampler()
