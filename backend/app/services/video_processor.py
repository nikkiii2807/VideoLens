import os
import shutil
import uuid
import subprocess
import json
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple
from app.config import settings
from app.utils.logger import logger

class VideoProcessor:
    def __init__(self):
        self.max_duration = settings.MAX_DURATION_SECONDS
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        self.allowed_extensions = set(settings.ALLOWED_EXTENSIONS)

    def validate_file(self, filename: str, file_size: int) -> str:
        ext = Path(filename).suffix.lower()
        if ext not in self.allowed_extensions:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed: {', '.join(self.allowed_extensions)}")
        if file_size > self.max_file_size:
            raise ValueError(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds limit of {settings.MAX_FILE_SIZE_MB}MB")
        return ext

    def probe_video(self, video_path: Path) -> Dict[str, Any]:
        """Probes video metadata using OpenCV with FFprobe fallback."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            # Fallback to ffprobe
            return self._probe_with_ffprobe(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        cap.release()

        # If OpenCV returned 0 duration, try ffprobe
        if duration <= 0.0 or width == 0:
            try:
                return self._probe_with_ffprobe(video_path)
            except Exception as e:
                logger.warning(f"FFprobe fallback failed: {e}")

        return {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "frame_count": int(frame_count)
        }

    def _probe_with_ffprobe(self, video_path: Path) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        stream = data.get("streams", [{}])[0] if data.get("streams") else {}
        format_info = data.get("format", {})
        
        duration = float(stream.get("duration") or format_info.get("duration") or 0.0)
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        
        r_fps = stream.get("r_frame_rate", "25/1")
        if "/" in r_fps:
            num, den = r_fps.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 25.0
        else:
            fps = float(r_fps)

        frame_count = int(stream.get("nb_frames") or (duration * fps))
        return {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "frame_count": frame_count
        }

    def save_uploaded_video(self, file_bytes: bytes, filename: str) -> Tuple[str, Path, Dict[str, Any]]:
        video_id = str(uuid.uuid4())[:12]
        video_dir = settings.DATA_DIR / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        
        ext = Path(filename).suffix.lower()
        target_path = video_dir / f"video{ext}"
        
        with open(target_path, "wb") as f:
            f.write(file_bytes)
            
        metadata = self.probe_video(target_path)
        
        if metadata["duration"] > self.max_duration:
            # Clean up and reject
            shutil.rmtree(video_dir, ignore_errors=True)
            raise ValueError(
                f"Video duration ({metadata['duration']}s) exceeds maximum allowed {self.max_duration}s"
            )
            
        return video_id, target_path, metadata

video_processor = VideoProcessor()
