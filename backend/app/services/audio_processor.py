import subprocess
from pathlib import Path
from typing import Optional
from app.config import FFMPEG_BIN, settings
from app.utils.logger import logger

class AudioProcessor:
    def extract_audio(self, video_id: str, video_path: Path) -> Optional[Path]:
        """
        Extracts 16kHz mono audio as WAV using FFmpeg.
        Returns path to extracted WAV file, or None if video has no audio stream.
        """
        audio_dir = settings.DATA_DIR / "videos" / video_id / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        wav_path = audio_dir / "audio.wav"

        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(wav_path)
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 100:
                logger.info(f"Video {video_id} has no valid audio stream or audio extraction was skipped.")
                if wav_path.exists():
                    wav_path.unlink()
                return None
            
            logger.info(f"Audio extracted for {video_id}: {wav_path} ({wav_path.stat().st_size / 1024:.1f} KB)")
            return wav_path
        except Exception as e:
            logger.warning(f"Failed to extract audio for {video_id}: {e}")
            return None

audio_processor = AudioProcessor()
