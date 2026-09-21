import gc
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings
from app.models.database import get_db
from app.utils.logger import logger

class TranscriptionService:
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading faster-whisper model '{self.model_size}' on CPU (int8)...")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            logger.info("faster-whisper model loaded successfully.")
        return self._model

    def transcribe(self, video_id: str, audio_path: Optional[Path]) -> List[Dict[str, Any]]:
        """
        Transcribes audio into timestamped segments and persists them in the database.
        Returns list of segments: [{'chunk_id', 'start_time', 'end_time', 'text'}]
        """
        if audio_path is None or not audio_path.exists():
            logger.info(f"No audio file provided for {video_id}. Returning empty transcript.")
            return []

        segments_list = []
        try:
            model = self._get_model()
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=1,
                word_timestamps=False,
                language="en"
            )

            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))

                idx = 0
                for seg in segments:
                    text = seg.text.strip()
                    if not text:
                        continue

                    chunk_id = f"{video_id}_t{idx:04d}"
                    start_time = round(seg.start, 2)
                    end_time = round(seg.end, 2)

                    cursor.execute("""
                        INSERT INTO transcripts (chunk_id, video_id, start_time, end_time, text)
                        VALUES (?, ?, ?, ?, ?)
                    """, (chunk_id, video_id, start_time, end_time, text))

                    segments_list.append({
                        "chunk_id": chunk_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "text": text
                    })
                    idx += 1

                conn.commit()

            logger.info(f"Transcription completed for {video_id}: {len(segments_list)} segments transcribed.")
        except Exception as e:
            logger.error(f"Error during transcription of {video_id}: {e}", exc_info=True)

        return segments_list

    def unload(self):
        """Release model from memory to free RAM for next pipeline stage."""
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("Whisper model unloaded from memory.")

transcription_service = TranscriptionService()
