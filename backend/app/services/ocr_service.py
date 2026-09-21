import gc
import cv2
import uuid
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings
from app.models.database import get_db
from app.utils.logger import logger

class OCRService:
    def __init__(self, languages: List[str] = ["en"]):
        self.languages = languages
        self._reader = None
        self.min_confidence = 0.35
        self.min_text_length = 3

    def _get_reader(self):
        if self._reader is None:
            logger.info("Initializing EasyOCR reader on CPU...")
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized.")
        return self._reader

    def has_text_regions(self, image_path: Path) -> bool:
        """
        Heuristic pre-filter to detect high gradient density indicative of text
        before invoking heavy deep learning OCR.
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
            
        # Sobel gradient in horizontal direction (text has frequent vertical stroke edges)
        grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        grad_mag = np.abs(grad_x)
        high_grad_ratio = np.mean(grad_mag > 35)
        
        # If image has sufficient edge energy, candidate for OCR
        return bool(high_grad_ratio > 0.04)

    def extract_text_from_frames(self, video_id: str, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs selective OCR on candidate keyframes and stores detected text in database.
        Returns list of structured OCR chunks.
        """
        ocr_results = []
        if not frames:
            return ocr_results

        reader = self._get_reader()

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM ocr_chunks WHERE video_id = ?", (video_id,))

            detection_idx = 0
            for f in frames:
                frame_path = Path(f["image_path"])
                if not frame_path.exists():
                    continue

                # Selective check: only scan keyframes that show potential text contours
                if not self.has_text_regions(frame_path):
                    continue

                try:
                    # Run EasyOCR
                    detections = reader.readtext(str(frame_path))
                    
                    frame_texts = []
                    for bbox, text, conf in detections:
                        cleaned = text.strip()
                        if conf < self.min_confidence or len(cleaned) < self.min_text_length:
                            continue
                        
                        # Discard pure symbol noise
                        if not any(c.isalnum() for c in cleaned):
                            continue

                        ocr_id = f"{video_id}_ocr_{detection_idx:04d}"
                        timestamp = f["timestamp"]

                        cursor.execute("""
                            INSERT INTO ocr_chunks (ocr_id, video_id, frame_id, timestamp, text, confidence)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (ocr_id, video_id, f["frame_id"], timestamp, cleaned, round(conf, 4)))

                        ocr_results.append({
                            "ocr_id": ocr_id,
                            "video_id": video_id,
                            "frame_id": f["frame_id"],
                            "timestamp": timestamp,
                            "text": cleaned,
                            "confidence": round(conf, 4)
                        })
                        detection_idx += 1
                        frame_texts.append(cleaned)

                    if frame_texts:
                        logger.info(f"Frame {f['frame_id']} @ {f['timestamp']}s OCR found: {frame_texts}")

                except Exception as e:
                    logger.warning(f"OCR failed for frame {f['frame_id']}: {e}")

            conn.commit()

        logger.info(f"Selective OCR completed for {video_id}: {len(ocr_results)} text chunks detected.")
        return ocr_results

    def unload(self):
        """Release EasyOCR reader from memory to free RAM for next pipeline stage."""
        if self._reader is not None:
            del self._reader
            self._reader = None
            gc.collect()
            logger.info("EasyOCR reader unloaded from memory.")

ocr_service = OCRService()
