import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.frame_sampler import frame_sampler
from app.services.ocr_service import ocr_service
from app.models.database import init_db, get_db

def test_ocr_pipeline():
    init_db()
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists()
    
    test_video_id = "test_ocr_vid_001"
    
    # 1. Sample frames
    frames = frame_sampler.extract_frames(test_video_id, sample_path)
    assert len(frames) > 0
    
    # 2. Run selective OCR
    ocr_chunks = ocr_service.extract_text_from_frames(test_video_id, frames)
    print(f"Extracted {len(ocr_chunks)} OCR chunks")
    assert len(ocr_chunks) > 0, "Expected OCR detections on the sample lecture slides"
    
    found_texts = [c["text"].lower() for c in ocr_chunks]
    print(f"Sample detected texts: {[c['text'] for c in ocr_chunks[:8]]}")
    
    # Check that key lecture phrases are found
    has_videolens = any("videolens" in t or "lecture" in t for t in found_texts)
    has_gaussian = any("gaussian" in t or "kernel" in t for t in found_texts)
    print(f"Detected 'videolens/lecture': {has_videolens}, Detected 'gaussian/kernel': {has_gaussian}")
    assert has_videolens or has_gaussian, "Expected lecture text to be recognized"

    # 3. Test database persistence
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ocr_chunks WHERE video_id = ?", (test_video_id,))
        count = cursor.fetchone()[0]
        assert count == len(ocr_chunks)

    print("ALL TEST_OCR CHECKS PASSED!")

if __name__ == "__main__":
    test_ocr_pipeline()
