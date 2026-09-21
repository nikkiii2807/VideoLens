import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.frame_sampler import frame_sampler
from app.models.database import init_db, get_db

def test_frame_sampler():
    init_db()
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists()
    
    test_video_id = "test_vid_001"
    frames = frame_sampler.extract_frames(test_video_id, sample_path)
    
    print(f"Extracted {len(frames)} frames for {test_video_id}")
    assert len(frames) > 5, "Should have extracted multiple keyframes"
    
    # Check first frame
    assert frames[0]["timestamp"] == 0.0
    assert Path(frames[0]["image_path"]).exists()
    
    # Check database persistence
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = ?", (test_video_id,))
        count = cursor.fetchone()[0]
        assert count == len(frames)
        
    print(f"Sample frames extracted: {[f['timestamp'] for f in frames[:6]]}")
    print("ALL TEST_FRAMES CHECKS PASSED!")

if __name__ == "__main__":
    test_frame_sampler()
