import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.video_processor import video_processor
from app.config import settings
from app.models.database import init_db, get_db

def test_video_processing():
    init_db()
    
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists(), f"Sample video does not exist at {sample_path}"
    
    # 1. Test probe
    metadata = video_processor.probe_video(sample_path)
    print(f"Probe metadata: {metadata}")
    assert metadata["duration"] > 0, "Duration must be positive"
    assert metadata["width"] == 640, "Width should be 640"
    assert metadata["height"] == 360, "Height should be 360"
    
    # 2. Test upload validation
    with open(sample_path, "rb") as f:
        content = f.read()
    
    video_processor.validate_file("sample.mp4", len(content))
    
    # Test rejection of invalid extension
    try:
        video_processor.validate_file("sample.exe", 100)
        assert False, "Should reject invalid extension"
    except ValueError:
        pass

    # 3. Test save uploaded video
    vid_id, target_path, meta = video_processor.save_uploaded_video(content, "sample_lecture.mp4")
    assert target_path.exists()
    assert meta["duration"] > 0
    print(f"Uploaded successfully as {vid_id} at {target_path}")

    print("ALL TEST_UPLOAD CHECKS PASSED!")

if __name__ == "__main__":
    test_video_processing()
