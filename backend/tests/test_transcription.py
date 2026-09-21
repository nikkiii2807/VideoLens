import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.audio_processor import audio_processor
from app.services.transcription import transcription_service
from app.models.database import init_db, get_db

def test_transcription():
    init_db()
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists()
    
    test_video_id = "test_transcribe_001"
    
    # 1. Extract audio
    wav_path = audio_processor.extract_audio(test_video_id, sample_path)
    print(f"Extracted wav: {wav_path}")
    assert wav_path is not None and wav_path.exists()
    
    # 2. Transcribe
    segments = transcription_service.transcribe(test_video_id, wav_path)
    print(f"Transcription segments: {segments}")
    assert len(segments) > 0, "Expected at least 1 speech segment"
    
    for seg in segments:
        assert "start_time" in seg
        assert "end_time" in seg
        assert len(seg["text"]) > 0
        print(f"[{seg['start_time']}s - {seg['end_time']}s]: {seg['text']}")

    # 3. Test database persistence
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transcripts WHERE video_id = ?", (test_video_id,))
        count = cursor.fetchone()[0]
        assert count == len(segments)

    # 4. Test silent / None audio
    empty_segs = transcription_service.transcribe("empty_vid", None)
    assert empty_segs == []
    
    print("ALL TEST_TRANSCRIPTION CHECKS PASSED!")

if __name__ == "__main__":
    test_transcription()
