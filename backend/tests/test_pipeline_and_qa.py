import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.database import init_db, get_db
from app.services.video_processor import video_processor
from app.services.pipeline import run_video_pipeline
from app.models.schemas import QuestionRequest
from app.api.query import ask_question

def test_full_pipeline_and_qa():
    init_db()
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists()

    # 1. Save video
    with open(sample_path, "rb") as f:
        content = f.read()

    video_id, target_path, meta = video_processor.save_uploaded_video(content, "sample_lecture.mp4")
    print(f"Video uploaded: {video_id}")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (video_id, filename, filepath, duration, fps, width, height, file_size_bytes, status, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploaded', 'ready')
        """, (
            video_id, "sample_lecture.mp4", str(target_path),
            meta["duration"], meta["fps"], meta["width"], meta["height"], len(content)
        ))
        conn.commit()

    # 2. Run Video Understanding Pipeline
    print(f"Executing pipeline for {video_id}...")
    run_video_pipeline(video_id)

    # 3. Verify status in database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, stage FROM videos WHERE video_id = ?", (video_id,))
        row = cursor.fetchone()
        assert row["status"] == "completed", f"Status should be completed, got {row['status']}"
        assert row["stage"] == "ready"

        cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = ?", (video_id,))
        n_frames = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM transcripts WHERE video_id = ?", (video_id,))
        n_transcripts = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ocr_chunks WHERE video_id = ?", (video_id,))
        n_ocr = cursor.fetchone()[0]

    print(f"Pipeline Stats: {n_frames} frames, {n_transcripts} transcript chunks, {n_ocr} OCR detections.")
    assert n_frames > 0
    assert n_transcripts > 0
    assert n_ocr > 0

    # 4. Ask Question
    q = "What happens after the Gaussian filtering is introduced?"
    req = QuestionRequest(question=q, temporal_window=4.0, top_k=5)
    
    # Run async function
    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(ask_question(video_id, req))

    print("\n--- GROUNDED QA RESPONSE ---")
    print(f"Question: {resp.question}")
    print(f"Answer: {resp.answer}")
    print(f"Confidence: {resp.confidence}")
    print(f"Timestamps: {[(ts.start, ts.end, ts.description) for ts in resp.timestamps]}")
    print(f"Supporting Frames: {len(resp.supporting_frames)}")
    print(f"Evidence Used: {len(resp.evidence_used)}")
    print(f"Latency: {resp.latency_seconds}s")

    assert len(resp.answer) > 10
    assert resp.confidence > 0.0
    assert len(resp.timestamps) > 0
    assert len(resp.supporting_frames) > 0
    assert len(resp.evidence_used) > 0

    print("\nALL PIPELINE AND QA CHECKS PASSED!")

if __name__ == "__main__":
    test_full_pipeline_and_qa()
