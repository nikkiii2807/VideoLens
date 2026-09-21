import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.config import settings
from app.models.database import get_db
from app.models.schemas import VideoUploadResponse, VideoStatusResponse, FrameInfo, TranscriptSegment
from app.services.video_processor import video_processor
from app.utils.logger import logger

router = APIRouter()

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_size = len(content)
        
        # Validate format and size
        video_processor.validate_file(file.filename, file_size)
        
        # Save and probe video
        video_id, filepath, meta = video_processor.save_uploaded_video(content, file.filename)
        
        # Relative URL for static serving
        ext = Path(file.filename).suffix.lower()
        video_url = f"/data/videos/{video_id}/video{ext}"
        
        # Insert into SQLite
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (video_id, filename, filepath, duration, fps, width, height, file_size_bytes, status, stage)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploaded', 'ready')
            """, (
                video_id,
                file.filename,
                str(filepath),
                meta["duration"],
                meta["fps"],
                meta["width"],
                meta["height"],
                file_size
            ))
            conn.commit()
            
        logger.info(f"Video uploaded: {video_id} ({file.filename}, {meta['duration']}s, {meta['width']}x{meta['height']})")
        
        return VideoUploadResponse(
            video_id=video_id,
            filename=file.filename,
            duration=meta["duration"],
            fps=meta["fps"],
            width=meta["width"],
            height=meta["height"],
            file_size_bytes=file_size,
            status="uploaded",
            video_url=video_url
        )
    except ValueError as e:
        logger.warning(f"Validation error for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error uploading video {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

@router.post("/load-demo", response_model=VideoUploadResponse)
def load_demo_video(demo_name: str = "lecture"):
    demo_files = {
        "cooking": "sample_cooking.mp4",
        "robotics": "sample_robotics.mp4",
        "lecture": "sample_lecture.mp4",
        "60s": "sample_60s.mp4",
        "full": "sample_60s.mp4"
    }
    filename = demo_files.get(demo_name.lower(), "sample_60s.mp4")
    demo_dir = Path(__file__).resolve().parent.parent.parent.parent / "demo_data"
    demo_path = demo_dir / filename

    if not demo_path.exists():
        if filename == "sample_60s.mp4":
            from demo_data.generate_60s_demo import generate_60s_video
            generate_60s_video(demo_path)
        elif filename == "sample_lecture.mp4":
            from demo_data.sample_generator import create_sample_video
            create_sample_video(demo_path)
        else:
            from demo_data.generate_additional_demos import generate_cooking_demo, generate_robotics_demo
            if filename == "sample_cooking.mp4":
                generate_cooking_demo()
            else:
                generate_robotics_demo()

    with open(demo_path, "rb") as f:
        content = f.read()

    video_id, filepath, meta = video_processor.save_uploaded_video(content, filename)
    video_url = f"/data/videos/{video_id}/video.mp4"

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (video_id, filename, filepath, duration, fps, width, height, file_size_bytes, status, stage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'uploaded', 'ready')
        """, (
            video_id, filename, str(filepath),
            meta["duration"], meta["fps"], meta["width"], meta["height"], len(content)
        ))
        conn.commit()

    logger.info(f"Loaded demo video '{demo_name}' ({filename}) as {video_id}")
    return VideoUploadResponse(
        video_id=video_id,
        filename=filename,
        duration=meta["duration"],
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        file_size_bytes=len(content),
        status="uploaded",
        video_url=video_url
    )

@router.post("/{video_id}/process")
def process_video(video_id: str, background_tasks: BackgroundTasks):
    from app.services.pipeline import run_video_pipeline
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM videos WHERE video_id = ?", (video_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Video not found")
        if row["status"] == "processing":
            return {"message": "Video is already being processed", "video_id": video_id}

    background_tasks.add_task(run_video_pipeline, video_id)
    return {"message": "Video processing started", "video_id": video_id, "status": "processing"}

@router.get("/{video_id}/status", response_model=VideoStatusResponse)
def get_video_status(video_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Video not found")
            
        cursor.execute("SELECT COUNT(*) FROM frames WHERE video_id = ?", (video_id,))
        frames_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM transcripts WHERE video_id = ?", (video_id,))
        transcript_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ocr_chunks WHERE video_id = ?", (video_id,))
        ocr_count = cursor.fetchone()[0]
        
        # Compute progress estimation
        progress = 0
        if row["status"] == "uploaded":
            progress = 10
        elif row["status"] == "processing":
            stage_weights = {
                "sampling": 25,
                "transcribing": 45,
                "ocr": 65,
                "embedding": 85,
                "indexing": 95,
            }
            progress = stage_weights.get(row["stage"], 30)
        elif row["status"] == "completed":
            progress = 100
        elif row["status"] == "failed":
            progress = 0
            
        return VideoStatusResponse(
            video_id=video_id,
            status=row["status"],
            stage=row["stage"],
            progress=progress,
            error_message=row["error_message"],
            frames_count=frames_count,
            transcript_segments_count=transcript_count,
            ocr_detections_count=ocr_count
        )

@router.get("/{video_id}/frames", response_model=list[FrameInfo])
def get_video_frames(video_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT frame_id, frame_idx, timestamp, image_path, is_keyframe, difference_score
            FROM frames WHERE video_id = ? ORDER BY timestamp ASC
        """, (video_id,))
        rows = cursor.fetchall()
        
        # Check OCR existence
        cursor.execute("SELECT DISTINCT frame_id FROM ocr_chunks WHERE video_id = ?", (video_id,))
        ocr_frame_ids = {r[0] for r in cursor.fetchall()}
        
        frames = []
        for r in rows:
            # Build relative image url
            rel_path = Path(r["image_path"]).as_posix()
            data_index = rel_path.find("/data/")
            image_url = rel_path[data_index:] if data_index != -1 else f"/data/videos/{video_id}/frames/{Path(r['image_path']).name}"
            
            frames.append(FrameInfo(
                frame_id=r["frame_id"],
                frame_idx=r["frame_idx"],
                timestamp=r["timestamp"],
                image_url=image_url,
                is_keyframe=bool(r["is_keyframe"]),
                has_ocr=r["frame_id"] in ocr_frame_ids,
                difference_score=r["difference_score"]
            ))
        return frames

@router.get("/{video_id}/transcript", response_model=list[TranscriptSegment])
def get_video_transcript(video_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chunk_id, start_time, end_time, text
            FROM transcripts WHERE video_id = ? ORDER BY start_time ASC
        """, (video_id,))
        rows = cursor.fetchall()
        return [
            TranscriptSegment(
                chunk_id=r["chunk_id"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                text=r["text"]
            )
            for r in rows
        ]

@router.get("/list")
def list_videos():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT video_id, filename, duration, fps, width, height, status, created_at
            FROM videos ORDER BY created_at DESC LIMIT 20
        """)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
