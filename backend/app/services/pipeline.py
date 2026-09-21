import gc
import traceback
from pathlib import Path
from app.config import settings
from app.models.database import get_db
from app.services.frame_sampler import frame_sampler
from app.services.audio_processor import audio_processor
from app.services.transcription import transcription_service
from app.services.ocr_service import ocr_service
from app.services.embedding_service import embedding_service
from app.services.vector_store import VectorStore
from app.utils.logger import logger

def update_video_progress(video_id: str, status: str, stage: str, error_message: str = None):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE videos SET status = ?, stage = ?, error_message = ?
            WHERE video_id = ?
        """, (status, stage, error_message, video_id))
        conn.commit()

def run_video_pipeline(video_id: str):
    """
    Executes the complete VideoLens multimodal pipeline with memory-efficient
    model loading: each heavy model is unloaded after its stage to stay within
    free-tier RAM limits (512MB).

    Stages: Sampling -> Transcription -> OCR -> Embeddings -> FAISS Indexing
    """
    logger.info(f"Starting VideoLens pipeline for {video_id}...")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM videos WHERE video_id = ?", (video_id,))
        row = cursor.fetchone()
        if not row:
            logger.error(f"Video {video_id} not found in database.")
            return
        video_path = Path(row[0])

    try:
        # Stage 1: Frame Sampling & Redundancy Filtering (no heavy ML, uses OpenCV)
        update_video_progress(video_id, status="processing", stage="sampling")
        frames = frame_sampler.extract_frames(video_id, video_path)
        logger.info(f"Pipeline: {len(frames)} frames extracted for {video_id}.")
        gc.collect()

        # Stage 2: Audio Extraction & Transcription (Whisper tiny ~70MB)
        update_video_progress(video_id, status="processing", stage="transcribing")
        wav_path = audio_processor.extract_audio(video_id, video_path)
        transcripts = transcription_service.transcribe(video_id, wav_path)
        logger.info(f"Pipeline: {len(transcripts)} speech segments transcribed for {video_id}.")
        # ✅ Unload Whisper immediately to free ~70MB before loading EasyOCR
        transcription_service.unload()

        # Stage 3: Selective OCR (EasyOCR ~300-400MB)
        update_video_progress(video_id, status="processing", stage="ocr")
        ocr_chunks = ocr_service.extract_text_from_frames(video_id, frames)
        logger.info(f"Pipeline: {len(ocr_chunks)} OCR detections for {video_id}.")
        # ✅ Unload EasyOCR immediately to free ~300-400MB before loading CLIP
        ocr_service.unload()

        # Stage 4: Multimodal Embeddings (CLIP ~350MB + SentenceTransformers ~90MB)
        update_video_progress(video_id, status="processing", stage="embedding")
        frame_paths = [Path(f["image_path"]) for f in frames]
        visual_embeddings = embedding_service.embed_images_batch(frame_paths)

        # Build text chunks for semantic embedding
        text_chunks_meta = []
        text_strings = []

        for t in transcripts:
            text_chunks_meta.append({
                "type": "transcript",
                "id": t["chunk_id"],
                "timestamp": t["start_time"],
                "end_time": t["end_time"],
                "text": t["text"]
            })
            text_strings.append(t["text"])

        for o in ocr_chunks:
            text_chunks_meta.append({
                "type": "ocr",
                "id": o["ocr_id"],
                "frame_id": o["frame_id"],
                "timestamp": o["timestamp"],
                "text": o["text"],
                "confidence": o["confidence"]
            })
            text_strings.append(o["text"])

        text_embeddings = embedding_service.embed_texts_semantic_batch(text_strings)
        # ✅ Unload CLIP + SentenceTransformers to free ~440MB before FAISS indexing
        embedding_service.unload()

        # Stage 5: FAISS Vector Store Indexing (lightweight, numpy only)
        update_video_progress(video_id, status="processing", stage="indexing")
        store = VectorStore(video_id)
        store.build_indices(
            visual_embeddings=visual_embeddings,
            visual_meta=frames,
            text_embeddings=text_embeddings,
            text_meta=text_chunks_meta
        )

        # Complete
        update_video_progress(video_id, status="completed", stage="ready")
        logger.info(f"Pipeline successfully completed for {video_id}!")

    except Exception as e:
        err_str = str(e)
        tb = traceback.format_exc()
        logger.error(f"Pipeline failed for {video_id}: {err_str}\n{tb}")
        update_video_progress(video_id, status="failed", stage="error", error_message=err_str)
        # ✅ Always clean up on failure too
        try:
            transcription_service.unload()
            ocr_service.unload()
            embedding_service.unload()
        except Exception:
            pass
        gc.collect()
