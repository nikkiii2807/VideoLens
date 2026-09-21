import time
import json
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.config import settings
from app.models.database import get_db
from app.models.schemas import (
    QuestionRequest, GroundedAnswerResponse, GroundedTimestamp,
    FrameInfo, StructuredEvidenceItem, EvidenceItem, DebugInfo
)
from app.services.retrieval import multimodal_retrieval_service
from app.services.llm_provider import get_llm_provider
from app.utils.logger import logger

router = APIRouter()

@router.post("/{video_id}/ask", response_model=GroundedAnswerResponse)
async def ask_question(video_id: str, req: QuestionRequest):
    start_time = time.time()
    logger.info(f"Received question for video {video_id}: '{req.question}'")

    # 1. Check video exists
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
        video_row = cursor.fetchone()
        if not video_row:
            raise HTTPException(status_code=404, detail="Video not found")

    try:
        # 2. Multimodal Retrieval & Temporal Context Expansion
        retrieval_result = multimodal_retrieval_service.retrieve(
            video_id=video_id,
            query=req.question,
            top_k=req.top_k or settings.RETRIEVAL_TOP_K,
            temporal_window=req.temporal_window or settings.TEMPORAL_WINDOW_SECONDS,
            modality_weights=req.modality_weights
        )

        q_type = retrieval_result.get("detected_question_type", "MULTIMODAL")
        weights_used = retrieval_result.get("modality_weights_used", {})
        evidence = retrieval_result["evidence"]
        episodes = retrieval_result["episodes"]
        t_context = retrieval_result.get("temporal_context", {})
        temporal_seq = retrieval_result.get("temporal_sequence", {})

        # 3. Collect supporting frames (from top visual evidence, context frames, and temporal sequence)
        supporting_frames_data = []
        supporting_frames_info = []
        seen_frame_ids = set()

        # Priority 1: frames directly in retrieved evidence
        for e in evidence:
            f_path = e.get("frame_path")
            f_id = e.get("frame_id") or (Path(f_path).stem if f_path else None)
            if f_path and f_id and f_id not in seen_frame_ids:
                p_obj = Path(f_path)
                if p_obj.exists():
                    seen_frame_ids.add(f_id)
                    t_val = e.get("timestamp_start", e.get("timestamp", 0.0))
                    rel_url = e.get("frame_url") or f"/data/videos/{video_id}/frames/{p_obj.name}"
                    supporting_frames_data.append({
                        "frame_id": f_id,
                        "timestamp": t_val,
                        "image_path": str(p_obj)
                    })
                    supporting_frames_info.append(FrameInfo(
                        frame_id=f_id,
                        frame_idx=0,
                        timestamp=t_val,
                        image_url=rel_url,
                        is_keyframe=True,
                        difference_score=float(e.get("relevance_score", 0.0))
                    ))

        # Priority 2: temporal sequence frames
        for f in temporal_seq.get("frames", []):
            f_id = f.get("frame_id")
            if f_id and f_id not in seen_frame_ids:
                img_p = Path(f.get("image_path", ""))
                if img_p.exists():
                    seen_frame_ids.add(f_id)
                    rel_url = f"/data/videos/{video_id}/frames/{img_p.name}"
                    supporting_frames_data.append({
                        "frame_id": f_id,
                        "timestamp": f.get("timestamp", 0.0),
                        "image_path": str(img_p)
                    })
                    supporting_frames_info.append(FrameInfo(
                        frame_id=f_id,
                        frame_idx=0,
                        timestamp=f.get("timestamp", 0.0),
                        image_url=rel_url,
                        is_keyframe=True,
                        difference_score=f.get("difference_score", 0.0)
                    ))

        # Priority 3: context frames from temporal window
        for f in t_context.get("context_frames", []):
            f_id = f.get("frame_id")
            if f_id and f_id not in seen_frame_ids:
                seen_frame_ids.add(f_id)
                img_p = Path(f["image_path"])
                if img_p.exists():
                    rel_url = f"/data/videos/{video_id}/frames/{img_p.name}"
                    supporting_frames_data.append({
                        "frame_id": f_id,
                        "timestamp": f["timestamp"],
                        "image_path": str(img_p)
                    })
                    supporting_frames_info.append(FrameInfo(
                        frame_id=f_id,
                        frame_idx=f.get("frame_idx", 0),
                        timestamp=f["timestamp"],
                        image_url=rel_url,
                        is_keyframe=bool(f.get("is_keyframe", 1)),
                        difference_score=f.get("difference_score", 0.0)
                    ))

        # 4. Generate Grounded Answer via Modular LLM/VLM Provider
        provider = get_llm_provider()
        llm_response = provider.generate_grounded_answer(
            question=req.question,
            question_type=q_type,
            evidence=evidence,
            episodes=episodes,
            context_transcripts=t_context.get("context_transcripts", []),
            context_ocr=t_context.get("context_ocr", []),
            supporting_frames=supporting_frames_data,
            temporal_sequence=temporal_seq
        )

        latency = round(time.time() - start_time, 3)

        # Convert timestamps
        timestamps_out = [
            GroundedTimestamp(
                start=float(ts.get("start", 0.0)),
                end=float(ts.get("end", 0.0)),
                description=ts.get("description")
            )
            for ts in llm_response.get("timestamps", [])
        ]

        # Convert evidence used to StructuredEvidenceItem
        structured_evidence_out = [
            StructuredEvidenceItem(
                timestamp_start=float(e.get("timestamp_start", e.get("timestamp", 0.0))),
                timestamp_end=float(e.get("timestamp_end", e.get("timestamp", 0.0) + 2.0)),
                type=e.get("type", "visual"),
                text=e.get("text", e.get("content", "")),
                frame_path=e.get("frame_path"),
                frame_url=e.get("frame_url"),
                relevance_score=float(e.get("relevance_score", e.get("score", 0.0))),
                similarity=float(e.get("similarity", 0.0)) if e.get("similarity") is not None else None,
                temporal_score=float(e.get("temporal_score", 0.0)) if e.get("temporal_score") is not None else None
            )
            for e in evidence
        ]

        # Backward-compatible EvidenceItem list
        evidence_legacy = [
            EvidenceItem(
                type=e.type,
                timestamp=e.timestamp_start,
                end_timestamp=e.timestamp_end,
                content=e.text,
                score=e.relevance_score,
                frame_url=e.frame_url,
                frame_path=e.frame_path
            )
            for e in structured_evidence_out
        ]

        # 5. Developer Debug Info
        debug_info = DebugInfo(
            question=req.question,
            detected_question_type=q_type,
            retrieved_evidence=structured_evidence_out[:8],
            final_llm_context=llm_response.get("debug_context", ""),
            final_llm_response=llm_response.get("debug_response", ""),
            modality_weights_used=weights_used
        )

        # 6. Persist QA in database
        query_id = str(uuid.uuid4())[:12]
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO queries (query_id, video_id, question, answer, confidence, evidence_json, timestamps_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                video_id,
                req.question,
                llm_response.get("answer", ""),
                llm_response.get("confidence", 0.8),
                json.dumps([e.dict() for e in structured_evidence_out]),
                json.dumps([ts.dict() for ts in timestamps_out])
            ))
            conn.commit()

        logger.info(f"Question answered for {video_id} in {latency}s: '{llm_response.get('answer')[:60]}...'")

        return GroundedAnswerResponse(
            question=req.question,
            answer=llm_response.get("answer", ""),
            confidence=float(llm_response.get("confidence", 0.8)),
            confidence_level=llm_response.get("confidence_level", "Medium"),
            timestamps=timestamps_out,
            supporting_frames=supporting_frames_info[:8],
            evidence_used=structured_evidence_out[:8],
            evidence_breakdown=evidence_legacy[:8],
            latency_seconds=latency,
            detected_question_type=q_type,
            reasoning_summary=llm_response.get("evidence_summary"),
            debug_info=debug_info
        )

    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@router.get("/{video_id}/history")
def get_qa_history(video_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT query_id, question, answer, confidence, timestamps_json, created_at
            FROM queries WHERE video_id = ? ORDER BY created_at DESC
        """, (video_id,))
        rows = cursor.fetchall()
        return [
            {
                "query_id": r["query_id"],
                "question": r["question"],
                "answer": r["answer"],
                "confidence": r["confidence"],
                "timestamps": json.loads(r["timestamps_json"]),
                "created_at": r["created_at"]
            }
            for r in rows
        ]
