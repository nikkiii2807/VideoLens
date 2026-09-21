from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import settings
from app.models.database import get_db
from app.services.embedding_service import embedding_service
from app.services.vector_store import VectorStore
from app.services.temporal_reasoning import temporal_reasoning_service
from app.services.question_classifier import question_classifier
from app.utils.logger import logger

class MultimodalRetrievalService:
    def __init__(self):
        self.default_top_k = settings.RETRIEVAL_TOP_K
        self.default_window = settings.TEMPORAL_WINDOW_SECONDS

    def retrieve(
        self,
        video_id: str,
        query: str,
        top_k: Optional[int] = None,
        temporal_window: Optional[float] = None,
        modality_weights: Optional[Dict[str, float]] = None,
        include_temporal_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Executes multimodal hybrid retrieval:
        1. Classifies question into VISUAL / ACTION / TEMPORAL / TRANSCRIPT / OCR / MULTIMODAL.
        2. Embeds query cross-modally (OpenCLIP for visual, SentenceTransformers for text/OCR).
        3. Identifies true primary seed timestamp across all modalities.
        4. Calculates multimodal weighted relevance scores using question-type weights.
        5. Deduplicates adjacent frames to prevent redundancy.
        6. Preserves structured evidence objects with exact timestamps.
        7. Expands temporal context and builds chronological event sequence.
        """
        k = top_k or self.default_top_k
        window_sec = temporal_window or self.default_window

        # 1. Question Classification & Dynamic Weights
        q_type, default_weights = question_classifier.classify(query)
        weights = dict(default_weights)
        if modality_weights:
            weights.update(modality_weights)

        # Normalize weights so they sum to 1.0
        total_w = sum(weights.values()) or 1.0
        weights = {k_w: round(v_w / total_w, 4) for k_w, v_w in weights.items()}

        # Detect global/overview queries that need full-video coverage
        GLOBAL_QUERY_KEYWORDS = [
            "what happens", "what is happening", "what happened", "summarize", "summary",
            "overview", "all phases", "all steps", "all stages", "describe the video",
            "what does the video show", "what is shown", "tell me about", "what occurs",
            "whole video", "entire video", "throughout the video",
            "what are the phases", "what are the stages", "what are the steps"
        ]
        q_lower = query.strip().lower()
        is_global_query = any(kw in q_lower for kw in GLOBAL_QUERY_KEYWORDS)

        # For global queries: expand retrieval to cover full timeline
        if is_global_query:
            k = max(k, 20)  # retrieve more candidates
            window_sec = 999.0  # effectively no temporal clamping
            logger.info(f"Global overview query detected: '{query}' — expanding retrieval across full timeline.")

        # 2. Embed query across modalities
        q_clip = embedding_service.embed_text_clip(query)
        q_text = embedding_service.embed_text_semantic(query)

        # 3. Vector search in FAISS
        store = VectorStore(video_id)
        visual_hits = store.search_visual(q_clip, top_k=k * 2)
        text_hits = store.search_text(q_text, top_k=k * 2)

        # 4. Identify primary seed timestamp (highest similarity across both modalities)
        all_raw_hits = []
        for v in visual_hits:
            all_raw_hits.append((v.get("similarity", 0.0), v["timestamp"], "visual"))
        for t in text_hits:
            all_raw_hits.append((t.get("similarity", 0.0), t["timestamp"], t.get("type", "text")))

        all_raw_hits.sort(key=lambda x: x[0], reverse=True)
        primary_t = all_raw_hits[0][1] if all_raw_hits else 0.0

        # For global queries: spread seeds evenly across video so temporal score doesn't crush distant frames
        if is_global_query and all_raw_hits:
            all_timestamps = sorted(set([h[1] for h in all_raw_hits]))
            if len(all_timestamps) > 1:
                # Compute median as a neutral center — temporal proximity from it is close to 1.0 for most hits
                primary_t = all_timestamps[len(all_timestamps) // 2]
            logger.info(f"Global query: overriding primary_seed to median timestamp {primary_t:.2f}s for full coverage.")

        # Load frame OCR map from SQLite for enriching visual context
        frame_ocr_map: Dict[str, List[str]] = {}
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT frame_id, text FROM ocr_chunks WHERE video_id = ?", (video_id,))
            for fid, o_text in cursor.fetchall():
                frame_ocr_map.setdefault(fid, []).append(o_text)

        # 5. Score and construct candidates
        all_evidence = []

        # Process visual candidates
        seen_visual_timestamps = []
        for v in visual_hits:
            vis_sim = max(0.0, min(1.0, float(v.get("similarity", 0.0))))
            t_val = float(v["timestamp"])

            # Deduplication: skip if a frame within 1.0s has already been accepted with higher score
            if any(abs(t_val - prev_t) < 1.0 for prev_t in seen_visual_timestamps):
                continue
            seen_visual_timestamps.append(t_val)

            temp_score = temporal_reasoning_service.compute_temporal_proximity(t_val, primary_t, window_sec)
            
            # Weighted multimodal score
            rel_score = (
                weights["visual"] * vis_sim +
                weights["temporal"] * temp_score
            ) / (weights["visual"] + weights["temporal"] or 1.0)

            frame_id = v.get("frame_id", "")
            img_path = Path(v.get("image_path", ""))
            rel_url = f"/data/videos/{video_id}/frames/{img_path.name}" if img_path.name else ""

            # Enrich visual text with detected OCR if present
            ocr_detected = frame_ocr_map.get(frame_id, [])
            if ocr_detected:
                vis_text = f"Visual Keyframe at {t_val:.1f}s [Onscreen OCR: '{', '.join(ocr_detected)}']"
            else:
                vis_text = f"Visual Keyframe at {t_val:.1f}s"

            all_evidence.append({
                "timestamp_start": round(t_val, 2),
                "timestamp_end": round(t_val + 1.5, 2),
                "type": "visual",
                "text": vis_text,
                "frame_path": str(img_path) if img_path.exists() else None,
                "frame_url": rel_url,
                "relevance_score": round(float(rel_score), 4),
                "similarity": round(vis_sim, 4),
                "temporal_score": round(temp_score, 4),
                "frame_id": frame_id,
                # Backward compatibility
                "content": vis_text,
                "timestamp": round(t_val, 2),
                "score": round(float(rel_score), 4)
            })

        # Process text candidates (transcripts and OCR)
        for t in text_hits:
            text_sim = max(0.0, min(1.0, float(t.get("similarity", 0.0))))
            t_val = float(t["timestamp"])
            temp_score = temporal_reasoning_service.compute_temporal_proximity(t_val, primary_t, window_sec)
            item_type = t.get("type", "transcript")

            w_mod = weights["ocr"] if item_type == "ocr" else weights["text"]
            rel_score = (
                w_mod * text_sim +
                weights["temporal"] * temp_score
            ) / (w_mod + weights["temporal"] or 1.0)

            frame_id = t.get("frame_id")
            frame_url = None
            frame_path = None
            if item_type == "ocr" and frame_id:
                frame_url = f"/data/videos/{video_id}/frames/{frame_id}.jpg"
                expected_p = settings.DATA_DIR / "videos" / video_id / "frames" / f"{frame_id}.jpg"
                if expected_p.exists():
                    frame_path = str(expected_p)

            t_start = round(t_val, 2)
            t_end = round(float(t.get("end_time", t_val + 2.0)), 2)

            all_evidence.append({
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "type": item_type,
                "text": str(t.get("text", "")),
                "frame_path": frame_path,
                "frame_url": frame_url,
                "relevance_score": round(float(rel_score), 4),
                "similarity": round(text_sim, 4),
                "temporal_score": round(temp_score, 4),
                "frame_id": frame_id,
                # Backward compatibility
                "content": str(t.get("text", "")),
                "timestamp": t_start,
                "score": round(float(rel_score), 4)
            })

        # Sort all retrieved evidence by final multimodal relevance score descending
        all_evidence.sort(key=lambda x: x["relevance_score"], reverse=True)

        # 6. Temporal Sequence & Episodes
        episodes = []
        temporal_context = {}
        temporal_sequence = {}
        if include_temporal_expansion and all_evidence:
            if is_global_query:
                # For global queries, use ALL evidence seeds to cover full video timeline
                all_seeds = sorted(set([e["timestamp_start"] for e in all_evidence]))
                raw_episodes = temporal_reasoning_service.merge_temporal_windows(
                    all_seeds,
                    window_sec=min(window_sec, 5.0)  # merge nearby but not the entire video into one blob
                )
                gather_max_frames = 15
            else:
                all_seeds = [e["timestamp_start"] for e in all_evidence[:3]]
                raw_episodes = temporal_reasoning_service.merge_temporal_windows(
                    all_seeds,
                    window_sec=window_sec
                )
                gather_max_frames = 5

            episodes = [{"start": round(s, 2), "end": round(e, 2)} for s, e in raw_episodes]
            temporal_context = temporal_reasoning_service.gather_temporal_context(video_id, raw_episodes)
            temporal_sequence = temporal_reasoning_service.get_temporal_event_sequence(
                video_id=video_id,
                center_timestamp=primary_t,
                window_sec=window_sec,
                max_frames=gather_max_frames
            )

        logger.info(
            f"Retrieval for query '{query}': type={q_type}, evidence={len(all_evidence)}, "
            f"primary_seed={primary_t}s, episodes={len(episodes)}"
        )

        return {
            "query": query,
            "detected_question_type": q_type,
            "modality_weights_used": weights,
            "evidence": all_evidence[:k * 2],
            "episodes": episodes,
            "temporal_context": temporal_context,
            "temporal_sequence": temporal_sequence,
            "primary_seed": primary_t
        }

multimodal_retrieval_service = MultimodalRetrievalService()
