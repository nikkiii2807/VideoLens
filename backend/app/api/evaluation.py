from fastapi import APIRouter, HTTPException
from typing import List
from app.models.database import get_db
from app.models.schemas import BenchmarkRunRequest, EvaluationResult
from app.evaluation.benchmark import benchmark_runner
from app.utils.logger import logger

router = APIRouter()

@router.post("/run", response_model=List[EvaluationResult])
def run_benchmark(req: BenchmarkRunRequest):
    # Find any completed video, or use the latest
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT video_id FROM videos WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            # Check if any video uploaded
            cursor.execute("SELECT video_id FROM videos ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=400, detail="No processed video available for benchmark. Please load or upload a video first.")
        
        video_id = row[0]

    logger.info(f"Running scientific benchmark evaluation on video {video_id}...")

    # If "all" or "multimodal_temporal", execute the 4-way comparison suite
    experiments = ["text_only", "visual_only", "multimodal", "multimodal_temporal"]
    if req.experiment_type in experiments:
        # Run specific experiment or all
        experiments = [req.experiment_type] if req.experiment_type != "multimodal_temporal" else experiments

    results = []
    for exp in experiments:
        try:
            res = benchmark_runner.run_experiment(video_id, exp)
            results.append(EvaluationResult(
                experiment=res["experiment"],
                recall_at_k=res["recall_at_k"],
                grounding_iou=res["grounding_iou"],
                avg_latency_ms=res["avg_latency_ms"],
                answer_accuracy=res["answer_accuracy"],
                total_evaluated=res["total_evaluated"]
            ))
        except Exception as e:
            logger.error(f"Failed to run benchmark for {exp}: {e}")

    return results

@router.get("/results", response_model=List[EvaluationResult])
def get_benchmark_history():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT metadata_json FROM evaluations
            ORDER BY created_at DESC LIMIT 10
        """)
        rows = cursor.fetchall()
        import json
        results = []
        for r in rows:
            if r[0]:
                data = json.loads(r[0])
                results.append(EvaluationResult(
                    experiment=data["experiment"],
                    recall_at_k=data["recall_at_k"],
                    grounding_iou=data["grounding_iou"],
                    avg_latency_ms=data["avg_latency_ms"],
                    answer_accuracy=data["answer_accuracy"],
                    total_evaluated=data["total_evaluated"]
                ))
        return results
