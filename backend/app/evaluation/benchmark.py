import time
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from app.config import settings
from app.models.database import get_db
from app.services.retrieval import multimodal_retrieval_service
from app.services.llm_provider import get_llm_provider
from app.utils.logger import logger

def compute_temporal_iou(p_start: float, p_end: float, gt_start: float, gt_end: float) -> float:
    """Calculates temporal Intersection-over-Union between predicted window and ground truth."""
    inter_start = max(p_start, gt_start)
    inter_end = min(p_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    union_start = min(p_start, gt_start)
    union_end = max(p_end, gt_end)
    union = max(1e-6, union_end - union_start)
    
    return min(1.0, round(intersection / union, 4))

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Computes token-level F1 overlap between predicted answer and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)

class BenchmarkRunner:
    def __init__(self):
        self.dataset_path = Path(__file__).resolve().parent.parent.parent.parent / "demo_data" / "evaluation_benchmark.json"

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not self.dataset_path.exists():
            return []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_experiment(self, video_id: str, experiment_type: str) -> Dict[str, Any]:
        """
        Runs one of the 4 experimental conditions:
        - text_only (Experiment A)
        - visual_only (Experiment B)
        - multimodal (Experiment C)
        - multimodal_temporal (Experiment D)
        """
        dataset = self.load_dataset()
        if not dataset:
            raise ValueError("Evaluation dataset not found")

        # Configure weights and options per experiment
        if experiment_type == "text_only":
            weights = {"text": 1.0, "visual": 0.0, "ocr": 0.0, "temporal": 0.0}
            use_temporal = False
            label = "Experiment A: Text-Only RAG"
        elif experiment_type == "visual_only":
            weights = {"text": 0.0, "visual": 1.0, "ocr": 0.0, "temporal": 0.0}
            use_temporal = False
            label = "Experiment B: Visual-Only Retrieval"
        elif experiment_type == "multimodal":
            weights = {"text": 0.45, "visual": 0.35, "ocr": 0.20, "temporal": 0.0}
            use_temporal = False
            label = "Experiment C: Multimodal (No Temporal)"
        else: # multimodal_temporal
            weights = {"text": 0.35, "visual": 0.35, "ocr": 0.20, "temporal": 0.10}
            use_temporal = True
            label = "Experiment D: Multimodal + Temporal"

        recalls = []
        ious = []
        f1s = []
        latencies = []

        provider = get_llm_provider()

        for item in dataset:
            t0 = time.time()
            q = item["question"]
            gt_start = item["relevant_start"]
            gt_end = item["relevant_end"]
            gt_ans = item["ground_truth_answer"]

            # Run retrieval
            retrieval = multimodal_retrieval_service.retrieve(
                video_id=video_id,
                query=q,
                top_k=5,
                temporal_window=4.0,
                modality_weights=weights,
                include_temporal_expansion=use_temporal
            )

            # Check recall@k: did any retrieved evidence hit within the ground truth interval?
            hit = any(
                (gt_start <= e.get("timestamp_start", e.get("timestamp", 0.0)) <= gt_end)
                for e in retrieval["evidence"]
            )
            recalls.append(1.0 if hit else 0.0)

            # Run answer generation
            t_context = retrieval.get("temporal_context", {})
            supporting_frames = []
            for f in t_context.get("context_frames", []):
                supporting_frames.append({
                    "frame_id": f.get("frame_id"),
                    "timestamp": f.get("timestamp", 0.0),
                    "image_path": f.get("image_path")
                })

            resp = provider.generate_grounded_answer(
                question=q,
                question_type=retrieval.get("detected_question_type", "MULTIMODAL"),
                evidence=retrieval["evidence"],
                episodes=retrieval["episodes"],
                context_transcripts=t_context.get("context_transcripts", []),
                context_ocr=t_context.get("context_ocr", []),
                supporting_frames=supporting_frames,
                temporal_sequence=retrieval.get("temporal_sequence", {})
            )

            latency_ms = (time.time() - t0) * 1000
            latencies.append(latency_ms)

            # Calculate Grounding IoU
            pred_timestamps = resp.get("timestamps", [])
            if pred_timestamps:
                best_iou = max(
                    compute_temporal_iou(ts["start"], ts["end"], gt_start, gt_end)
                    for ts in pred_timestamps
                )
            elif retrieval["episodes"]:
                best_iou = max(
                    compute_temporal_iou(ep["start"], ep["end"], gt_start, gt_end)
                    for ep in retrieval["episodes"]
                )
            else:
                best_iou = 0.0
            ious.append(best_iou)

            # Calculate Token F1
            f1 = compute_token_f1(resp.get("answer", ""), gt_ans)
            f1s.append(f1)

        avg_recall = round(sum(recalls) / len(recalls), 4)
        avg_iou = round(sum(ious) / len(ious), 4)
        avg_f1 = round(sum(f1s) / len(f1s), 4)
        avg_latency = round(sum(latencies) / len(latencies), 1)

        result = {
            "experiment": label,
            "experiment_key": experiment_type,
            "recall_at_k": avg_recall,
            "grounding_iou": avg_iou,
            "avg_latency_ms": avg_latency,
            "answer_accuracy": avg_f1,
            "total_evaluated": len(dataset)
        }

        # Persist to database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations (eval_id, video_id, experiment_type, question, recall_at_k, grounding_iou, latency_seconds, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"eval_{experiment_type}_{int(time.time())}",
                video_id,
                experiment_type,
                "dataset_benchmark_suite",
                avg_recall,
                avg_iou,
                round(avg_latency / 1000.0, 3),
                json.dumps(result)
            ))
            conn.commit()

        logger.info(f"Completed {label}: Recall@K={avg_recall}, IoU={avg_iou}, Latency={avg_latency}ms")
        return result

benchmark_runner = BenchmarkRunner()
