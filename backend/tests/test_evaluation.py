import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.database import init_db, get_db
from app.models.schemas import BenchmarkRunRequest
from app.api.evaluation import run_benchmark

def test_evaluation_suite():
    init_db()
    
    # Check we have at least one completed video in database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT video_id FROM videos WHERE status = 'completed' LIMIT 1")
        row = cursor.fetchone()
        assert row is not None, "Need at least one processed video to run benchmark"
        video_id = row[0]

    print(f"Running evaluation benchmark on video {video_id}...")
    req = BenchmarkRunRequest(experiment_type="multimodal_temporal")
    results = run_benchmark(req)

    print("\n--- SCIENTIFIC EVALUATION RESULTS ---")
    for r in results:
        print(f"[{r.experiment}]: Recall@K={r.recall_at_k * 100:.1f}%, IoU={r.grounding_iou * 100:.1f}%, F1={r.answer_accuracy * 100:.1f}%, Latency={r.avg_latency_ms:.0f}ms")

    assert len(results) >= 4, "Expected 4 experiment conditions evaluated"
    
    # Verify metrics are within valid ranges
    for r in results:
        assert 0.0 <= r.recall_at_k <= 1.0
        assert 0.0 <= r.grounding_iou <= 1.0
        assert 0.0 <= r.answer_accuracy <= 1.0
        assert r.avg_latency_ms > 0

    print("\nALL EVALUATION CHECKS PASSED!")

if __name__ == "__main__":
    test_evaluation_suite()
