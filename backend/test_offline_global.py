"""
Test to verify the global query synthesis in OfflineHeuristicProvider.
"""
import sys
sys.path.insert(0, ".")

from app.services.llm_provider import OfflineHeuristicProvider

evidence = [
    {
        "type": "ocr",
        "timestamp_start": 0.5,
        "timestamp_end": 2.5,
        "text": "3D LiDAR Sensor Array Initialization",
        "relevance_score": 0.82,
        "similarity": 0.72,
        "temporal_score": 0.8,
        "frame_id": "frame_001",
        "frame_path": None,
        "frame_url": None,
    },
    {
        "type": "ocr",
        "timestamp_start": 6.0,
        "timestamp_end": 8.0,
        "text": "Phase 2: Obstacle Alert 2.0m - Rapid Left Turn",
        "relevance_score": 0.75,
        "similarity": 0.65,
        "temporal_score": 0.6,
        "frame_id": "frame_010",
        "frame_path": None,
        "frame_url": None,
    },
    {
        "type": "ocr",
        "timestamp_start": 12.5,
        "timestamp_end": 14.5,
        "text": "Phase 3: Waypoint Bravo Reached Successfully",
        "relevance_score": 0.70,
        "similarity": 0.60,
        "temporal_score": 0.5,
        "frame_id": "frame_020",
        "frame_path": None,
        "frame_url": None,
    },
    {
        "type": "transcript",
        "timestamp_start": 1.0,
        "timestamp_end": 5.0,
        "text": "Phase 1 LiDAR initialization complete",
        "relevance_score": 0.68,
        "similarity": 0.58,
        "temporal_score": 0.5,
        "frame_id": None,
        "frame_path": None,
        "frame_url": None,
    },
]

context_ocr = [
    {"timestamp": 0.5, "text": "Phase 1: 3D LiDAR Sensor Initialization"},
    {"timestamp": 6.0, "text": "Phase 2: Obstacle Alert at 2.0m"},
    {"timestamp": 12.5, "text": "Phase 3: Waypoint Bravo Reached Successfully"},
]

context_transcripts = [
    {"start_time": 1.0, "end_time": 5.0, "text": "Phase 1 LiDAR initialization complete"},
    {"start_time": 5.5, "end_time": 11.0, "text": "Obstacle detected at 2.0m rapid left turn"},
    {"start_time": 11.5, "end_time": 17.0, "text": "Waypoint Bravo reached mission success"},
]

provider = OfflineHeuristicProvider()

queries = [
    ("what happens in all phases", "MULTIMODAL"),
    ("what is happening in the video", "MULTIMODAL"),
    ("summarize the video", "MULTIMODAL"),
    ("what are the phases", "MULTIMODAL"),
]

for question, qtype in queries:
    result = provider.generate_grounded_answer(
        question=question,
        question_type=qtype,
        evidence=evidence,
        episodes=[{"start": 0.0, "end": 18.0}],
        context_transcripts=context_transcripts,
        context_ocr=context_ocr,
        supporting_frames=[],
        temporal_sequence={}
    )
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"Confidence: {result['confidence_level']}")
    print(f"Answer:\n{result['answer'][:500]}")
    assert "don't have enough evidence" not in result["answer"].lower(), "FAIL: returned unanswerable!"
    print("PASS: Got a substantive answer!")
