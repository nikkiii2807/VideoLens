from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Video Upload & Status
class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    duration: float
    fps: float
    width: int
    height: int
    file_size_bytes: int
    status: str
    video_url: str

class VideoStatusResponse(BaseModel):
    video_id: str
    status: str # "uploaded", "processing", "completed", "failed"
    stage: str  # "sampling", "transcribing", "ocr", "embedding", "indexing", "ready"
    progress: int # 0 to 100
    error_message: Optional[str] = None
    frames_count: int = 0
    transcript_segments_count: int = 0
    ocr_detections_count: int = 0

# Processing Pipeline Stage
class ProcessingStageStatus(BaseModel):
    name: str
    status: str # "pending", "in_progress", "completed", "skipped", "failed"
    details: Optional[str] = None

# Frame and Chunk Models
class FrameInfo(BaseModel):
    frame_id: str
    frame_idx: int
    timestamp: float
    image_url: str
    is_keyframe: bool
    has_ocr: bool = False
    difference_score: Optional[float] = 0.0

class TranscriptSegment(BaseModel):
    chunk_id: str
    start_time: float
    end_time: float
    text: str

class OCRDetection(BaseModel):
    ocr_id: str
    frame_id: str
    timestamp: float
    text: str
    confidence: float

# QA and Grounded Reasoning
class QuestionRequest(BaseModel):
    question: str
    temporal_window: Optional[float] = None
    top_k: Optional[int] = None
    modality_weights: Optional[Dict[str, float]] = None

class GroundedTimestamp(BaseModel):
    start: float
    end: float
    description: Optional[str] = None

# Structured Evidence Item per Requirement 2
class StructuredEvidenceItem(BaseModel):
    timestamp_start: float
    timestamp_end: float
    type: str # "visual" | "transcript" | "ocr"
    text: str
    frame_path: Optional[str] = None
    frame_url: Optional[str] = None
    relevance_score: float
    similarity: Optional[float] = None
    temporal_score: Optional[float] = None

# Backward-compatible alias for existing components
class EvidenceItem(BaseModel):
    type: str # "visual", "transcript", "ocr"
    timestamp: float
    end_timestamp: Optional[float] = None
    content: str
    score: float
    frame_url: Optional[str] = None
    frame_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DebugInfo(BaseModel):
    question: str
    detected_question_type: str
    retrieved_evidence: List[StructuredEvidenceItem]
    final_llm_context: str
    final_llm_response: str
    modality_weights_used: Dict[str, float]

class GroundedAnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    confidence_level: Optional[str] = "Medium" # "High" | "Medium" | "Low"
    timestamps: List[GroundedTimestamp]
    supporting_frames: List[FrameInfo]
    evidence_used: List[StructuredEvidenceItem]
    evidence_breakdown: Optional[List[EvidenceItem]] = None
    latency_seconds: float
    detected_question_type: Optional[str] = "MULTIMODAL"
    reasoning_summary: Optional[str] = None
    debug_info: Optional[DebugInfo] = None

# Benchmark & Evaluation
class BenchmarkRunRequest(BaseModel):
    experiment_type: str = "multimodal_temporal" # "text_only", "visual_only", "multimodal", "multimodal_temporal"

class EvaluationResult(BaseModel):
    experiment: str
    recall_at_k: float
    grounding_iou: float
    avg_latency_ms: float
    answer_accuracy: float
    total_evaluated: int
