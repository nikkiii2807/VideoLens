export interface VideoMetadata {
  video_id: string;
  filename: string;
  duration: number;
  fps: number;
  width: number;
  height: number;
  file_size_bytes: number;
  status: string;
  video_url: string;
}

export interface VideoStatus {
  video_id: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  stage: 'sampling' | 'transcribing' | 'ocr' | 'embedding' | 'indexing' | 'ready' | 'error';
  progress: number;
  error_message?: string | null;
  frames_count: number;
  transcript_segments_count: number;
  ocr_detections_count: number;
}

export interface FrameInfo {
  frame_id: string;
  frame_idx: number;
  timestamp: number;
  image_url: string;
  is_keyframe: boolean;
  has_ocr?: boolean;
  difference_score?: number;
}

export interface TranscriptSegment {
  chunk_id: string;
  start_time: number;
  end_time: number;
  text: string;
}

export interface GroundedTimestamp {
  start: number;
  end: number;
  description?: string;
}

export interface StructuredEvidenceItem {
  timestamp_start: number;
  timestamp_end: number;
  type: 'visual' | 'transcript' | 'ocr';
  text: string;
  frame_path?: string | null;
  frame_url?: string | null;
  relevance_score: number;
  similarity?: number;
  temporal_score?: number;
}

export interface EvidenceItem {
  type: 'visual' | 'transcript' | 'ocr';
  timestamp: number;
  end_timestamp?: number | null;
  content: string;
  score: number;
  frame_url?: string | null;
  similarity?: number;
  temporal_score?: number;
}

export interface DebugInfo {
  question: string;
  detected_question_type: string;
  retrieved_evidence: StructuredEvidenceItem[];
  final_llm_context: string;
  final_llm_response: string;
  modality_weights_used: Record<string, number>;
}

export interface GroundedAnswer {
  question: string;
  answer: string;
  confidence: number;
  confidence_level?: string;
  timestamps: GroundedTimestamp[];
  supporting_frames: FrameInfo[];
  evidence_used: StructuredEvidenceItem[];
  evidence_breakdown?: EvidenceItem[];
  latency_seconds: number;
  detected_question_type?: string;
  reasoning_summary?: string;
  debug_info?: DebugInfo;
}

export interface BenchmarkMetrics {
  experiment: string;
  recall_at_k: number;
  grounding_iou: number;
  avg_latency_ms: number;
  answer_accuracy: number;
  total_evaluated: number;
}
