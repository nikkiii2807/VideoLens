import type { VideoMetadata, VideoStatus, FrameInfo, TranscriptSegment, GroundedAnswer, BenchmarkMetrics } from '../types';

export const api = {
  async uploadVideo(file: File): Promise<VideoMetadata> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/video/upload', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async loadDemoVideo(demoName: string = 'lecture'): Promise<VideoMetadata> {
    const res = await fetch(`/api/video/load-demo?demo_name=${encodeURIComponent(demoName)}`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to load demo video');
    }
    return res.json();
  },

  async processVideo(videoId: string): Promise<void> {
    const res = await fetch(`/api/video/${videoId}/process`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start processing');
    }
  },

  async getVideoStatus(videoId: string): Promise<VideoStatus> {
    const res = await fetch(`/api/video/${videoId}/status`);
    if (!res.ok) {
      throw new Error('Failed to fetch video status');
    }
    return res.json();
  },

  async getVideoFrames(videoId: string): Promise<FrameInfo[]> {
    const res = await fetch(`/api/video/${videoId}/frames`);
    if (!res.ok) {
      throw new Error('Failed to fetch frames');
    }
    return res.json();
  },

  async getVideoTranscript(videoId: string): Promise<TranscriptSegment[]> {
    const res = await fetch(`/api/video/${videoId}/transcript`);
    if (!res.ok) {
      throw new Error('Failed to fetch transcript');
    }
    return res.json();
  },

  async askQuestion(
    videoId: string,
    question: string,
    options?: {
      temporal_window?: number;
      top_k?: number;
      modality_weights?: Record<string, number>;
    }
  ): Promise<GroundedAnswer> {
    const res = await fetch(`/api/video/${videoId}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        temporal_window: options?.temporal_window,
        top_k: options?.top_k,
        modality_weights: options?.modality_weights,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to generate grounded answer');
    }
    return res.json();
  },

  async runBenchmark(experimentType: string): Promise<BenchmarkMetrics[]> {
    const res = await fetch('/api/evaluation/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        experiment_type: experimentType,
      }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Benchmark run failed');
    }
    return res.json();
  },

  async getBenchmarkResults(): Promise<BenchmarkMetrics[]> {
    const res = await fetch('/api/evaluation/results');
    if (!res.ok) {
      throw new Error('Failed to fetch benchmark results');
    }
    return res.json();
  }
};
