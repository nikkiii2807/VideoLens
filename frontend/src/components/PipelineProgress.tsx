import React from 'react';
import { CheckCircle2, Loader2, Circle, AlertCircle } from 'lucide-react';
import type { VideoStatus } from '../types';

interface PipelineProgressProps {
  status: VideoStatus | null;
}

interface Step {
  id: string;
  label: string;
  stageKey: string;
}

const STEPS: Step[] = [
  { id: '1', label: 'Video uploaded & validated', stageKey: 'uploaded' },
  { id: '2', label: 'Frames extracted & filtered', stageKey: 'sampling' },
  { id: '3', label: 'Speech transcribed (Whisper)', stageKey: 'transcribing' },
  { id: '4', label: 'Selective OCR completed', stageKey: 'ocr' },
  { id: '5', label: 'Multimodal embeddings generated', stageKey: 'embedding' },
  { id: '6', label: 'FAISS multimodal index created', stageKey: 'indexing' },
];

export const PipelineProgress: React.FC<PipelineProgressProps> = ({ status }) => {
  if (!status) return null;

  const getStepState = (stepIndex: number) => {
    if (status.status === 'failed') return 'error';
    if (status.status === 'completed') return 'completed';

    const stageOrder = ['uploaded', 'sampling', 'transcribing', 'ocr', 'embedding', 'indexing', 'ready'];
    const currentIdx = stageOrder.indexOf(status.stage);

    if (currentIdx > stepIndex) return 'completed';
    if (currentIdx === stepIndex && status.status === 'processing') return 'active';
    return 'pending';
  };

  return (
    <div className="glass-panel rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
          Video Understanding Pipeline
        </h3>
        <div className="flex items-center space-x-2">
          {status.status === 'processing' && (
            <span className="flex items-center space-x-1.5 text-xs text-accent-cyan font-medium animate-pulse">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Processing ({status.progress}%)</span>
            </span>
          )}
          {status.status === 'completed' && (
            <span className="flex items-center space-x-1 text-xs text-brand-500 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Index Ready</span>
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 pt-1">
        {STEPS.map((step, idx) => {
          const state = getStepState(idx);

          return (
            <div
              key={step.id}
              className={`p-2.5 rounded-lg border text-xs flex items-center space-x-2 transition-all ${
                state === 'completed'
                  ? 'bg-brand-500/10 border-brand-500/30 text-slate-200'
                  : state === 'active'
                  ? 'bg-accent-cyan/10 border-accent-cyan/40 text-accent-cyan glow-accent'
                  : state === 'error'
                  ? 'bg-accent-rose/10 border-accent-rose/30 text-accent-rose'
                  : 'bg-dark-800/40 border-dark-700/60 text-slate-500'
              }`}
            >
              {state === 'completed' && <CheckCircle2 className="w-4 h-4 text-brand-500 shrink-0" />}
              {state === 'active' && <Loader2 className="w-4 h-4 text-accent-cyan animate-spin shrink-0" />}
              {state === 'error' && <AlertCircle className="w-4 h-4 text-accent-rose shrink-0" />}
              {state === 'pending' && <Circle className="w-4 h-4 text-dark-600 shrink-0" />}

              <span className="text-[11px] leading-tight font-medium">{step.label}</span>
            </div>
          );
        })}
      </div>

      {status.frames_count > 0 && (
        <div className="pt-2 border-t border-dark-700 flex items-center space-x-4 text-[11px] text-slate-400 font-mono">
          <span>Keyframes: <strong className="text-slate-200">{status.frames_count}</strong></span>
          <span>&bull;</span>
          <span>Transcripts: <strong className="text-slate-200">{status.transcript_segments_count}</strong></span>
          <span>&bull;</span>
          <span>OCR Detections: <strong className="text-slate-200">{status.ocr_detections_count}</strong></span>
        </div>
      )}
    </div>
  );
};
