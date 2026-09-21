import React, { useState } from 'react';
import {
  Check,
  Copy,
  Clock,
  ShieldCheck,
  Zap,
  Image as ImageIcon,
  Terminal,
  ChevronDown,
  ChevronUp,
  Code,
  AlertTriangle
} from 'lucide-react';
import type { GroundedAnswer as GroundedAnswerType } from '../types';

interface GroundedAnswerProps {
  answerData: GroundedAnswerType;
  onSeek: (timestamp: number, endTimestamp?: number) => void;
}

export const GroundedAnswer: React.FC<GroundedAnswerProps> = ({ answerData, onSeek }) => {
  const [copied, setCopied] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [copiedContext, setCopiedContext] = useState(false);
  const [copiedResponse, setCopiedResponse] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(answerData.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const copyDebugItem = (text: string, isCtx: boolean) => {
    navigator.clipboard.writeText(text);
    if (isCtx) {
      setCopiedContext(true);
      setTimeout(() => setCopiedContext(false), 2000);
    } else {
      setCopiedResponse(true);
      setTimeout(() => setCopiedResponse(false), 2000);
    }
  };

  const formatSec = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const tenths = Math.floor((sec % 1) * 10);
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${tenths}`;
  };

  const confLevel = answerData.confidence_level || (
    answerData.confidence >= 0.85 ? 'High' : answerData.confidence >= 0.5 ? 'Medium' : 'Low'
  );

  const confBadgeStyle =
    confLevel === 'High'
      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
      : confLevel === 'Medium'
      ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
      : 'bg-rose-500/10 border-rose-500/30 text-rose-400';

  const qType = answerData.detected_question_type || 'MULTIMODAL';

  const isLowConfidence = confLevel === 'Low';

  return (
    <div className="glass-panel rounded-xl p-5 space-y-5 border-brand-500/20 shadow-xl">
      {/* Header with Badges & Actions */}
      <div className="flex items-center justify-between border-b border-dark-700 pb-3">
        <div className="flex items-center space-x-2">
          <div className="p-1 rounded bg-brand-500/20 text-brand-500">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
            Grounded Answer
          </span>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30 uppercase">
            {qType}
          </span>
        </div>

        <div className="flex items-center space-x-2 text-xs">
          {/* Confidence Badge */}
          <div className={`flex items-center space-x-1 px-2.5 py-0.5 rounded-full border font-mono text-[11px] ${confBadgeStyle}`}>
            {isLowConfidence && <AlertTriangle className="w-3 h-3 text-rose-400 mr-0.5" />}
            <span>Confidence:</span>
            <strong>{confLevel} ({Math.round(answerData.confidence * 100)}%)</strong>
          </div>

          {/* Latency */}
          <div className="flex items-center space-x-1 text-slate-400 font-mono text-[11px]">
            <Zap className="w-3 h-3 text-accent-amber" />
            <span>{answerData.latency_seconds}s</span>
          </div>

          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="p-1 text-slate-400 hover:text-slate-200 hover:bg-dark-700 rounded transition-colors"
            title="Copy answer"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-brand-500" /> : <Copy className="w-3.5 h-3.5" />}
          </button>

          {/* Debug Toggle Button */}
          <button
            onClick={() => setShowDebug(!showDebug)}
            className={`px-2 py-1 rounded text-[11px] font-mono flex items-center space-x-1 transition-colors border ${
              showDebug
                ? 'bg-accent-violet/20 border-accent-violet text-accent-violet'
                : 'bg-dark-800 border-dark-600 text-slate-400 hover:text-slate-200 hover:border-slate-500'
            }`}
            title="Toggle Developer Debug Mode"
          >
            <Terminal className="w-3 h-3" />
            <span>Debug</span>
            {showDebug ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* ANSWER Section */}
      <div className="space-y-1.5">
        <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <span>ANSWER</span>
        </div>
        <div className={`p-3.5 rounded-xl border ${isLowConfidence ? 'bg-rose-950/20 border-rose-800/40 text-rose-200' : 'bg-dark-800/60 border-dark-700 text-slate-100'}`}>
          <p className="text-sm font-medium leading-relaxed whitespace-pre-line">
            {answerData.answer}
          </p>
        </div>
      </div>

      {/* EVIDENCE Section */}
      <div className="space-y-3 pt-2 border-t border-dark-700">
        <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <Clock className="w-3.5 h-3.5 text-accent-cyan" />
          <span>EVIDENCE</span>
        </div>

        {/* Clickable Timestamps */}
        {answerData.timestamps && answerData.timestamps.length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {answerData.timestamps.map((ts, idx) => (
              <button
                key={idx}
                onClick={() => onSeek(ts.start, ts.end)}
                className="group flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-dark-800 hover:bg-accent-cyan/20 border border-dark-600 hover:border-accent-cyan/60 text-xs font-mono transition-all hover:scale-[1.02]"
                title="Click to seek video"
              >
                <span className="text-base">⏱</span>
                <span className="text-accent-cyan font-bold text-xs">
                  {formatSec(ts.start)} – {formatSec(ts.end)}
                </span>
                {ts.description && (
                  <span className="text-[11px] text-slate-400 font-sans group-hover:text-slate-200 truncate max-w-[200px]">
                    &bull; {ts.description}
                  </span>
                )}
              </button>
            ))}
          </div>
        ) : (
          <div className="text-xs text-slate-500 italic">
            No specific video timestamp intervals matched for this query.
          </div>
        )}

        {/* Supporting Video Frames */}
        {answerData.supporting_frames && answerData.supporting_frames.length > 0 && (
          <div className="space-y-2 pt-1">
            <div className="flex items-center space-x-1.5 text-xs text-slate-400 font-medium">
              <ImageIcon className="w-3.5 h-3.5 text-accent-cyan" />
              <span>Supporting Video Frames (click to jump):</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {answerData.supporting_frames.map((frame) => (
                <div
                  key={frame.frame_id}
                  onClick={() => onSeek(frame.timestamp)}
                  className="group cursor-pointer rounded-lg border border-dark-600 hover:border-accent-cyan overflow-hidden bg-dark-800 transition-all hover:scale-[1.03]"
                >
                  <div className="relative aspect-video">
                    <img
                      src={frame.image_url}
                      alt={`Frame at ${frame.timestamp}s`}
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <span className="text-[10px] text-white font-mono bg-dark-900/90 px-1.5 py-0.5 rounded">
                        Seek to {frame.timestamp.toFixed(1)}s
                      </span>
                    </div>
                  </div>
                  <div className="p-1 text-center font-mono text-[10px] text-slate-300 bg-dark-900/90 flex items-center justify-between px-2">
                    <span className="text-accent-cyan">{frame.timestamp.toFixed(1)}s</span>
                    {frame.difference_score !== undefined && (
                      <span className="text-slate-500 text-[9px]">score: {frame.difference_score.toFixed(2)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Explanation */}
        {answerData.reasoning_summary && (
          <div className="p-3 rounded-lg bg-dark-800/40 border border-dark-700/60 text-xs space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Explanation:</span>
            <p className="text-slate-300 leading-relaxed">
              {answerData.reasoning_summary}
            </p>
          </div>
        )}
      </div>

      {/* DEVELOPER / DEBUG PANEL (Requirement 8) */}
      {showDebug && (
        <div className="pt-4 border-t-2 border-accent-violet/30 space-y-4 animate-in fade-in duration-200">
          <div className="flex items-center justify-between bg-accent-violet/10 px-3 py-2 rounded-lg border border-accent-violet/30">
            <div className="flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-accent-violet" />
              <span className="text-xs font-bold text-accent-violet uppercase tracking-wider font-mono">
                Developer Debug Mode & Retrieval Diagnostics
              </span>
            </div>
            <span className="text-[10px] text-slate-400 font-mono">
              Inspect Retrieval vs Generation
            </span>
          </div>

          {/* User Question & Classification */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg bg-dark-900 border border-dark-700 space-y-1">
              <span className="text-[10px] uppercase text-slate-400 font-semibold font-mono">User Question:</span>
              <p className="text-slate-200 font-mono text-[11px]">{answerData.question}</p>
            </div>
            <div className="p-2.5 rounded-lg bg-dark-900 border border-dark-700 space-y-1">
              <span className="text-[10px] uppercase text-slate-400 font-semibold font-mono">Detected Question Type:</span>
              <div className="flex items-center space-x-2">
                <span className="px-2 py-0.5 rounded text-[11px] font-bold font-mono bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40">
                  {answerData.debug_info?.detected_question_type || qType}
                </span>
                {answerData.debug_info?.modality_weights_used && (
                  <span className="text-[10px] text-slate-400 font-mono">
                    w: {JSON.stringify(answerData.debug_info.modality_weights_used)}
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Retrieved Evidence Breakdown */}
          <div className="space-y-2">
            <span className="text-[11px] font-semibold text-slate-300 font-mono uppercase">
              Retrieved Evidence Items ({answerData.evidence_used?.length || 0}):
            </span>
            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {(answerData.debug_info?.retrieved_evidence || answerData.evidence_used || []).map((item, idx) => (
                <div
                  key={idx}
                  onClick={() => onSeek(item.timestamp_start, item.timestamp_end)}
                  className="p-2.5 rounded-lg bg-dark-900 border border-dark-700 hover:border-slate-500 text-xs flex items-start justify-between cursor-pointer space-x-3 transition-colors"
                >
                  <div className="flex items-start space-x-2.5 min-w-0">
                    <span className="font-mono text-slate-500 text-[10px] pt-0.5">#{idx + 1}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-bold shrink-0 font-mono ${
                        item.type === 'transcript'
                          ? 'bg-brand-500/20 text-brand-400 border border-brand-500/30'
                          : item.type === 'ocr'
                          ? 'bg-accent-amber/20 text-accent-amber border border-accent-amber/30'
                          : 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                      }`}
                    >
                      {item.type}
                    </span>
                    <span className="font-mono text-[11px] text-accent-cyan shrink-0 pt-0.5">
                      {formatSec(item.timestamp_start)} – {formatSec(item.timestamp_end)}
                    </span>
                    <span className="text-slate-300 text-[11px] line-clamp-2">
                      {item.text}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2 shrink-0">
                    {item.frame_url && (
                      <img
                        src={item.frame_url}
                        alt="frame thumb"
                        className="w-12 h-7 object-cover rounded border border-dark-600"
                      />
                    )}
                    <div className="text-right font-mono">
                      <span className="text-[10px] text-slate-400 block">score</span>
                      <span className="text-xs font-bold text-accent-cyan">{item.relevance_score.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* FINAL LLM CONTEXT */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-accent-violet font-mono uppercase flex items-center space-x-1.5">
                <Code className="w-3.5 h-3.5" />
                <span>FINAL LLM CONTEXT (Sent to Model)</span>
              </span>
              <button
                onClick={() => copyDebugItem(answerData.debug_info?.final_llm_context || '', true)}
                className="text-[10px] font-mono text-slate-400 hover:text-slate-200 flex items-center space-x-1"
              >
                {copiedContext ? <Check className="w-3 h-3 text-brand-500" /> : <Copy className="w-3 h-3" />}
                <span>{copiedContext ? 'Copied' : 'Copy Context'}</span>
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#05070A] border border-dark-700 text-[10px] font-mono text-slate-300 max-h-48 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {answerData.debug_info?.final_llm_context || '(No context recorded)'}
            </pre>
          </div>

          {/* FINAL LLM RESPONSE */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-accent-cyan font-mono uppercase flex items-center space-x-1.5">
                <Code className="w-3.5 h-3.5" />
                <span>FINAL LLM RESPONSE (Raw Model Output)</span>
              </span>
              <button
                onClick={() => copyDebugItem(answerData.debug_info?.final_llm_response || '', false)}
                className="text-[10px] font-mono text-slate-400 hover:text-slate-200 flex items-center space-x-1"
              >
                {copiedResponse ? <Check className="w-3 h-3 text-brand-500" /> : <Copy className="w-3 h-3" />}
                <span>{copiedResponse ? 'Copied' : 'Copy Response'}</span>
              </button>
            </div>
            <pre className="p-3 rounded-lg bg-[#05070A] border border-dark-700 text-[10px] font-mono text-slate-300 max-h-36 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {answerData.debug_info?.final_llm_response || '(No raw response recorded)'}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
