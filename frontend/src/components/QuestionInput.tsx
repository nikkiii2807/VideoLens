import React, { useState } from 'react';
import { Send, SlidersHorizontal, Sparkles, Loader2 } from 'lucide-react';

interface QuestionInputProps {
  onAsk: (question: string, options?: { temporal_window?: number; top_k?: number; modality_weights?: Record<string, number> }) => void;
  isLoading: boolean;
  disabled: boolean;
}

const SAMPLE_QUERIES = [
  "What happens after the red object appears?",
  "What ingredients are added in step 2 of the recipe?",
  "What does the rover do when it detects an obstacle?",
  "Explain Gaussian filtering and image convolution.",
  "Which waypoint does the robot reach at the end?",
  "When is the golden olive oil drizzled?",
];

export const QuestionInput: React.FC<QuestionInputProps> = ({ onAsk, isLoading, disabled }) => {
  const [question, setQuestion] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  // Tunable hyperparameters
  const [temporalWindow, setTemporalWindow] = useState(4.0);
  const [topK, setTopK] = useState(5);
  const [textWeight, setTextWeight] = useState(0.35);
  const [visWeight, setVisWeight] = useState(0.35);
  const [ocrWeight, setOcrWeight] = useState(0.20);
  const [tempWeight, setTempWeight] = useState(0.10);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading || disabled) return;

    onAsk(question.trim(), {
      temporal_window: temporalWindow,
      top_k: topK,
      modality_weights: {
        text: textWeight,
        visual: visWeight,
        ocr: ocrWeight,
        temporal: tempWeight,
      },
    });
  };

  return (
    <div className="w-full space-y-3">
      <form onSubmit={handleSubmit} className="relative">
        <div className="flex items-center space-x-2 bg-dark-800/80 rounded-xl border border-dark-600 focus-within:border-accent-cyan transition-all p-1.5 shadow-lg">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              disabled
                ? "Upload or load a video first to ask questions..."
                : "Ask anything about the video (e.g., 'What happens after...')..."
            }
            disabled={disabled || isLoading}
            className="flex-1 bg-transparent px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none"
          />

          <button
            type="button"
            onClick={() => setShowSettings(!showSettings)}
            className={`p-2 rounded-lg transition-colors ${
              showSettings
                ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40'
                : 'text-slate-400 hover:text-slate-200 hover:bg-dark-700'
            }`}
            title="Configure multimodal retrieval hyperparameters"
          >
            <SlidersHorizontal className="w-4 h-4" />
          </button>

          <button
            type="submit"
            disabled={!question.trim() || disabled || isLoading}
            className="px-4 py-2 rounded-lg bg-accent-cyan hover:bg-cyan-500 text-dark-900 font-semibold text-xs flex items-center space-x-1.5 transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-md hover:shadow-cyan-500/20"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Grounding...</span>
              </>
            ) : (
              <>
                <span>Ask Video</span>
                <Send className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </form>

      {/* Suggested Questions */}
      {!disabled && (
        <div className="space-y-1.5">
          <div className="flex items-center space-x-1 text-[11px] text-slate-400 font-medium">
            <Sparkles className="w-3 h-3 text-accent-amber" />
            <span>Try an example question:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {SAMPLE_QUERIES.map((q, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQuestion(q)}
                disabled={isLoading}
                className="text-[11px] px-2.5 py-1 rounded-full bg-dark-800 hover:bg-dark-700 text-slate-300 border border-dark-700 hover:border-slate-500 transition-colors text-left"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Settings Drawer */}
      {showSettings && (
        <div className="glass-panel rounded-xl p-4 space-y-3 text-xs border-accent-cyan/30 animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-dark-700 pb-2">
            <span className="font-semibold text-slate-200 uppercase tracking-wider text-[11px]">
              Retrieval & Temporal Hyperparameters
            </span>
            <span className="text-[10px] text-slate-400">Experimental tuning</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Temporal Window (&plusmn;sec):</span>
                <span className="font-mono text-accent-cyan">{temporalWindow}s</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="10.0"
                step="0.5"
                value={temporalWindow}
                onChange={(e) => setTemporalWindow(parseFloat(e.target.value))}
                className="w-full accent-accent-cyan"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Top-K Chunks:</span>
                <span className="font-mono text-accent-cyan">{topK}</span>
              </div>
              <input
                type="range"
                min="2"
                max="12"
                step="1"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                className="w-full accent-accent-cyan"
              />
            </div>
          </div>

          {/* Modality Weights */}
          <div className="pt-2 border-t border-dark-700 space-y-2">
            <span className="text-[11px] font-semibold text-slate-400">Multimodal Scoring Weights</span>
            <div className="grid grid-cols-4 gap-2 text-[10px]">
              <div>
                <span className="text-slate-400">w_text: {textWeight.toFixed(2)}</span>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={textWeight}
                  onChange={(e) => setTextWeight(parseFloat(e.target.value))}
                  className="w-full accent-brand-500"
                />
              </div>
              <div>
                <span className="text-slate-400">w_vis: {visWeight.toFixed(2)}</span>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={visWeight}
                  onChange={(e) => setVisWeight(parseFloat(e.target.value))}
                  className="w-full accent-accent-cyan"
                />
              </div>
              <div>
                <span className="text-slate-400">w_ocr: {ocrWeight.toFixed(2)}</span>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={ocrWeight}
                  onChange={(e) => setOcrWeight(parseFloat(e.target.value))}
                  className="w-full accent-accent-amber"
                />
              </div>
              <div>
                <span className="text-slate-400">w_temp: {tempWeight.toFixed(2)}</span>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={tempWeight}
                  onChange={(e) => setTempWeight(parseFloat(e.target.value))}
                  className="w-full accent-accent-violet"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
