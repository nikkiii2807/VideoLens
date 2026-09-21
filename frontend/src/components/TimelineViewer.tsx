import type { FrameInfo, TranscriptSegment, GroundedTimestamp } from '../types';

interface TimelineViewerProps {
  duration: number;
  currentTime: number;
  frames: FrameInfo[];
  transcripts: TranscriptSegment[];
  highlightTimestamps: GroundedTimestamp[];
  onSeek: (timestamp: number) => void;
}

export const TimelineViewer: React.FC<TimelineViewerProps> = ({
  duration,
  currentTime,
  frames,
  transcripts,
  highlightTimestamps,
  onSeek,
}) => {
  if (duration <= 0) return null;

  return (
    <div className="w-full glass-panel rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold text-slate-300 uppercase tracking-wider text-[11px]">
          Multimodal Timeline Breakdown
        </span>
        <div className="flex items-center space-x-3 text-[11px]">
          <span className="flex items-center space-x-1 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-brand-500" />
            <span>Speech</span>
          </span>
          <span className="flex items-center space-x-1 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-accent-amber" />
            <span>OCR</span>
          </span>
          <span className="flex items-center space-x-1 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-accent-cyan" />
            <span>Grounded Span</span>
          </span>
        </div>
      </div>

      {/* Main Timeline Bar */}
      <div
        className="relative h-10 w-full bg-dark-900 rounded-lg overflow-hidden border border-dark-700 cursor-pointer group"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const clickX = e.clientX - rect.left;
          const ratio = Math.max(0, Math.min(1, clickX / rect.width));
          onSeek(ratio * duration);
        }}
      >
        {/* Playhead indicator */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white z-30 shadow-[0_0_8px_white] pointer-events-none transition-all duration-75"
          style={{ left: `${(currentTime / duration) * 100}%` }}
        />

        {/* Speech Transcript Segments (Green bars) */}
        {transcripts.map((t) => {
          const leftPct = (t.start_time / duration) * 100;
          const widthPct = ((t.end_time - t.start_time) / duration) * 100;
          return (
            <div
              key={t.chunk_id}
              className="absolute top-1 bottom-5 bg-brand-500/30 border border-brand-500/50 rounded-sm hover:bg-brand-500/60 transition-colors z-10"
              style={{ left: `${leftPct}%`, width: `${Math.max(1, widthPct)}%` }}
              title={`Speech: "${t.text}" (${t.start_time}s - ${t.end_time}s)`}
            />
          );
        })}

        {/* OCR Event Dots (Amber dots) */}
        {frames
          .filter((f) => f.has_ocr)
          .map((f) => {
            const leftPct = (f.timestamp / duration) * 100;
            return (
              <div
                key={`ocr-dot-${f.frame_id}`}
                className="absolute bottom-1 w-1.5 h-1.5 rounded-full bg-accent-amber z-20"
                style={{ left: `${leftPct}%`, transform: 'translateX(-50%)' }}
                title={`Text detected @ ${f.timestamp}s`}
              />
            );
          })}

        {/* Highlighted Grounded QA Regions (Cyan overlay) */}
        {highlightTimestamps.map((ts, idx) => {
          const leftPct = (ts.start / duration) * 100;
          const widthPct = ((ts.end - ts.start) / duration) * 100;
          return (
            <div
              key={`grounded-span-${idx}`}
              className="absolute inset-y-0 bg-accent-cyan/30 border-x-2 border-accent-cyan pointer-events-none z-20 animate-pulse"
              style={{ left: `${leftPct}%`, width: `${Math.max(2, widthPct)}%` }}
            />
          );
        })}
      </div>

      {/* Frame Thumbnail Ribbon */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1 pt-0.5">
        {frames.map((f) => (
          <div
            key={f.frame_id}
            onClick={() => onSeek(f.timestamp)}
            className="flex-shrink-0 group/frame cursor-pointer relative rounded border border-dark-600 hover:border-accent-cyan overflow-hidden transition-all duration-150 hover:scale-105"
          >
            <img
              src={f.image_url}
              alt={`Frame at ${f.timestamp}s`}
              className="w-14 h-9 object-cover"
              loading="lazy"
            />
            <span className="absolute bottom-0 inset-x-0 bg-dark-900/80 text-[9px] font-mono text-center text-slate-300 py-0.5">
              {f.timestamp.toFixed(1)}s
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
