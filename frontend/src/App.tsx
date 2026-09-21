import { useState, useEffect, useRef } from 'react';
import { VideoUploader } from './components/VideoUploader';
import { VideoPlayer } from './components/VideoPlayer';
import type { VideoPlayerRef } from './components/VideoPlayer';
import { TimelineViewer } from './components/TimelineViewer';
import { PipelineProgress } from './components/PipelineProgress';
import { QuestionInput } from './components/QuestionInput';
import { GroundedAnswer } from './components/GroundedAnswer';
import { BenchmarkModal } from './components/BenchmarkModal';
import { api } from './services/api';
import type { VideoMetadata, VideoStatus, FrameInfo, TranscriptSegment, GroundedAnswer as GroundedAnswerType } from './types';
import { Eye, BarChart2, Video, RefreshCw, Cpu, Layers } from 'lucide-react';

export function App() {
  const playerRef = useRef<VideoPlayerRef>(null);

  const [videoMeta, setVideoMeta] = useState<VideoMetadata | null>(null);
  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null);
  const [frames, setFrames] = useState<FrameInfo[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptSegment[]>([]);
  const [currentTime, setCurrentTime] = useState(0);

  const [isAsking, setIsAsking] = useState(false);
  const [groundedAnswer, setGroundedAnswer] = useState<GroundedAnswerType | null>(null);
  const [highlightRange, setHighlightRange] = useState<{ start: number; end: number } | null>(null);

  const [isBenchmarkOpen, setIsBenchmarkOpen] = useState(false);

  // Poll video status if processing
  useEffect(() => {
    if (!videoMeta) return;

    let interval: any = null;
    const checkStatus = async () => {
      try {
        const s = await api.getVideoStatus(videoMeta.video_id);
        setVideoStatus(s);

        if (s.status === 'completed') {
          clearInterval(interval);
          // Load frames and transcripts
          const [f, t] = await Promise.all([
            api.getVideoFrames(videoMeta.video_id),
            api.getVideoTranscript(videoMeta.video_id),
          ]);
          setFrames(f);
          setTranscripts(t);
        } else if (s.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error fetching video status:', err);
      }
    };

    checkStatus();
    interval = setInterval(checkStatus, 1500);

    return () => clearInterval(interval);
  }, [videoMeta]);

  const handleVideoSelected = async (meta: VideoMetadata) => {
    setVideoMeta(meta);
    setGroundedAnswer(null);
    setHighlightRange(null);
    setFrames([]);
    setTranscripts([]);

    // Automatically trigger multimodal understanding pipeline
    try {
      await api.processVideo(meta.video_id);
      setVideoStatus({
        video_id: meta.video_id,
        status: 'processing',
        stage: 'sampling',
        progress: 15,
        frames_count: 0,
        transcript_segments_count: 0,
        ocr_detections_count: 0,
      });
    } catch (err) {
      console.error('Failed to trigger pipeline:', err);
    }
  };

  const handleAsk = async (
    question: string,
    options?: { temporal_window?: number; top_k?: number; modality_weights?: Record<string, number> }
  ) => {
    if (!videoMeta) return;
    setIsAsking(true);
    try {
      const resp = await api.askQuestion(videoMeta.video_id, question, options);
      setGroundedAnswer(resp);

      // Set player highlight to the first grounded timestamp interval if available
      if (resp.timestamps && resp.timestamps.length > 0) {
        const primary = resp.timestamps[0];
        setHighlightRange({ start: primary.start, end: primary.end });
        playerRef.current?.seekTo(primary.start);
      }
    } catch (err: any) {
      alert(err.message || 'Inference failed');
    } finally {
      setIsAsking(false);
    }
  };

  const handleSeek = (timestamp: number, endTimestamp?: number) => {
    playerRef.current?.seekTo(timestamp);
    if (endTimestamp !== undefined) {
      setHighlightRange({ start: timestamp, end: endTimestamp });
    }
  };

  const isReady = videoStatus?.status === 'completed';

  return (
    <div className="min-h-screen bg-[#080B11] text-slate-100 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="border-b border-dark-700/80 bg-dark-900/80 backdrop-blur-md sticky top-0 z-40 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-accent-cyan via-brand-500 to-accent-violet flex items-center justify-center text-dark-900 shadow-md">
              <Eye className="w-5 h-5 stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-base font-bold tracking-tight text-white font-sans">
                  VideoLens
                </h1>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30">
                  Multimodal RAG &bull; Temporal QA
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Grounded Video Understanding with Cross-Modal Retrieval
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => setIsBenchmarkOpen(true)}
              className="px-3 py-1.5 rounded-lg bg-dark-800 hover:bg-dark-700 text-slate-200 border border-dark-600 hover:border-slate-500 text-xs font-medium flex items-center space-x-1.5 transition-colors shadow-sm"
            >
              <BarChart2 className="w-3.5 h-3.5 text-accent-cyan" />
              <span>Scientific Evaluation Suite</span>
            </button>

            {videoMeta && (
              <button
                onClick={() => handleVideoSelected(videoMeta)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-dark-800 transition-colors"
                title="Reprocess current video"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* LEFT COLUMN: Video Upload, Player, Timeline (7 Cols) */}
          <div className="lg:col-span-7 space-y-5">
            {!videoMeta ? (
              <div className="glass-panel rounded-2xl p-6 space-y-4 shadow-xl">
                <div className="flex items-center space-x-2 text-slate-200 font-semibold text-sm">
                  <Video className="w-4 h-4 text-accent-cyan" />
                  <span>Step 1: Upload Video for Multimodal Analysis</span>
                </div>
                <VideoUploader
                  onVideoSelected={handleVideoSelected}
                  isLoading={videoStatus?.status === 'processing'}
                />
              </div>
            ) : (
              <div className="space-y-4">
                {/* Video Player */}
                <VideoPlayer
                  ref={playerRef}
                  src={videoMeta.video_url}
                  duration={videoMeta.duration}
                  highlightRange={highlightRange}
                  onTimeUpdate={setCurrentTime}
                />

                {/* Video Info Strip */}
                <div className="flex items-center justify-between text-xs text-slate-400 bg-dark-800/60 px-3.5 py-2 rounded-xl border border-dark-700">
                  <div className="flex items-center space-x-2 font-mono text-[11px] truncate max-w-md">
                    <span className="text-slate-200 font-semibold truncate">{videoMeta.filename}</span>
                    <span>&bull;</span>
                    <span>{videoMeta.duration}s</span>
                    <span>&bull;</span>
                    <span>{videoMeta.width}x{videoMeta.height}</span>
                    <span>&bull;</span>
                    <span>{videoMeta.fps} fps</span>
                  </div>

                  <button
                    onClick={() => {
                      setVideoMeta(null);
                      setVideoStatus(null);
                      setFrames([]);
                      setTranscripts([]);
                      setGroundedAnswer(null);
                    }}
                    className="text-[11px] text-accent-cyan hover:underline shrink-0 font-medium"
                  >
                    Change Video
                  </button>
                </div>

                {/* Timeline Viewer */}
                <TimelineViewer
                  duration={videoMeta.duration}
                  currentTime={currentTime}
                  frames={frames}
                  transcripts={transcripts}
                  highlightTimestamps={groundedAnswer?.timestamps || []}
                  onSeek={handleSeek}
                />
              </div>
            )}
          </div>

          {/* RIGHT COLUMN: Grounded QA & Multimodal Evidence (5 Cols) */}
          <div className="lg:col-span-5 space-y-5">
            {/* Question Input Card */}
            <div className="glass-panel rounded-2xl p-5 space-y-4 shadow-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-slate-200 font-semibold text-sm">
                  <Cpu className="w-4 h-4 text-accent-cyan" />
                  <span>Step 2: Ask Grounded Questions</span>
                </div>
                {isReady && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-brand-500/20 text-brand-500 border border-brand-500/30">
                    Multimodal Index Ready
                  </span>
                )}
              </div>

              <QuestionInput
                onAsk={handleAsk}
                isLoading={isAsking}
                disabled={!isReady}
              />
            </div>

            {/* Grounded Answer Card */}
            {groundedAnswer ? (
              <GroundedAnswer
                answerData={groundedAnswer}
                onSeek={handleSeek}
              />
            ) : (
              <div className="glass-panel rounded-2xl p-8 text-center space-y-3 border-dashed border-dark-600">
                <div className="w-12 h-12 rounded-full bg-dark-800 flex items-center justify-center mx-auto text-slate-500">
                  <Layers className="w-6 h-6" />
                </div>
                <h3 className="text-sm font-semibold text-slate-300">
                  Grounded Evidence Will Appear Here
                </h3>
                <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                  When you submit a query, VideoLens retrieves candidate frames, speech segments, and OCR text, performs temporal expansion, and generates verifiable answers with clickable timestamps.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* BOTTOM PANEL: Video Understanding Pipeline Indicator */}
        <div className="pt-2">
          <PipelineProgress status={videoStatus} />
        </div>
      </main>

      {/* Evaluation & Benchmark Modal */}
      <BenchmarkModal
        isOpen={isBenchmarkOpen}
        onClose={() => setIsBenchmarkOpen(false)}
      />

      {/* Footer */}
      <footer className="border-t border-dark-700/60 py-4 px-6 text-center text-xs text-slate-500 font-mono">
        VideoLens &bull; Multimodal Video Understanding & Grounded QA &bull; OpenCLIP ViT-B/32 + Faster-Whisper + EasyOCR + FAISS
      </footer>
    </div>
  );
}

export default App;
