import React, { forwardRef, useImperativeHandle, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Maximize } from 'lucide-react';

export interface VideoPlayerRef {
  seekTo: (time: number) => void;
  play: () => void;
  pause: () => void;
  getCurrentTime: () => number;
}

interface VideoPlayerProps {
  src: string;
  duration: number;
  highlightRange?: { start: number; end: number } | null;
  onTimeUpdate?: (time: number) => void;
}

export const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(
  ({ src, duration, highlightRange, onTimeUpdate }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [isMuted, setIsMuted] = useState(false);

    useImperativeHandle(ref, () => ({
      seekTo: (time: number) => {
        if (videoRef.current) {
          videoRef.current.currentTime = Math.max(0, Math.min(time, duration));
          videoRef.current.play().catch(() => {});
          setIsPlaying(true);
        }
      },
      play: () => videoRef.current?.play(),
      pause: () => videoRef.current?.pause(),
      getCurrentTime: () => videoRef.current?.currentTime || 0,
    }));

    const togglePlay = () => {
      if (videoRef.current) {
        if (isPlaying) {
          videoRef.current.pause();
        } else {
          videoRef.current.play();
        }
        setIsPlaying(!isPlaying);
      }
    };

    const handleTimeUpdate = () => {
      if (videoRef.current) {
        const t = videoRef.current.currentTime;
        setCurrentTime(t);
        if (onTimeUpdate) {
          onTimeUpdate(t);
        }
      }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
      const targetTime = parseFloat(e.target.value);
      if (videoRef.current) {
        videoRef.current.currentTime = targetTime;
        setCurrentTime(targetTime);
      }
    };

    const toggleMute = () => {
      if (videoRef.current) {
        videoRef.current.muted = !isMuted;
        setIsMuted(!isMuted);
      }
    };

    const toggleFullscreen = () => {
      if (videoRef.current) {
        if (!document.fullscreenElement) {
          videoRef.current.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      }
    };

    const formatTime = (seconds: number) => {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      const tenths = Math.floor((seconds % 1) * 10);
      return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${tenths}`;
    };

    return (
      <div className="relative rounded-xl overflow-hidden bg-black border border-dark-700 shadow-2xl group">
        <video
          ref={videoRef}
          src={src}
          className="w-full aspect-video object-contain"
          onTimeUpdate={handleTimeUpdate}
          onEnded={() => setIsPlaying(false)}
          onClick={togglePlay}
        />

        {/* Video Scrubber Overlay */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-dark-900/95 via-dark-900/70 to-transparent p-3 pt-6 flex flex-col space-y-2 opacity-95 group-hover:opacity-100 transition-opacity">
          {/* Seek Bar with Highlight Range */}
          <div className="relative w-full flex items-center">
            {/* Highlighted Evidence Range Indicator */}
            {highlightRange && duration > 0 && (
              <div
                className="absolute top-1/2 -translate-y-1/2 h-2 rounded bg-brand-500/50 pointer-events-none z-10 glow-brand"
                style={{
                  left: `${(highlightRange.start / duration) * 100}%`,
                  width: `${((highlightRange.end - highlightRange.start) / duration) * 100}%`,
                }}
              />
            )}
            <input
              type="range"
              min="0"
              max={duration || 1}
              step="0.05"
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-1.5 bg-dark-600 rounded-lg appearance-none cursor-pointer accent-accent-cyan z-20"
            />
          </div>

          {/* Controls Bar */}
          <div className="flex items-center justify-between text-xs text-slate-300">
            <div className="flex items-center space-x-3">
              <button
                onClick={togglePlay}
                className="p-1.5 rounded-lg bg-dark-700 hover:bg-dark-600 text-white transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              </button>

              <button
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.currentTime = 0;
                    videoRef.current.play();
                    setIsPlaying(true);
                  }
                }}
                className="p-1.5 rounded-lg bg-dark-700 hover:bg-dark-600 text-slate-300 transition-colors"
                title="Restart"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={toggleMute}
                className="p-1.5 rounded-lg bg-dark-700 hover:bg-dark-600 text-slate-300 transition-colors"
              >
                {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
              </button>

              <div className="font-mono text-xs text-slate-300">
                <span className="text-accent-cyan font-medium">{formatTime(currentTime)}</span>
                <span className="text-slate-500"> / {formatTime(duration)}</span>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {highlightRange && (
                <div className="px-2 py-0.5 rounded bg-brand-500/20 text-brand-500 border border-brand-500/30 text-[11px] font-mono">
                  Grounded: {formatTime(highlightRange.start)} - {formatTime(highlightRange.end)}
                </div>
              )}
              <button
                onClick={toggleFullscreen}
                className="p-1.5 rounded-lg bg-dark-700 hover:bg-dark-600 text-slate-300 transition-colors"
              >
                <Maximize className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }
);
