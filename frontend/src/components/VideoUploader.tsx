import React, { useRef, useState } from 'react';
import { UploadCloud, Film, PlayCircle, AlertCircle, Sparkles } from 'lucide-react';
import { api } from '../services/api';
import type { VideoMetadata } from '../types';

interface VideoUploaderProps {
  onVideoSelected: (meta: VideoMetadata) => void;
  isLoading: boolean;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({ onVideoSelected, isLoading }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const handleFile = async (file: File) => {
    setError(null);
    if (!file) return;

    if (file.size > 50 * 1024 * 1024) {
      setError('File exceeds 50MB maximum size limit.');
      return;
    }

    try {
      setUploading(true);
      const meta = await api.uploadVideo(file);
      onVideoSelected(meta);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDemoVideo = async (demoName: string = 'lecture') => {
    setError(null);
    try {
      setUploading(true);
      const meta = await api.loadDemoVideo(demoName);
      onVideoSelected(meta);
    } catch (err: any) {
      setError(err.message || 'Failed to load demo video');
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
          dragActive
            ? 'border-accent-cyan bg-accent-cyan/10 glow-accent'
            : 'border-dark-600 hover:border-slate-400 bg-dark-800/60'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/x-msvideo"
          className="hidden"
          onChange={(e) => e.target.files && handleFile(e.target.files[0])}
          disabled={uploading || isLoading}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-dark-700 flex items-center justify-center text-accent-cyan shadow-inner">
            <UploadCloud className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-200">
              Drag & drop video here, or <span className="text-accent-cyan hover:underline">browse</span>
            </p>
            <p className="text-xs text-slate-400 mt-1">
              MP4, WebM, MOV, AVI (Max duration: 60s &bull; Max size: 50MB)
            </p>
          </div>
        </div>

        {uploading && (
          <div className="absolute inset-0 bg-dark-900/80 backdrop-blur-sm rounded-xl flex items-center justify-center">
            <div className="flex items-center space-x-2 text-accent-cyan text-sm">
              <Film className="w-4 h-4 animate-spin" />
              <span>Uploading & probing video...</span>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3 p-3 bg-accent-rose/10 border border-accent-rose/30 rounded-lg flex items-center space-x-2 text-accent-rose text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Demo Video Quick Load */}
      <div className="mt-3.5 p-3 rounded-xl bg-dark-800/50 border border-dark-700/80 space-y-2">
        <div className="flex items-center space-x-1.5 text-xs text-slate-300 font-medium">
          <Sparkles className="w-3.5 h-3.5 text-accent-amber" />
          <span>Load Pre-rendered Thematic Test Videos:</span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDemoVideo('lecture');
            }}
            disabled={uploading || isLoading}
            className="flex items-center justify-center space-x-1 px-2.5 py-1.5 rounded-lg bg-dark-700/70 hover:bg-brand-500/20 text-slate-200 hover:text-brand-500 text-xs font-medium transition-all border border-dark-600 hover:border-brand-500/40"
          >
            <PlayCircle className="w-3.5 h-3.5 text-accent-cyan shrink-0" />
            <span className="truncate">AI Lecture (16s)</span>
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDemoVideo('cooking');
            }}
            disabled={uploading || isLoading}
            className="flex items-center justify-center space-x-1 px-2.5 py-1.5 rounded-lg bg-dark-700/70 hover:bg-accent-amber/20 text-slate-200 hover:text-accent-amber text-xs font-medium transition-all border border-dark-600 hover:border-accent-amber/40"
          >
            <PlayCircle className="w-3.5 h-3.5 text-accent-amber shrink-0" />
            <span className="truncate">Pasta Recipe (18s)</span>
          </button>

          <button
            onClick={(e) => {
              e.stopPropagation();
              handleDemoVideo('robotics');
            }}
            disabled={uploading || isLoading}
            className="flex items-center justify-center space-x-1 px-2.5 py-1.5 rounded-lg bg-dark-700/70 hover:bg-accent-violet/20 text-slate-200 hover:text-accent-violet text-xs font-medium transition-all border border-dark-600 hover:border-accent-violet/40"
          >
            <PlayCircle className="w-3.5 h-3.5 text-accent-violet shrink-0" />
            <span className="truncate">Robotics Lab (18s)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
