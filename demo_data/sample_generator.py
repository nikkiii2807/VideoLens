import os
import cv2
import numpy as np
import subprocess
from pathlib import Path

import shutil

# Resolve FFmpeg
winget_ffmpeg = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe"
FFMPEG_EXE = str(winget_ffmpeg) if winget_ffmpeg.exists() else (shutil.which("ffmpeg") or "ffmpeg")

DEMO_DIR = Path(__file__).resolve().parent
DEMO_DIR.mkdir(parents=True, exist_ok=True)

def generate_speech_audio(wav_path: Path):
    """Uses Windows SAPI to synthesize realistic spoken speech into a WAV file."""
    text = (
        "Welcome to the VideoLens multimodal lecture. "
        "First, we introduce Gaussian filtering and 2D convolution kernels. "
        "Next, we use FAISS vector search to index visual and temporal embeddings. "
        "Finally, the multimodal model grounds the question in the video timeline."
    )
    wav_str = str(wav_path).replace("\\", "/")
    ps_script = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SetOutputToWaveFile('{wav_str}')
    $synth.Speak('{text}')
    $synth.Dispose()
    """
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script]
    subprocess.run(cmd, check=True, capture_output=True)

def create_sample_video(output_mp4: Path = DEMO_DIR / "sample_lecture.mp4"):
    temp_avi = DEMO_DIR / "temp_video.avi"
    temp_wav = DEMO_DIR / "temp_audio.wav"
    
    # 1. Generate spoken audio
    print(f"Generating spoken audio at {temp_wav}...")
    try:
        generate_speech_audio(temp_wav)
        has_audio = temp_wav.exists() and temp_wav.stat().st_size > 1000
    except Exception as e:
        print(f"Warning: could not generate SAPI audio: {e}. Generating silent WAV.")
        has_audio = False

    # Get audio duration if available
    target_duration = 16.0
    fps = 25
    width, height = 640, 360
    total_frames = int(target_duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))

    print(f"Rendering {total_frames} video frames ({target_duration}s)...")
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Episode 1: 0.0 - 4.5s (Introduction)
        if t < 4.5:
            img[:] = (35, 25, 15) # Dark navy
            cv2.putText(img, "VideoLens Multimodal AI Lecture", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Speaker: Prof. Alan Vance", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            # Animated red circle moving across
            cx = int(80 + (t / 4.5) * 460)
            cy = 190
            cv2.circle(img, (cx, cy), 35, (50, 50, 240), -1)
            cv2.putText(img, "Red Object", (cx - 45, cy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Episode 2: 4.5 - 10.0s (Gaussian Filtering)
        elif t < 10.0:
            rel_t = t - 4.5
            img[:] = (20, 40, 20) # Dark forest green
            cv2.putText(img, "Gaussian Filtering & Image Kernels", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Concept: 2D Convolution & Blur", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            # Yellow square rotating/scaling
            sx = int(120 + (rel_t / 5.5) * 380)
            sy = 190
            cv2.rectangle(img, (sx - 30, sy - 30), (sx + 30, sy + 30), (0, 230, 255), -1)
            cv2.putText(img, "Kernel Filter", (sx - 45, sy + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Episode 3: 10.0 - 16.0s (FAISS & Temporal Indexing)
        else:
            rel_t = t - 10.0
            img[:] = (40, 20, 35) # Dark purple
            cv2.putText(img, "FAISS Vector Search & Temporal Grounding", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "Multimodal RAG with Chronological Context", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 255), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}", (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            # Cyan polygon
            tx = int(100 + (rel_t / 6.0) * 400)
            ty = 190
            pts = np.array([[tx, ty - 35], [tx - 35, ty + 30], [tx + 35, ty + 30]], np.int32)
            cv2.fillPoly(img, [pts], (255, 220, 0))
            cv2.putText(img, "Vector Node", (tx - 45, ty + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(img)
        
    out.release()
    print("Video track rendered.")

    # 3. Mux video and audio with FFmpeg into clean H.264 / AAC MP4
    print(f"Muxing into {output_mp4} with FFmpeg...")
    if has_audio:
        cmd = [
            FFMPEG_EXE, "-y",
            "-i", str(temp_avi),
            "-i", str(temp_wav),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(output_mp4)
        ]
    else:
        cmd = [
            FFMPEG_EXE, "-y",
            "-i", str(temp_avi),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_mp4)
        ]
        
    subprocess.run(cmd, check=True, capture_output=True)
    
    # Clean up temp files
    temp_avi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)
    print(f"Sample lecture video generated successfully: {output_mp4} ({output_mp4.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_sample_video()
