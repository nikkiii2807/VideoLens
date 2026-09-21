import os
import cv2
import numpy as np
import subprocess
from pathlib import Path
import shutil

DEMO_DIR = Path(__file__).resolve().parent
DEMO_DIR.mkdir(parents=True, exist_ok=True)

winget_ffmpeg = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-essentials_build/bin/ffmpeg.exe"
FFMPEG_EXE = str(winget_ffmpeg) if winget_ffmpeg.exists() else (shutil.which("ffmpeg") or "ffmpeg")

def synthesize_speech(text: str, wav_path: Path):
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

def generate_60s_video(output_mp4: Path = DEMO_DIR / "sample_60s.mp4"):
    temp_avi = DEMO_DIR / "temp_60s.avi"
    temp_wav = DEMO_DIR / "temp_60s.wav"

    # Narration for the 4 episodes across 60 seconds
    speech_text = (
        "Welcome to the VideoLens one minute multimodal demonstration. "
        "In section one, bring water to a rolling boil, then add fresh organic basil and ripe red tomatoes. "
        "Drizzle cold pressed golden olive oil to finish the dish. "
        "In section two, our autonomous robotics lab begins. The rover initializes its 3D LiDAR sensor array. "
        "The robot detects an obstacle at two meters and performs a rapid left turn maneuver. "
        "The rover reaches waypoint Bravo successfully. "
        "In section three, we study Gaussian filtering and 2D convolution kernels for image blur. "
        "Finally in section four, we use FAISS vector search to index visual and temporal embeddings and ground questions in the video timeline."
    )

    print(f"Synthesizing 60-second audio track to {temp_wav}...")
    try:
        synthesize_speech(speech_text, temp_wav)
        has_audio = temp_wav.exists() and temp_wav.stat().st_size > 1000
    except Exception as e:
        print(f"Warning: could not synthesize audio: {e}")
        has_audio = False

    fps = 25
    duration = 60.0
    total_frames = int(fps * duration)
    width, height = 640, 360

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))

    print(f"Rendering {total_frames} frames (60.0s) for 1-minute multimodal demo video...")
    for f_idx in range(total_frames):
        t = f_idx / fps
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # ----------------------------------------------------
        # EPISODE 1: 00:00 - 00:15 (Cooking: Boiling -> Tomatoes -> Olive Oil)
        # ----------------------------------------------------
        if t < 15.0:
            if t < 5.0:
                img[:] = (45, 30, 20) # Bronze
                cv2.putText(img, "Quick Kitchen Guide: Step 1", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Bring Water to Rolling Boil", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2)
                cv2.putText(img, "Temperature: 100C Boiling Pot", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                # Pot graphic
                bx, by = 320, 220
                cv2.rectangle(img, (bx - 60, by - 35), (bx + 60, by + 35), (180, 180, 190), -1)
                cv2.circle(img, (bx - 20, by - 10), 6, (255, 255, 200), -1)
                cv2.circle(img, (bx + 20, by - 5), 8, (255, 255, 200), -1)
                cv2.putText(img, "Boiling Pot", (bx - 45, by + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif t < 10.0:
                img[:] = (25, 45, 25) # Herb green
                cv2.putText(img, "Quick Kitchen Guide: Step 2", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Fresh Organic Basil & Ripe Tomatoes", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 150), 2)
                cv2.putText(img, "Ingredients: Red Tomatoes & Sweet Basil", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                # Red tomatoes
                tx = 240
                cv2.circle(img, (tx, 220), 32, (40, 40, 230), -1)
                cv2.circle(img, (tx + 65, 225), 26, (45, 45, 235), -1)
                cv2.putText(img, "Ripe Tomatoes", (tx - 40, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Green basil leaf
                lx = 420
                leaf_pts = np.array([[lx, 190], [lx + 30, 220], [lx, 250], [lx - 30, 220]], np.int32)
                cv2.fillPoly(img, [leaf_pts], (40, 200, 80))
                cv2.putText(img, "Fresh Basil", (lx - 35, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            else:
                img[:] = (20, 30, 50) # Terracotta
                cv2.putText(img, "Quick Kitchen Guide: Step 3", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Cold-Pressed Golden Olive Oil Drizzle", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
                cv2.putText(img, "Finish: Drizzle on Ceramic Plate", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.ellipse(img, (320, 225), (110, 40), 0, 0, 360, (210, 210, 220), -1)
                cv2.circle(img, (320, 180), 8, (0, 215, 255), -1)
                cv2.putText(img, "Golden Olive Oil", (260, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)

        # ----------------------------------------------------
        # EPISODE 2: 00:15 - 00:30 (Robotics: LiDAR -> Left Turn -> Waypoint Bravo)
        # ----------------------------------------------------
        elif t < 30.0:
            rel_t = t - 15.0
            if rel_t < 5.0:
                img[:] = (25, 20, 15) # Cyber navy
                cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Phase 1: 3D LiDAR Sensor Array Active", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
                cv2.putText(img, "Telemetry: Mission Start at 15s", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
                rx, ry = 320, 220
                cv2.rectangle(img, (rx - 45, ry - 25), (rx + 45, ry + 25), (120, 110, 100), -1)
                cv2.circle(img, (rx, ry), 12, (255, 180, 0), -1)
                cv2.putText(img, "LiDAR Sensor Array", (rx - 70, ry + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif rel_t < 11.0:
                img[:] = (20, 20, 45) # Warning crimson
                cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Phase 2: Obstacle Alert at 2.0m Distance", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 255), 2)
                cv2.putText(img, "Navigation: Rapid Left Turn Maneuver", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                # Obstacle
                cv2.rectangle(img, (390, 180), (440, 250), (30, 30, 220), -1)
                cv2.putText(img, "HAZARD 2.0m", (370, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 2)
                # Rover turning left
                rx, ry = 250, 220
                cv2.rectangle(img, (rx - 35, ry - 20), (rx + 35, ry + 20), (120, 110, 100), -1)
                cv2.arrowedLine(img, (rx, ry), (rx - 35, ry - 25), (0, 255, 255), 3)
                cv2.putText(img, "Left Turn", (rx - 35, ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            else:
                img[:] = (20, 45, 20) # Success green
                cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.putText(img, "Phase 3: Waypoint Bravo Reached", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 120), 2)
                cv2.putText(img, "Mission Outcome: Target Secured at 26s", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                wx, wy = 320, 210
                cv2.circle(img, (wx, wy), 20, (0, 255, 180), -1)
                cv2.putText(img, "WAYPOINT BRAVO", (wx - 70, wy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 150), 2)

        # ----------------------------------------------------
        # EPISODE 3: 00:30 - 00:45 (Lecture: Gaussian Filtering & 2D Convolution)
        # ----------------------------------------------------
        elif t < 45.0:
            rel_t = t - 30.0
            img[:] = (20, 40, 20) # Forest green
            cv2.putText(img, "Computer Vision Lecture: Image Blur", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, "Gaussian Filtering & 2D Convolution", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 150), 2)
            cv2.putText(img, "Mathematical Kernel: Blur & Noise Reduction", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            # Yellow square rotating/moving
            sx = int(140 + (rel_t / 15.0) * 360)
            sy = 210
            cv2.rectangle(img, (sx - 30, sy - 30), (sx + 30, sy + 30), (0, 230, 255), -1)
            cv2.putText(img, "Yellow Square Kernel", (sx - 65, sy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ----------------------------------------------------
        # EPISODE 4: 00:45 - 01:00 (Lecture: FAISS Vector Search & Temporal Grounding)
        # ----------------------------------------------------
        else:
            rel_t = t - 45.0
            img[:] = (40, 20, 35) # Dark purple
            cv2.putText(img, "Multimodal RAG: Temporal Grounding", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, "FAISS Vector Search & Chronological Context", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 255), 2)
            cv2.putText(img, "Visual Embeddings & Transcript Alignment", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            # Cyan polygon
            tx = int(120 + (rel_t / 15.0) * 380)
            ty = 210
            pts = np.array([[tx, ty - 35], [tx - 35, ty + 30], [tx + 35, ty + 30]], np.int32)
            cv2.fillPoly(img, [pts], (255, 220, 0))
            cv2.putText(img, "Vector Node", (tx - 45, ty + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw running time counter
        mins = int(t // 60)
        secs = int(t % 60)
        cv2.putText(img, f"Timestamp: {mins:02d}:{secs:02d}s", (35, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        out.write(img)

    out.release()
    print("Video rendered, now muxing with FFmpeg...")

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
    temp_avi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)
    print(f"Sample 60-second video generated: {output_mp4} ({output_mp4.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    generate_60s_video()
