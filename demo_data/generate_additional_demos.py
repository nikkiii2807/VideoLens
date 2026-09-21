import os
import cv2
import numpy as np
import subprocess
from pathlib import Path
import shutil

DEMO_DIR = Path(__file__).resolve().parent
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Resolve FFmpeg
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

def generate_cooking_demo():
    output_mp4 = DEMO_DIR / "sample_cooking.mp4"
    temp_avi = DEMO_DIR / "temp_cooking.avi"
    temp_wav = DEMO_DIR / "temp_cooking.wav"

    speech_text = (
        "Welcome to quick kitchen. "
        "First, bring a pot of salted water to a rolling boil. "
        "Next, add the fresh organic basil leaves and stir in the ripe red tomatoes. "
        "Finally, drizzle golden olive oil on top and serve hot on a ceramic plate."
    )

    print(f"Generating audio for Cooking Demo: {output_mp4}...")
    synthesize_speech(speech_text, temp_wav)

    fps = 25
    duration = 18.0
    total_frames = int(fps * duration)
    width, height = 640, 360

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))

    print(f"Rendering {total_frames} frames for Cooking Demo...")
    for f_idx in range(total_frames):
        t = f_idx / fps
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Episode 1: 0.0 - 5.5s (Boiling Water)
        if t < 5.5:
            img[:] = (45, 30, 20) # Deep warm bronze
            cv2.putText(img, "Quick Kitchen: Italian Pasta Guide", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Step 1: Bring Water to Rolling Boil", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2)
            cv2.putText(img, "Temperature: 100C Boiling Point", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Draw boiling pot graphic
            bx = 320
            by = 220
            cv2.rectangle(img, (bx - 70, by - 40), (bx + 70, by + 40), (180, 180, 190), -1)
            cv2.rectangle(img, (bx - 85, by - 35), (bx - 70, by - 15), (140, 140, 150), -1)
            cv2.rectangle(img, (bx + 70, by - 35), (bx + 85, by - 15), (140, 140, 150), -1)
            # Bubbles
            bubble_y = int(by - 20 - ((t * 40) % 40))
            cv2.circle(img, (bx - 25, bubble_y), 7, (255, 255, 200), -1)
            cv2.circle(img, (bx + 20, bubble_y + 8), 9, (255, 255, 200), -1)

        # Episode 2: 5.5 - 11.5s (Basil and Tomatoes)
        elif t < 11.5:
            rel_t = t - 5.5
            img[:] = (25, 45, 25) # Herb green
            cv2.putText(img, "Quick Kitchen: Italian Pasta Guide", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Step 2: Fresh Organic Basil & Tomatoes", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 255, 150), 2)
            cv2.putText(img, "Ingredients: Sweet Basil, San Marzano Tomatoes", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Red Tomato circles
            tx = int(220 + (rel_t / 6.0) * 80)
            cv2.circle(img, (tx, 220), 32, (40, 40, 230), -1)
            cv2.circle(img, (tx + 75, 225), 28, (45, 45, 235), -1)
            cv2.putText(img, "Ripe Tomatoes", (tx - 50, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Green Basil Leaf (Polygon)
            lx = tx + 170
            leaf_pts = np.array([[lx, 190], [lx + 35, 220], [lx, 250], [lx - 35, 220]], np.int32)
            cv2.fillPoly(img, [leaf_pts], (40, 200, 80))
            cv2.putText(img, "Fresh Basil", (lx - 40, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Episode 3: 11.5 - 18.0s (Olive Oil Drizzle)
        else:
            rel_t = t - 11.5
            img[:] = (20, 30, 50) # Warm terracotta / dark gold
            cv2.putText(img, "Quick Kitchen: Italian Pasta Guide", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Step 3: Cold-Pressed Olive Oil Drizzle", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
            cv2.putText(img, "Garnish & Finish: Serve immediately hot", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Golden ceramic plate
            cv2.ellipse(img, (320, 225), (120, 45), 0, 0, 360, (210, 210, 220), -1)
            cv2.ellipse(img, (320, 225), (95, 30), 0, 0, 360, (50, 140, 200), -1)
            
            # Olive oil golden stream
            drop_y = int(160 + ((rel_t * 60) % 55))
            cv2.circle(img, (320, drop_y), 8, (0, 215, 255), -1)
            cv2.putText(img, "Golden Olive Oil", (260, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1)

        out.write(img)

    out.release()

    # Mux with FFmpeg
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
    subprocess.run(cmd, check=True, capture_output=True)
    temp_avi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)
    print(f"Cooking demo generated: {output_mp4}")


def generate_robotics_demo():
    output_mp4 = DEMO_DIR / "sample_robotics.mp4"
    temp_avi = DEMO_DIR / "temp_robotics.avi"
    temp_wav = DEMO_DIR / "temp_robotics.wav"

    speech_text = (
        "In our robotics laboratory, the rover initializes its 3D LiDAR sensor array. "
        "The robot detects an obstacle at two meters distance and performs a rapid left turn. "
        "After avoiding the hazard, the rover reaches target waypoint Bravo successfully."
    )

    print(f"Generating audio for Robotics Demo: {output_mp4}...")
    synthesize_speech(speech_text, temp_wav)

    fps = 25
    duration = 18.0
    total_frames = int(fps * duration)
    width, height = 640, 360

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))

    print(f"Rendering {total_frames} frames for Robotics Demo...")
    for f_idx in range(total_frames):
        t = f_idx / fps
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Episode 1: 0.0 - 5.5s (LiDAR Sensor Array Initialization)
        if t < 5.5:
            img[:] = (25, 20, 15) # Cyber navy
            cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, "Phase 1: 3D LiDAR Sensor Initialization", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
            cv2.putText(img, "Telemetry: Sensor Range 360deg Active", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Rover chassis & rotating radar
            rx, ry = 320, 220
            cv2.rectangle(img, (rx - 50, ry - 30), (rx + 50, ry + 30), (120, 110, 100), -1)
            cv2.circle(img, (rx, ry), 15, (255, 180, 0), -1)
            # Radar rings
            radius = int(35 + ((t * 45) % 65))
            cv2.circle(img, (rx, ry), radius, (255, 220, 50), 2)
            cv2.putText(img, "LiDAR Scanning", (rx - 55, ry + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Episode 2: 5.5 - 11.5s (Hazard Detected & Rapid Left Turn)
        elif t < 11.5:
            rel_t = t - 5.5
            img[:] = (20, 20, 45) # Warning crimson/dark red
            cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, "Phase 2: Obstacle Alert at 2.0m Distance", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 255), 2)
            cv2.putText(img, "Navigation: Performing Rapid Left Turn Maneuver", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Obstacle barrier
            cv2.rectangle(img, (400, 180), (450, 260), (30, 30, 220), -1)
            cv2.putText(img, "HAZARD 2.0m", (375, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 2)

            # Rover pivoting left
            rx = int(260 - (rel_t / 6.0) * 70)
            ry = int(230 - (rel_t / 6.0) * 40)
            cv2.rectangle(img, (rx - 40, ry - 25), (rx + 40, ry + 25), (120, 110, 100), -1)
            cv2.arrowedLine(img, (rx, ry), (rx - 40, ry - 30), (0, 255, 255), 3)
            cv2.putText(img, "Left Turn", (rx - 40, ry + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Episode 3: 11.5 - 18.0s (Waypoint Bravo Reached)
        else:
            rel_t = t - 11.5
            img[:] = (20, 45, 20) # Success dark green
            cv2.putText(img, "Autonomous Robotics Lab: Mission Log", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(img, "Phase 3: Waypoint Bravo Reached Successfully", (35, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 120), 2)
            cv2.putText(img, "Mission Coordinates: Target (45.2, 12.8) Secured", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(img, f"Timestamp: 00:{int(t):02d}s", (35, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            # Target Waypoint Flag / Beacon
            wx, wy = 360, 220
            cv2.circle(img, (wx, wy), 22, (0, 255, 180), -1)
            cv2.circle(img, (wx, wy), 35, (0, 255, 180), 2)
            cv2.putText(img, "WAYPOINT BRAVO", (wx - 75, wy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 150), 2)

        out.write(img)

    out.release()

    # Mux with FFmpeg
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
    subprocess.run(cmd, check=True, capture_output=True)
    temp_avi.unlink(missing_ok=True)
    temp_wav.unlink(missing_ok=True)
    print(f"Robotics demo generated: {output_mp4}")

if __name__ == "__main__":
    generate_cooking_demo()
    generate_robotics_demo()
