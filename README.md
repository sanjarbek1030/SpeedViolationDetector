# Speed Violation Detector

A real-time vehicle speed estimation and violation detection system built with **YOLOv8** (Ultralytics) and **OpenCV**. It detects and tracks vehicles in traffic footage, estimates their speed in km/h, and flags any vehicle exceeding a configurable speed limit with a red bounding box overlay.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

##  Demo

| Legal Speed | Violation |
|:---:|:---:|
| 🟩 Green box + speed overlay | 🟥 Red box + "VIOLATION" overlay |

##  Features

- **Multi-object tracking** — Uses YOLOv8's built-in ByteTrack tracker to assign a persistent ID to every vehicle across frames.
- **Vehicle-class filtering** — Detects cars, trucks, buses, and motorcycles (ignores pedestrians, cyclists, etc.).
- **Speed estimation** — Converts pixel displacement between frames into real-world km/h using a configurable `PIXELS_PER_METER` calibration constant.
- **Jitter smoothing** — Rolling average over the last 3–5 frames prevents noisy, jumpy speed readings.
- **Automatic violation flagging** — Vehicles over the speed limit get a red box + "VIOLATION" label; compliant vehicles get a green box.
- **Fully annotated output video** — Saves a processed `.mp4` with all overlays baked in.

##  Requirements

```bash
pip install ultralytics opencv-python numpy
```

Python 3.8+ recommended. A GPU is optional but speeds up inference significantly on longer videos.

##  Usage

1. Place your input video at `data/traffic.mp4` (or update `INPUT_VIDEO_PATH` in the script).
2. Run the script:

```bash
python speed_violation_detector.py
```

3. Find your annotated output at `data/output_violators.mp4`.

##  Configuration

All key parameters live at the top of `speed_violation_detector.py`:

| Variable | Description |
|---|---|
| `INPUT_VIDEO_PATH` / `OUTPUT_VIDEO_PATH` | Input/output video file paths |
| `MODEL_PATH` | YOLO checkpoint to use (default `yolov8n.pt`) |
| `PIXELS_PER_METER` | **Calibration constant** — see below |
| `FPS` | Fallback FPS if video metadata is unreadable |
| `SPEED_LIMIT` | Threshold (km/h) above which a vehicle is flagged |
| `SMOOTHING_WINDOW` | Number of frames averaged for speed smoothing (3–5) |

###  Calibrating `PIXELS_PER_METER`

This is the most important value to get right, since it converts pixel movement into real-world distance.

**Method: known-object measurement**
1. Pause your video on a frame showing a reference object of known real-world length lying on the road (e.g., a lane width ≈ 3.7 m, a parking space ≈ 5.5 m, a US lane-dash segment ≈ 3 m).
2. Measure that object's length in pixels (image editor, or click two points in OpenCV).
3. Compute:
   ```
   PIXELS_PER_METER = pixel_length_of_object / real_world_length_in_meters
   ```

For long stretches of road with strong perspective distortion, consider replacing the constant-scale approximation with a full ground-plane homography (`cv2.getPerspectiveTransform`) for higher accuracy.

##  How Speed Is Calculated

```
distance_in_meters       = pixel_distance / PIXELS_PER_METER
speed_in_meters_per_sec  = distance_in_meters / time_elapsed_seconds
speed_in_kmh             = speed_in_meters_per_sec * 3.6
```

Each vehicle's speed is smoothed by averaging its last `SMOOTHING_WINDOW` readings, removing frame-to-frame jitter from small bounding-box fluctuations.

##  Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection & tracking
- [OpenCV](https://opencv.org/) — video I/O and annotation
- Python 3 / NumPy

##  Limitations & Disclaimer

- Speed estimates are approximate and depend heavily on accurate calibration of `PIXELS_PER_METER` and camera angle.
- Not intended for legal/enforcement use without proper camera calibration, validation against ground truth, and compliance with local regulations.
- Works best on relatively flat, non-perspective-heavy road sections.

##  License

MIT — feel free to use, modify, and build on this project.
