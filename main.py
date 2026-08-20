"""
================================================================================
 SPEED VIOLATION DETECTOR
================================================================================
An end-to-end computer vision pipeline that:
    1. Reads a traffic video.
    2. Detects & tracks vehicles frame-to-frame using YOLOv8's built-in tracker
       (ByteTrack under the hood), which assigns a STABLE ID to each vehicle
       for as long as it stays in view.
    3. Estimates each vehicle's real-world speed (km/h) from how far its
       center point moves (in pixels) between frames, converted to meters
       using a calibration constant (PIXELS_PER_METER).
    4. Smooths the speed estimate using a rolling average to remove jitter
       caused by detection-box noise.
    5. Flags vehicles that exceed a configurable SPEED_LIMIT, drawing a RED
       box + "VIOLATION" text on them, and GREEN boxes on legal vehicles.
    6. Writes the annotated video to disk.

Author: (generated for user)
Dependencies: pip install ultralytics opencv-python numpy
================================================================================
"""

import math
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# 1. CONFIGURATION — EDIT THESE VALUES FOR YOUR OWN CAMERA / VIDEO
# ==============================================================================

# --- File paths -------------------------------------------------------------
INPUT_VIDEO_PATH = "data/traffic.mp4"
OUTPUT_VIDEO_PATH = "data/output_violators.mp4"

# --- YOLO model ---------------------------------------------------------------
# "yolov8n.pt" (nano) is the smallest/fastest model — good default for CPU use.
# You can swap in "yolov8s.pt", "yolov8m.pt", or newer (e.g. "yolo11n.pt") for
# more accuracy at the cost of speed. Ultralytics will auto-download weights
# the first time the script runs if they are not already cached locally.
MODEL_PATH = "yolov8n.pt"

# COCO class names that correspond to vehicles we care about. YOLOv8's default
# weights are trained on the COCO dataset, where these classes already exist:
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}

# --- Speed estimation calibration --------------------------------------------
# FPS: frames per second of the INPUT video. This tells us how much real time
# passes between two consecutive frames (dt = 1 / FPS seconds). If you don't
# know your video's FPS, the script will auto-read it from the video file
# below (cv2.CAP_PROP_FPS) and override this default — but keep this as a
# sane fallback in case that metadata is missing/corrupted.
FPS = 30.0

# PIXELS_PER_METER: THE MOST IMPORTANT CALIBRATION VALUE.
#
# Your camera sees the real world projected onto a 2D pixel grid. To turn a
# "distance traveled in pixels" into a "distance traveled in meters", you
# need to know how many pixels correspond to one real-world meter *in the
# region of the frame where vehicles are being tracked*.
#
# HOW TO CALIBRATE IT (pick one method):
#   A) Known object method: Find an object of known real-world length that
#      lies flat on the road plane in your video (e.g., a lane width is
#      typically ~3.7 m in the US, a standard parking space is ~5.5 m long,
#      dashed lane-line segments are often ~3 m long with ~9 m gaps in the
#      US MUTCD standard). Pause the video, measure that object's length in
#      pixels (e.g., using an image editor or by clicking two points in
#      OpenCV), then:
#           PIXELS_PER_METER = pixel_length_of_object / real_world_length_m
#
#   B) Camera geometry method: If you know the camera's height, tilt angle,
#      and focal length, you can compute a ground-plane homography and get a
#      much more accurate (and position-dependent) pixel-to-meter mapping.
#      This script uses a SIMPLE constant-scale approximation, which works
#      reasonably well for a camera looking mostly straight down a fairly
#      flat, short stretch of road. For long stretches with strong
#      perspective, consider a full homography (cv2.getPerspectiveTransform)
#      instead of a single constant.
#
# Example placeholder value — REPLACE with your own measured value:
PIXELS_PER_METER = 8.0

# --- Violation rule -----------------------------------------------------------
SPEED_LIMIT = 60.0  # km/h — vehicles faster than this are flagged as violators

# --- Smoothing ------------------------------------------------------------
# Number of recent speed readings to average per vehicle, to prevent a single
# noisy detection (e.g., box briefly jumping a few pixels due to occlusion)
# from causing a wild, unrealistic speed spike on screen.
SMOOTHING_WINDOW = 5  # frames (must be within the 3-5 range requested)

# Minimum number of position samples required before we trust a speed value
# enough to display it (avoids a "flash" of speed=0 on a vehicle's first
# frame, when we have no previous position to compare against yet).
MIN_TRACK_HISTORY_FOR_SPEED = 2

# --- Display / drawing options ----------------------------------------------
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
COLOR_GREEN = (0, 200, 0)      # BGR — legal speed
COLOR_RED = (0, 0, 255)        # BGR — violation
COLOR_TEXT_BG = (0, 0, 0)      # background box behind text, for readability


# ==============================================================================
# 2. HELPER: convert a pixel displacement between two frames into km/h
# ==============================================================================
def pixel_distance_to_kmh(pixel_distance: float, dt_seconds: float,
                           pixels_per_meter: float) -> float:
    """
    Convert how many pixels a vehicle's center point moved between two
    frames into a speed in kilometers per hour.

    THE MATH:
        1. distance_in_meters = pixel_distance / pixels_per_meter
           (we divide by our calibration constant to go from "pixels" to
            "real-world meters")
        2. speed_in_meters_per_second = distance_in_meters / dt_seconds
           (dt_seconds is the real time that elapsed between the two frames,
            i.e., 1 / FPS for consecutive frames)
        3. speed_in_kmh = speed_in_meters_per_second * 3.6
           (there are 3.6 km/h for every 1 m/s: 1 m/s = 3600 m/hr = 3.6 km/hr)

    Args:
        pixel_distance: Euclidean distance (in pixels) the vehicle's center
                         moved between two frames.
        dt_seconds:      Real-world time elapsed between those two frames.
        pixels_per_meter: Calibration constant (see PIXELS_PER_METER above).

    Returns:
        Speed in km/h as a float.
    """
    if dt_seconds <= 0:
        return 0.0

    distance_in_meters = pixel_distance / pixels_per_meter
    speed_mps = distance_in_meters / dt_seconds          # meters per second
    speed_kmh = speed_mps * 3.6                          # meters/sec -> km/hr
    return speed_kmh


# ==============================================================================
# 3. MAIN PIPELINE
# ==============================================================================
def main():
    # --- Load the YOLO model --------------------------------------------------
    # Ultralytics handles downloading + loading the network weights for us.
    model = YOLO(MODEL_PATH)

    # Build a quick lookup of {class_id: class_name} from the model itself,
    # so we can filter detections to only the vehicle classes we care about,
    # regardless of which YOLO checkpoint/version is loaded.
    class_names = model.names  # dict like {0: 'person', 1: 'bicycle', 2: 'car', ...}
    vehicle_class_ids = {
        class_id for class_id, name in class_names.items()
        if name in VEHICLE_CLASS_NAMES
    }

    # --- Open the input video ---------------------------------------------
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open input video at '{INPUT_VIDEO_PATH}'. "
            f"Check that the file exists and the path is correct."
        )

    # Prefer the video's real FPS metadata over our hardcoded default, since
    # this directly controls the dt_seconds used in the speed formula above.
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = video_fps if video_fps and video_fps > 1 else FPS
    dt_seconds = 1.0 / fps  # real time elapsed between two consecutive frames

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Set up the output video writer ----------------------------------
    # 'mp4v' is a widely-supported codec fourcc for writing .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps,
                              (frame_width, frame_height))

    # --- Per-vehicle tracking state ----------------------------------------
    # For every tracked vehicle ID, we keep:
    #   - a short history of its recent (x, y) center positions AND the
    #     frame index they were seen at (so dt can account for any
    #     skipped/missing detections between frames)
    #   - a short history of recent speed readings, used for smoothing
    #
    # `deque(maxlen=N)` automatically discards the oldest entry once it's
    # full, giving us a rolling window without manual bookkeeping.
    track_positions = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW + 1))
    track_speeds = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

    frame_index = 0

    print(f"Processing '{INPUT_VIDEO_PATH}' "
          f"({frame_width}x{frame_height} @ {fps:.2f} FPS)...")

    # --- Main frame-by-frame loop -------------------------------------------
    while True:
        success, frame = cap.read()
        if not success:
            break  # end of video

        frame_index += 1

        # model.track() runs detection AND tracking together. YOLO's default
        # tracker (ByteTrack) associates detections across frames and assigns
        # each object a persistent `track_id` for as long as it's visible.
        # persist=True tells the tracker to remember state between calls
        # (i.e., across the frames of this video), rather than resetting
        # every call.
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=list(vehicle_class_ids),  # only detect vehicle classes
        )

        annotated_frame = frame.copy()

        # results is a list (one per input image); we only passed one frame.
        result = results[0]

        # If the tracker didn't find any boxes (or lost all tracks) this
        # frame, boxes.id will be None — handle that gracefully.
        if result.boxes is not None and result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()      # [N, 4] -> x1,y1,x2,y2
            track_ids = result.boxes.id.cpu().numpy().astype(int)  # [N]
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # [N]

            for box, track_id, cls_id in zip(boxes_xyxy, track_ids, class_ids):
                x1, y1, x2, y2 = box
                vehicle_label = class_names.get(cls_id, "vehicle")

                # --- Compute the center point of the bounding box ---------
                # We use the box center (rather than a corner) because it's
                # more stable — it wobbles less than the edges as the
                # detector's box size fluctuates slightly frame to frame.
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # --- Update this vehicle's position history ----------------
                track_positions[track_id].append((frame_index, center_x, center_y))

                # --- Estimate instantaneous speed from the last two points -
                current_speed_kmh = None
                history = track_positions[track_id]
                if len(history) >= MIN_TRACK_HISTORY_FOR_SPEED:
                    prev_frame_idx, prev_x, prev_y = history[-2]
                    curr_frame_idx, curr_x, curr_y = history[-1]

                    # Euclidean pixel distance between the vehicle's center
                    # in the previous sample vs. the current sample:
                    #   distance = sqrt( (dx)^2 + (dy)^2 )
                    pixel_dist = math.hypot(curr_x - prev_x, curr_y - prev_y)

                    # Real time elapsed between those two samples. Normally
                    # this is exactly one frame (dt_seconds), but we scale by
                    # however many frames actually passed in case the
                    # tracker temporarily lost this vehicle for a frame or
                    # two (frame_index difference > 1).
                    frames_elapsed = max(curr_frame_idx - prev_frame_idx, 1)
                    elapsed_seconds = frames_elapsed * dt_seconds

                    current_speed_kmh = pixel_distance_to_kmh(
                        pixel_dist, elapsed_seconds, PIXELS_PER_METER
                    )

                    # --- Smoothing: push into rolling window, then average -
                    # Averaging the last SMOOTHING_WINDOW readings removes
                    # the small frame-to-frame jitter that comes from the
                    # bounding box edges wobbling by a pixel or two, which
                    # would otherwise make the displayed speed number jump
                    # around erratically even when the vehicle's true speed
                    # is roughly constant.
                    track_speeds[track_id].append(current_speed_kmh)

                smoothed_speed_kmh = (
                    sum(track_speeds[track_id]) / len(track_speeds[track_id])
                    if track_speeds[track_id] else 0.0
                )

                # --- Decide box color + label based on the speed limit -----
                is_violation = smoothed_speed_kmh > SPEED_LIMIT
                color = COLOR_RED if is_violation else COLOR_GREEN

                if is_violation:
                    label_text = f"VIOLATION: {smoothed_speed_kmh:.1f} km/h"
                else:
                    label_text = f"ID {track_id} {vehicle_label}: {smoothed_speed_kmh:.1f} km/h"

                # --- Draw the bounding box ----------------------------------
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(annotated_frame, pt1, pt2, color, BOX_THICKNESS)

                # --- Draw a filled label background + text for readability -
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, FONT, FONT_SCALE, FONT_THICKNESS
                )
                label_bg_pt1 = (pt1[0], max(pt1[1] - text_h - baseline - 6, 0))
                label_bg_pt2 = (pt1[0] + text_w + 6, pt1[1])
                cv2.rectangle(annotated_frame, label_bg_pt1, label_bg_pt2,
                              color, cv2.FILLED)
                cv2.putText(
                    annotated_frame, label_text,
                    (pt1[0] + 3, pt1[1] - 5),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
                    cv2.LINE_AA,
                )

        # --- Overlay a small info banner (frame number + speed limit) -----
        info_text = f"Frame {frame_index} | Speed limit: {SPEED_LIMIT:.0f} km/h"
        cv2.putText(annotated_frame, info_text, (10, 25), FONT, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # --- Write this annotated frame to the output video ----------------
        writer.write(annotated_frame)

        if frame_index % 50 == 0:
            print(f"  ...processed {frame_index} frames")

    # --- Cleanup ---------------------------------------------------------
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done. Annotated video saved to '{OUTPUT_VIDEO_PATH}' "
          f"({frame_index} frames processed).")


if __name__ == "__main__":
    main()
