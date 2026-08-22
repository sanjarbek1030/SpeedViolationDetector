"""
Dependencies:
    pip install ultralytics opencv-python numpy
"""

import math
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# 1. CONFIGURATION

INPUT_VIDEO_PATH = "data/traffic.mp4"
OUTPUT_VIDEO_PATH = "data/output_violators.mp4"

MODEL_PATH = "yolov8n.pt"

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}

FPS = 30.0
PIXELS_PER_METER = 8.0
SPEED_LIMIT = 60.0  # km/h

ROAD_TYPE = "two_way"

ROAD_DIRECTION = (1.0, 0.0)

ONE_WAY_DIRECTION = "FORWARD"

MIN_DIRECTION_PIXELS = 2.0

SMOOTHING_WINDOW = 5

MIN_TRACK_HISTORY = 2

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
COLOR_GREEN = (0, 200, 0)     # legal / normal
COLOR_RED = (0, 0, 255)      # violation
COLOR_YELLOW = (0, 255, 255) # wrong-way warning
COLOR_TEXT_BG = (0, 0, 0)

# 2. HELPERS

def pixel_distance_to_kmh(
    pixel_distance: float,
    dt_seconds: float,
    pixels_per_meter: float,
) -> float:
    """Convert pixel displacement into km/h using a simple calibration."""
    if dt_seconds <= 0 or pixels_per_meter <= 0:
        return 0.0

    distance_in_meters = pixel_distance / pixels_per_meter
    speed_mps = distance_in_meters / dt_seconds
    return speed_mps * 3.6


def normalize_vector(x: float, y: float) -> tuple[float, float]:
    """Return a unit vector."""
    length = math.hypot(x, y)
    if length == 0:
        return 0.0, 0.0
    return x / length, y / length


def get_direction(dx: float, dy: float) -> str:
    """
    Determine whether the vehicle is moving FORWARD or REVERSE relative to
    ROAD_DIRECTION.

    The dot product tells us whether movement is along or against the road axis.
    """
    road_x, road_y = normalize_vector(*ROAD_DIRECTION)
    movement_x, movement_y = normalize_vector(dx, dy)

    if road_x == 0 and road_y == 0:
        return "UNKNOWN"

    dot = movement_x * road_x + movement_y * road_y

    if abs(dx) + abs(dy) < MIN_DIRECTION_PIXELS:
        return "UNKNOWN"

    return "FORWARD" if dot >= 0 else "REVERSE"


def is_wrong_way(direction: str) -> bool:
    """Return True when a vehicle is travelling the wrong way."""
    if ROAD_TYPE != "one_way":
        return False
    if direction == "UNKNOWN":
        return False
    return direction != ONE_WAY_DIRECTION


def draw_label(frame, text: str, x: int, y: int, color) -> None:
    """Draw readable text with a filled background."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, FONT, FONT_SCALE, FONT_THICKNESS
    )

    top = max(y - text_h - baseline - 6, 0)
    bottom = y

    cv2.rectangle(
        frame,
        (x, top),
        (x + text_w + 6, bottom),
        color,
        cv2.FILLED,
    )

    cv2.putText(
        frame,
        text,
        (x + 3, y - 5),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

# 3. MAIN PIPELINE

def main():
    model = YOLO(MODEL_PATH)

    class_names = model.names
    vehicle_class_ids = {
        class_id
        for class_id, name in class_names.items()
        if name in VEHICLE_CLASS_NAMES
    }

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open input video at '{INPUT_VIDEO_PATH}'. "
            "Check that the file exists and the path is correct."
        )

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = video_fps if video_fps and video_fps > 1 else FPS
    dt_seconds = 1.0 / fps

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video at '{OUTPUT_VIDEO_PATH}'."
        )

    track_positions = defaultdict(
        lambda: deque(maxlen=SMOOTHING_WINDOW + 1)
    )
    track_speeds = defaultdict(
        lambda: deque(maxlen=SMOOTHING_WINDOW)
    )

    frame_index = 0

    print(
        f"Processing '{INPUT_VIDEO_PATH}' "
        f"({frame_width}x{frame_height} @ {fps:.2f} FPS)..."
    )
    print(f"Road type: {ROAD_TYPE}")

    if ROAD_TYPE == "two_way":
        print("Two-way mode: FORWARD and REVERSE traffic are allowed.")
    else:
        print(f"One-way mode: allowed direction = {ONE_WAY_DIRECTION}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        annotated_frame = frame.copy()

        # YOLO detection + ByteTrack tracking.
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=list(vehicle_class_ids),
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(
                boxes_xyxy, track_ids, class_ids
            ):
                x1, y1, x2, y2 = box
                vehicle_label = class_names.get(cls_id, "vehicle")

                center_x = (x1 + x2) / 2.0
                center_y = y2

                track_positions[track_id].append(
                    (frame_index, center_x, center_y)
                )

                current_speed_kmh = 0.0
                direction = "UNKNOWN"
                history = track_positions[track_id]

                if len(history) >= MIN_TRACK_HISTORY:
                    prev_frame_idx, prev_x, prev_y = history[-2]
                    curr_frame_idx, curr_x, curr_y = history[-1]

                    dx = curr_x - prev_x
                    dy = curr_y - prev_y
                    pixel_dist = math.hypot(dx, dy)

                    frames_elapsed = max(curr_frame_idx - prev_frame_idx, 1)
                    elapsed_seconds = frames_elapsed * dt_seconds

                    current_speed_kmh = pixel_distance_to_kmh(
                        pixel_dist,
                        elapsed_seconds,
                        PIXELS_PER_METER,
                    )

                    direction = get_direction(dx, dy)
                    track_speeds[track_id].append(current_speed_kmh)

                smoothed_speed_kmh = (
                    sum(track_speeds[track_id]) / len(track_speeds[track_id])
                    if track_speeds[track_id]
                    else 0.0
                )

                speeding = smoothed_speed_kmh > SPEED_LIMIT
                wrong_way = is_wrong_way(direction)

                if wrong_way:
                    color = COLOR_YELLOW
                elif speeding:
                    color = COLOR_RED
                else:
                    color = COLOR_GREEN

                if wrong_way:
                    label_text = (
                        f"WRONG WAY | ID {track_id} | "
                        f"{smoothed_speed_kmh:.1f} km/h"
                    )
                elif speeding:
                    label_text = (
                        f"VIOLATION | ID {track_id} | "
                        f"{smoothed_speed_kmh:.1f} km/h"
                    )
                else:
                    label_text = (
                        f"ID {track_id} {vehicle_label} | "
                        f"{smoothed_speed_kmh:.1f} km/h"
                    )

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(
                    annotated_frame,
                    pt1,
                    pt2,
                    color,
                    BOX_THICKNESS,
                )
                draw_label(annotated_frame, label_text, pt1[0], pt1[1], color)

                if direction != "UNKNOWN":
                    direction_text = f"Direction: {direction}"
                    cv2.putText(
                        annotated_frame,
                        direction_text,
                        (int(x1), min(int(y2) + 20, frame_height - 5)),
                        FONT,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

        info_text = (
            f"Road: {ROAD_TYPE.upper()} | "
            f"Speed limit: {SPEED_LIMIT:.0f} km/h"
        )
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 25),
            FONT,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated_frame)

        if frame_index % 50 == 0:
            print(f"  ...processed {frame_index} frames")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(
        f"Done. Annotated video saved to '{OUTPUT_VIDEO_PATH}' "
        f"({frame_index} frames processed)."
    )


if __name__ == "__main__":
    main()
