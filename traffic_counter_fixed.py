
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import solutions

# =========================================
# TRAFFICFLOW CV
# Simplified Vehicle Analytics
# =========================================

BASE = Path(__file__).resolve().parent
VIDEO = BASE / "videos" / "traffic_test.mp4"

RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

OUTPUT = RESULTS / "trafficflow_final.mp4"
CSV_OUTPUT = RESULTS / "trafficflow_events.csv"

MAX_WIDTH = 1200
SIDE_WIDTH = 340

# Original COCO model classes.
# Do NOT remove bus: vans can be
# detected as either bus or truck.
CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

RAW_ORDER = [
    "car", "truck", "bus", "motorcycle"
]

# Display groups only. No effect on tracking.
GROUPS = {
    "car": "car",
    "truck": "other",
    "bus": "other",
    "motorcycle": "motorcycle",
}

DISPLAY_ORDER = [
    "car", "other", "motorcycle"
]

TITLES = {
    "car": "CARS",
    "other": "OTHER VEHICLES",
    "motorcycle": "MOTORCYCLES",
}

COLORS = {
    "car": (113, 221, 144),
    "other": (240, 179, 84),
    "motorcycle": (236, 208, 112),
}

BG = (25, 24, 23)
CARD = (42, 39, 37)
WHITE = (244, 244, 244)
MUTED = (167, 171, 176)
ACCENT = (135, 230, 167)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_text(
    image, value, x, y,
    size=0.55, color=WHITE, thickness=1
):
    cv2.putText(
        image,
        str(value),
        (int(x), int(y)),
        FONT,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_fill(image, x1, y1, x2, y2, color):
    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        -1,
    )


# =========================================
# ORIGINAL LINE SELECTION
# =========================================

def select_line(frame):
    points = []
    window = "TrafficFlow CV - Counting Line"

    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) == 2:
                points.clear()

            points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    print("Draw the SAME counting line as before.")
    print("S = START | R = RESET | Q = QUIT")

    while True:
        preview = frame.copy()
        height, width = preview.shape[:2]

        draw_fill(
            preview, 0, 0, width, 55, BG
        )

        draw_text(
            preview,
            "2 POINTS | S START | R RESET | Q QUIT",
            15, 35, 0.58, WHITE, 2
        )

        for index, point in enumerate(points):
            cv2.circle(
                preview, point, 7,
                ACCENT, -1, cv2.LINE_AA
            )

            draw_text(
                preview,
                "AB"[index],
                point[0] + 12,
                point[1] - 9,
                0.55, WHITE, 2
            )

        if len(points) == 2:
            cv2.line(
                preview,
                points[0],
                points[1],
                ACCENT, 3, cv2.LINE_AA
            )

        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()

        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

        elif key == ord("s") and len(points) == 2:
            if math.dist(points[0], points[1]) >= 30:
                break

            print("Line is too short.")

    cv2.destroyAllWindows()
    return points


# =========================================
# OFFICIAL COUNTS, GROUPED FOR DISPLAY
# =========================================

def grouped_counts(result):
    raw = result.classwise_count or {}

    values = {
        name: {"IN": 0, "OUT": 0}
        for name in DISPLAY_ORDER
    }

    for raw_name in RAW_ORDER:
        target = GROUPS[raw_name]
        data = raw.get(raw_name, {})

        values[target]["IN"] += int(
            data.get("IN", 0)
        )

        values[target]["OUT"] += int(
            data.get("OUT", 0)
        )

    return values


# =========================================
# TRACKING VISUALS
# =========================================

def draw_tracking(
    frame, counter, line,
    flashes, frame_number
):
    a, b = line

    cv2.line(
        frame, a, b,
        (33, 38, 39), 7, cv2.LINE_AA
    )

    cv2.line(
        frame, a, b,
        ACCENT, 3, cv2.LINE_AA
    )

    for point in (a, b):
        cv2.circle(
            frame, point,
            5, ACCENT, -1, cv2.LINE_AA
        )

    height, width = frame.shape[:2]

    for box, track_id, class_id in zip(
        counter.boxes,
        counter.track_ids,
        counter.clss,
    ):
        raw_name = CLASS_NAMES.get(int(class_id))

        if raw_name is None:
            continue

        group = GROUPS[raw_name]

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        active = (
            flashes.get(int(track_id), -1)
            >= frame_number
        )

        color = (
            ACCENT if active else COLORS[group]
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3 if active else 2,
            cv2.LINE_AA
        )

        if x2 - x1 >= 40 and y2 - y1 >= 25:
            label = (
                f"{TITLES[group].rstrip('S')} "
                f"#{int(track_id)}"
            )

            (tw, th), _ = cv2.getTextSize(
                label, FONT, 0.43, 1
            )

            lx = min(
                x1,
                max(0, width - tw - 12)
            )
            ly = max(0, y1 - th - 10)

            draw_fill(
                frame,
                lx, ly,
                lx + tw + 11,
                ly + th + 9,
                BG
            )

            draw_text(
                frame, label,
                lx + 5, ly + th + 2,
                0.43, color
            )

        if active:
            center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            cv2.circle(
                frame,
                center, 18,
                ACCENT, 3, cv2.LINE_AA
            )

    return frame


# =========================================
# CLEAN THREE-CATEGORY DASHBOARD
# =========================================

def draw_dashboard(
    frame, counts,
    in_count, out_count,
    active_tracks, last_event,
    video_second, duration,
    flash_active
):
    height, width = frame.shape[:2]
    full_width = width + SIDE_WIDTH

    full_height = max(
        height + 165,
        math.ceil(full_width * 9 / 16)
    )

    if full_height % 2:
        full_height += 1

    top = full_height - height

    canvas = np.full(
        (full_height, full_width, 3),
        BG, dtype=np.uint8
    )

    canvas[top:top + height, :width] = frame

    draw_text(
        canvas,
        "TRAFFICFLOW CV",
        26, 51, 0.94, WHITE, 2
    )

    draw_text(
        canvas,
        "VEHICLE FLOW ANALYTICS",
        27, 79, 0.48, MUTED
    )

    draw_text(
        canvas,
        "YOLO11s / ByteTrack / ObjectCounter",
        27, 108, 0.43, ACCENT
    )

    draw_text(
        canvas,
        f"VIDEO {video_second:05.1f}s / "
        f"{duration:05.1f}s",
        27, top - 18, 0.45, MUTED
    )

    # Exactly three visual categories
    start_x = 440
    gap = 10

    card_width = (
        full_width - start_x - 24 - gap * 2
    ) // 3

    for index, name in enumerate(DISPLAY_ORDER):
        x = start_x + index * (card_width + gap)

        draw_fill(
            canvas,
            x, 22,
            x + card_width, top - 19,
            CARD
        )

        draw_fill(
            canvas,
            x, 22,
            x + 4, top - 19,
            COLORS[name]
        )

        class_in = counts[name]["IN"]
        class_out = counts[name]["OUT"]
        total = class_in + class_out

        draw_text(
            canvas,
            TITLES[name],
            x + 15, 50,
            0.45, MUTED
        )

        draw_text(
            canvas,
            f"{total:02d}",
            x + 15, 107,
            1.35, WHITE, 2
        )

        draw_text(
            canvas,
            f"IN {class_in}  OUT {class_out}",
            x + 15, 134,
            0.39, COLORS[name]
        )

    sx = width + 22

    draw_text(
        canvas,
        "TRAFFIC SUMMARY",
        sx, top + 38,
        0.61, WHITE, 2
    )

    draw_text(
        canvas,
        "TOTAL COUNTED",
        sx, top + 80,
        0.47, MUTED
    )

    draw_text(
        canvas,
        f"{in_count + out_count:02d}",
        sx, top + 144,
        1.95, ACCENT, 3
    )

    draw_fill(
        canvas,
        sx, top + 166,
        full_width - 22, top + 167,
        (61, 58, 54)
    )

    draw_text(
        canvas,
        f"IN    {in_count:02d}",
        sx, top + 215,
        0.72, WHITE, 2
    )

    draw_text(
        canvas,
        f"OUT   {out_count:02d}",
        sx, top + 254,
        0.72, WHITE, 2
    )

    draw_fill(
        canvas,
        sx, top + 276,
        full_width - 22, top + 277,
        (61, 58, 54)
    )

    draw_text(
        canvas,
        "LATEST EVENT",
        sx, top + 316,
        0.47, MUTED
    )

    if last_event is not None:
        group = GROUPS[last_event["class"]]

        draw_text(
            canvas,
            f"{group.upper()} "
            f"{last_event['direction']}",
            sx, top + 352,
            0.58, COLORS[group], 2
        )

        draw_text(
            canvas,
            f"AT {last_event['second']:.2f}s",
            sx, top + 382,
            0.46, MUTED
        )
    else:
        draw_text(
            canvas,
            "Waiting for crossing...",
            sx, top + 352,
            0.48, MUTED
        )

    draw_text(
        canvas,
        f"ACTIVE TRACKS  {active_tracks}",
        sx, top + 427,
        0.47, WHITE
    )

    if flash_active:
        draw_text(
            canvas,
            "NEW CROSSING +1",
            sx, top + 468,
            0.60, ACCENT, 2
        )

    draw_text(
        canvas,
        "OTHER = BUS + TRUCK MODEL CLASSES",
        sx, full_height - 69,
        0.34, MUTED
    )

    draw_text(
        canvas,
        "COUNTING: ULTRALYTICS OBJECTCOUNTER",
        sx, full_height - 44,
        0.33, MUTED
    )

    return canvas, (full_width, full_height)


# =========================================
# MAIN - UNCHANGED COUNTING METHOD
# =========================================

def main():
    if not VIDEO.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO}"
        )

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if fps <= 0 or fps > 120:
        fps = 25.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    success, first = cap.read()

    if not success:
        cap.release()
        raise RuntimeError("Empty video.")

    original_h, original_w = first.shape[:2]

    scale = min(1.0, MAX_WIDTH / original_w)

    width = max(
        2,
        int(original_w * scale) // 2 * 2
    )

    height = max(
        2,
        int(original_h * scale) // 2 * 2
    )

    first = cv2.resize(first, (width, height))

    try:
        line = select_line(first)
    except BaseException:
        cap.release()
        raise

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Keep all four original COCO classes
    counter = solutions.ObjectCounter(
        model="yolo11s.pt",
        region=line,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.30,
        imgsz=640,
        device="cpu",
        show=False,
        verbose=False,
    )

    empty_counts = {
        name: {"IN": 0, "OUT": 0}
        for name in DISPLAY_ORDER
    }

    _, output_size = draw_dashboard(
        first,
        empty_counts,
        0, 0, 0,
        None,
        0,
        total_frames / fps,
        False,
    )

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Cannot create output video."
        )

    events = []
    last_event = None
    flashes = {}
    frame_number = 0
    last_result = None

    # Raw class count history for CSV events
    previous_raw = {
        name: {"IN": 0, "OUT": 0}
        for name in RAW_ORDER
    }

    print("\nTrafficFlow CV started...")
    print("Original counting logic active.")

    try:
        while True:
            success, image = cap.read()

            if not success:
                break

            frame = cv2.resize(
                image, (width, height)
            )

            old_counted_ids = set(
                counter.counted_ids
            )

            # Original Ultralytics method:
            # detection + tracking + counting
            result = counter(frame.copy())

            last_result = result
            counts = grouped_counts(result)

            video_second = frame_number / fps

            new_ids = (
                set(counter.counted_ids)
                - old_counted_ids
            )

            new_ids_by_class = {
                name: []
                for name in RAW_ORDER
            }

            for track_id, class_id in zip(
                counter.track_ids,
                counter.clss,
            ):
                class_name = CLASS_NAMES.get(
                    int(class_id)
                )

                if (
                    class_name in new_ids_by_class
                    and int(track_id) in new_ids
                ):
                    new_ids_by_class[class_name].append(
                        int(track_id)
                    )

                    flashes[int(track_id)] = (
                        frame_number
                        + max(1, round(fps * 0.75))
                    )

            new_event = False
            current_raw = (
                result.classwise_count or {}
            )

            for name in RAW_ORDER:
                for direction in ("IN", "OUT"):
                    before = previous_raw[name][direction]

                    after = int(
                        current_raw.get(name, {}).get(
                            direction, 0
                        )
                    )

                    difference = max(
                        0, after - before
                    )

                    for _ in range(difference):
                        ids = new_ids_by_class[name]

                        track_id = (
                            ids.pop(0) if ids else ""
                        )

                        event = {
                            "second": round(
                                video_second, 2
                            ),
                            "class": name,
                            "direction": direction,
                            "track_id": track_id,
                        }

                        events.append(event)
                        last_event = event
                        new_event = True

                    previous_raw[name][direction] = after

            flashes = {
                tid: end_frame
                for tid, end_frame in flashes.items()
                if end_frame >= frame_number
            }

            annotated = draw_tracking(
                frame,
                counter,
                line,
                flashes,
                frame_number,
            )

            canvas, _ = draw_dashboard(
                annotated,
                counts,
                int(result.in_count),
                int(result.out_count),
                len(counter.track_ids),
                last_event,
                video_second,
                total_frames / fps,
                new_event or bool(flashes),
            )

            writer.write(canvas)
            frame_number += 1

            if frame_number % 50 == 0:
                print(
                    f"Processed {frame_number}/"
                    f"{total_frames} frames",
                    end="\r",
                )

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    with CSV_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        csv_writer = csv.writer(file)

        csv_writer.writerow([
            "video_second",
            "model_class",
            "display_group",
            "direction",
            "track_id",
        ])

        for event in events:
            csv_writer.writerow([
                event["second"],
                event["class"],
                GROUPS[event["class"]],
                event["direction"],
                event["track_id"],
            ])

    print("\n========== FINAL RESULTS ==========")

    if last_result is not None:
        print("IN:", last_result.in_count)
        print("OUT:", last_result.out_count)

        print(
            "TOTAL:",
            last_result.in_count
            + last_result.out_count,
        )

        print(
            "MODEL CLASSES:",
            last_result.classwise_count,
        )

    print("\nVideo:", OUTPUT)
    print("Events:", CSV_OUTPUT)


if __name__ == "__main__":
    main()
