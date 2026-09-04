"""Helper utilities for editing the project's demo videos."""

import os
import cv2
import numpy as np


def detect_content_bbox(input_path, threshold=10, sample_frames=8):
    """Sample frames and find the bounding box of non-black content."""
    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    col_max = np.zeros(w, dtype=np.uint8)
    row_max = np.zeros(h, dtype=np.uint8)
    idxs = sorted(set(int(i * total / sample_frames) for i in range(sample_frames)))
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        col_max = np.maximum(col_max, gray.max(axis=0))
        row_max = np.maximum(row_max, gray.max(axis=1))
    cap.release()

    cols = np.where(col_max > threshold)[0]
    rows = np.where(row_max > threshold)[0]
    x1, x2 = int(cols.min()), int(cols.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    return x1, y1, x2, y2


def crop_box(input_path, box, output_path=None):
    """Crop every frame to the given (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    overwrite = output_path is None
    final_path = input_path if overwrite else output_path
    write_path = final_path + ".tmp.mp4" if overwrite else final_path

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(write_path, fourcc, fps, (x2 - x1, y2 - y1))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame[y1:y2, x1:x2])

    cap.release()
    writer.release()

    if overwrite:
        os.replace(write_path, final_path)


def remove_black_bars(input_path, output_path=None, threshold=10, sample_frames=8):
    """Detect and crop away black letterbox/pillarbox bars."""
    box = detect_content_bbox(input_path, threshold=threshold, sample_frames=sample_frames)
    crop_box(input_path, box, output_path=output_path)


def crop_to_ratio(input_path, output_path=None, target_w=4, target_h=3,
                   h_align="center", v_align="top"):
    """Crop a video to a target aspect ratio (default 4:3).

    h_align: "left" | "center" | "right"
    v_align: "top" | "center" | "bottom"

    If output_path is omitted, the input file is overwritten in place.
    """
    overwrite = output_path is None
    final_path = input_path if overwrite else output_path
    write_path = final_path + ".tmp.mp4" if overwrite else final_path

    cap = cv2.VideoCapture(input_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # source is wider than target -> crop width, keep height
        new_h = src_h
        new_w = int(round(src_h * target_ratio))
    else:
        # source is taller than target -> crop height, keep width
        new_w = src_w
        new_h = int(round(src_w / target_ratio))

    if h_align == "left":
        x1 = 0
    elif h_align == "right":
        x1 = src_w - new_w
    else:
        x1 = (src_w - new_w) // 2

    if v_align == "top":
        y1 = 0
    elif v_align == "bottom":
        y1 = src_h - new_h
    else:
        y1 = (src_h - new_h) // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(write_path, fourcc, fps, (new_w, new_h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    if overwrite:
        os.replace(write_path, final_path)


if __name__ == "__main__":
    names = ["redtoy_f", "redtoy_s"]
    for name in names:
        path = f"static/videos/simulation/{name}.mp4"
        remove_black_bars(path)
        crop_to_ratio(
            path,
            target_w=4,
            target_h=3,
            h_align="center",
            v_align="top",
        )
