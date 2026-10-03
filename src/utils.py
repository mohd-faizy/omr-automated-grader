"""
Computer Vision and Image Processing Utility Functions for OMR.
"""
from typing import List, Optional, Sequence, Union
import cv2
import numpy as np


def reorder(points: np.ndarray) -> np.ndarray:
    """
    Reorder 4 polygon corner points into consistent orientation:
    [0]: Top-Left [0, 0]
    [1]: Top-Right [w, 0]
    [2]: Bottom-Left [0, h]
    [3]: Bottom-Right [w, h]
    """
    pts = points.reshape((4, 2))
    reordered = np.zeros((4, 1, 2), dtype=np.int32)

    # Sum of coordinates: Top-Left has smallest sum, Bottom-Right has largest sum
    coord_sum = pts.sum(axis=1)
    reordered[0] = pts[np.argmin(coord_sum)]
    reordered[3] = pts[np.argmax(coord_sum)]

    # Difference of coordinates (y - x):
    # Top-Right [w, 0] has minimum (0 - w = -w)
    # Bottom-Left [0, h] has maximum (h - 0 = +h)
    coord_diff = np.diff(pts, axis=1)
    reordered[1] = pts[np.argmin(coord_diff)]
    reordered[2] = pts[np.argmax(coord_diff)]

    return reordered


def rect_contour(contours: Sequence[np.ndarray], min_area: float = 50.0) -> List[np.ndarray]:
    """
    Filter contours having 4 approximated vertices and area > min_area,
    sorted in descending order of contour area.
    """
    rect_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                rect_contours.append(cnt)
    return sorted(rect_contours, key=cv2.contourArea, reverse=True)


def get_corner_points(contour: np.ndarray) -> np.ndarray:
    """Approximate contour polygon to extract corner points."""
    perimeter = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.02 * perimeter, True)


def split_boxes(image: np.ndarray, questions: int = 5, choices: int = 5) -> List[np.ndarray]:
    """
    Split warped thresholded answer grid into individual question and choice bubble boxes.
    Returns list of (questions * choices) image patches in row-major order.
    """
    rows = np.vsplit(image, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choices)
        for box in cols:
            boxes.append(box)
    return boxes


def draw_grid(
    image: np.ndarray,
    questions: int = 5,
    choices: int = 5,
    color: Sequence[int] = (255, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw question and choice dividing grid lines on warped image."""
    h, w = image.shape[:2]
    sec_w = int(w / choices)
    sec_h = int(h / questions)

    for i in range(1, max(questions, choices)):
        if i < questions:
            cv2.line(image, (0, sec_h * i), (w, sec_h * i), color, thickness)
        if i < choices:
            cv2.line(image, (sec_w * i, 0), (sec_w * i, h), color, thickness)

    return image


def show_answers(
    image: np.ndarray,
    student_indices: List[int],
    grading: List[int],
    correct_answers: List[int],
    questions: int = 5,
    choices: int = 5
) -> np.ndarray:
    """
    Render visual answer circles on the warped question grid:
    - Green circle: Correct answer selection
    - Red circle: Incorrect answer selection
    - Inner green dot: Indication of actual correct answer
    """
    h, w = image.shape[:2]
    sec_w = int(w / choices)
    sec_h = int(h / questions)
    radius = min(sec_w, sec_h) // 3

    for q_idx in range(questions):
        user_ans = student_indices[q_idx]
        c_x = (user_ans * sec_w) + (sec_w // 2)
        c_y = (q_idx * sec_h) + (sec_h // 2)

        if grading[q_idx] == 1:
            # Correct: Green circle
            cv2.circle(image, (c_x, c_y), radius, (0, 255, 0), cv2.FILLED)
        else:
            # Incorrect: Red circle on student's choice
            cv2.circle(image, (c_x, c_y), radius, (0, 0, 255), cv2.FILLED)
            # Correct target: Smaller green circle
            correct_choice = correct_answers[q_idx]
            correct_cx = (correct_choice * sec_w) + (sec_w // 2)
            correct_cy = (q_idx * sec_h) + (sec_h // 2)
            cv2.circle(image, (correct_cx, correct_cy), radius // 2, (0, 255, 0), cv2.FILLED)

    return image


def stack_images(
    image_array: Union[List[List[np.ndarray]], List[np.ndarray]],
    scale: float = 0.5,
    labels: Optional[List[List[str]]] = None
) -> np.ndarray:
    """
    Stack multiple images in a clean 2D grid with optional text labels for each cell.
    Uniformly handles both color (BGR) and grayscale images.
    """
    # Normalize 1D list to 2D list of 1 row
    if not isinstance(image_array[0], (list, tuple)):
        grid = [image_array]
    else:
        grid = [[img for img in row] for row in image_array]

    rows = len(grid)
    cols = len(grid[0])

    processed_rows = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            img = grid[r][c]
            # Convert grayscale to BGR if single channel
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Resize
            if scale != 1.0:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            row_imgs.append(img)
        processed_rows.append(np.hstack(row_imgs))

    stacked = np.vstack(processed_rows)

    # Optional label badges
    if labels:
        cell_w = int(stacked.shape[1] / cols)
        cell_h = int(stacked.shape[0] / rows)
        for r in range(rows):
            for c in range(cols):
                if r < len(labels) and c < len(labels[r]):
                    label_text = labels[r][c]
                    if label_text:
                        tx = c * cell_w + 10
                        ty = r * cell_h + 24
                        box_w = len(label_text) * 11 + 18
                        cv2.rectangle(stacked, (c * cell_w, r * cell_h),
                                      (c * cell_w + box_w, r * cell_h + 32), (25, 25, 25), cv2.FILLED)
                        cv2.putText(stacked, label_text, (tx, ty),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 1, cv2.LINE_AA)

    return stacked

