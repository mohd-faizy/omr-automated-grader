"""
Configuration and constants for Optical Mark Recognition (OMR) MCQ Automated Grading.
"""
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
if not ASSETS_DIR.exists() and (PROJECT_ROOT / "Assets").exists():
    ASSETS_DIR = PROJECT_ROOT / "Assets"

SCANNED_DIR = PROJECT_ROOT / "Scanned"
DEFAULT_VIDEO_PATH = ASSETS_DIR / "demo_vid.mp4"
DEFAULT_IMAGE_PATH = ASSETS_DIR / "1.jpg"

# ============================================================================
# OMR MCQ Exam Parameters
# ============================================================================
QUESTIONS_COUNT: int = 5
CHOICES_COUNT: int = 5
CHOICE_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

# Default Answer Key: Q1=B, Q2=C, Q3=A, Q4=C, Q5=E
DEFAULT_ANSWERS: List[int] = [1, 2, 0, 2, 4]

# ============================================================================
# Image Dimensions & Normalization
# ============================================================================
IMAGE_WIDTH: int = 700
IMAGE_HEIGHT: int = 700
GRADE_BOX_WIDTH: int = 325
GRADE_BOX_HEIGHT: int = 150

# ============================================================================
# Computer Vision Processing Parameters
# ============================================================================
GAUSSIAN_KERNEL: Tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: int = 1
CANNY_LOW_THRESH: int = 10
CANNY_HIGH_THRESH: int = 70
BINARY_INV_THRESH: int = 170

# Diagnostic Dashboard Stage Labels
STAGE_LABELS = [
    ["Original", "Gray", "Edges", "Contours"],
    ["Biggest Contour", "Threshold", "Warped", "Final"]
]

# Benchmark Sheets Expected Results (filename -> (answers, score))
BENCHMARK_EXPECTED = {
    "1.jpg": ([1, 2, 0, 0, 4], 80.0),
    "2.jpg": ([1, 2, 0, 2, 4], 100.0),
    "3.jpg": ([4, 2, 0, 0, 3], 40.0),
    "4.jpg": ([4, 4, 1, 2, 2], 20.0),
    "5.jpg": ([0, 4, 3, 3, 0], 0.0),
}


def get_sample_images() -> List[Path]:
    """Return available sample exam sheet paths in sorted order."""
    image_names = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "MCQPaper.jpg"]
    found = [ASSETS_DIR / name for name in image_names if (ASSETS_DIR / name).exists()]
    if ASSETS_DIR.exists():
        extras = sorted([
            p for p in ASSETS_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"] and p not in found
        ])
        found.extend(extras)
    return found


def get_grade_info(score: float) -> Tuple[str, Tuple[int, int, int]]:
    """Return the letter grade and BGR display color for a given percentage score."""
    if score >= 80.0:
        letter = "A+" if score > 80.0 else "A"
        color = (0, 220, 50)     # Bright Green
    elif score >= 60.0:
        letter = "B"
        color = (0, 215, 255)    # Yellow / Gold
    elif score >= 40.0:
        letter = "C"
        color = (0, 140, 255)    # Orange
    else:
        letter = "F"
        color = (30, 30, 220)    # Red
    return letter, color
