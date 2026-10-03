"""
Automated Tests for OMR MCQ Automated Grading System.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from src.config import (
    ASSETS_DIR,
    BENCHMARK_EXPECTED,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SCANNED_DIR,
    get_sample_images,
)
from src.grader import OMRGrader, OMRResult
from src.utils import reorder, split_boxes


class TestOMRGrader:
    """Test suite for the core OMR grading engine."""

    @pytest.fixture
    def grader(self):
        return OMRGrader()

    def test_benchmark_sheets(self, grader):
        """
        Verify that all 5 benchmark exam sheets are graded with 100% accuracy.
        """
        sample_images = get_sample_images()
        assert len(sample_images) >= 5, "Benchmark images 1.jpg to 5.jpg must be present."

        for img_path in sample_images:
            if img_path.name not in BENCHMARK_EXPECTED:
                continue

            expected_answers, expected_score = BENCHMARK_EXPECTED[img_path.name]
            frame = cv2.imread(str(img_path))
            assert frame is not None, f"Failed to load image: {img_path}"

            result: OMRResult = grader.process(frame)

            assert result.success is True, f"Detection failed for {img_path.name}: {result.error_message}"
            assert result.score == expected_score, (
                f"Score mismatch on {img_path.name}: expected {expected_score}, got {result.score}"
            )
            assert result.student_answers == expected_answers, (
                f"Answers mismatch on {img_path.name}: expected {expected_answers}, got {result.student_answers}"
            )
            assert result.annotated_frame is not None
            assert result.annotated_frame.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    def test_custom_answer_key(self):
        """Verify that a custom answer key correctly adjusts the score."""
        perfect_key_for_1 = [1, 2, 0, 0, 4]
        custom_grader = OMRGrader(answer_key=perfect_key_for_1)

        sample_images = {p.name: p for p in get_sample_images()}
        frame = cv2.imread(str(sample_images["1.jpg"]))
        result = custom_grader.process(frame)

        assert result.success is True
        assert result.score == 100.0
        assert result.grading == [1, 1, 1, 1, 1]
        assert result.letter_grade == "A+"


class TestOMRUtilities:
    """Test suite for CV utility functions."""

    def test_reorder_points(self):
        """Verify that 4 corner points are reordered to [TL, TR, BL, BR]."""
        scrambled = np.array([
            [[200, 100]],  # BR
            [[0, 0]],      # TL
            [[200, 0]],    # TR
            [[0, 100]],    # BL
        ], dtype=np.int32)

        ordered = reorder(scrambled)
        assert np.array_equal(ordered[0][0], [0, 0])      # Top-Left
        assert np.array_equal(ordered[1][0], [200, 0])    # Top-Right
        assert np.array_equal(ordered[2][0], [0, 100])    # Bottom-Left
        assert np.array_equal(ordered[3][0], [200, 100])  # Bottom-Right

    def test_split_boxes(self):
        """Verify that split_boxes creates (questions * choices) equal-sized boxes."""
        img = np.zeros((700, 700), dtype=np.uint8)
        boxes = split_boxes(img, questions=5, choices=5)

        assert len(boxes) == 25
        for box in boxes:
            assert box.shape == (140, 140)


class TestOMRPathsAndConfig:
    """Test suite for path resolution and directory structure."""

    def test_assets_directory_and_files(self):
        """Verify that assets directory exists and contains all required media assets."""
        assert ASSETS_DIR.exists(), f"Assets directory not found at: {ASSETS_DIR}"
        assert ASSETS_DIR.is_dir()
        assert DEFAULT_VIDEO_PATH.exists(), f"Default video not found at: {DEFAULT_VIDEO_PATH}"
        assert DEFAULT_IMAGE_PATH.exists(), f"Default image not found at: {DEFAULT_IMAGE_PATH}"

        sample_images = get_sample_images()
        assert len(sample_images) >= 5, f"Expected at least 5 sample sheets, found {len(sample_images)}"
        for img in sample_images:
            assert img.exists(), f"Sample image file missing: {img}"

    def test_scanned_directory_path(self):
        """Verify that Scanned directory path is configured correctly."""
        assert SCANNED_DIR.name == "Scanned"

