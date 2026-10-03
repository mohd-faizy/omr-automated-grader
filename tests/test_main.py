"""
Tests for main CLI entry point, answer key parser, and path configurations.
"""
from pathlib import Path
import pytest

from main import parse_answer_key, run_benchmark_test
from src.config import (
    ASSETS_DIR,
    DEFAULT_ANSWERS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    SCANNED_DIR,
)


class TestMainCLI:
    """Test suite for CLI functions and answer key parser."""

    def test_parse_answer_key_letters(self):
        """Test parsing comma-separated letter choices."""
        key = parse_answer_key("B,C,A,C,E")
        assert key == [1, 2, 0, 2, 4]

    def test_parse_answer_key_indices(self):
        """Test parsing comma-separated integer indices."""
        key = parse_answer_key("1, 2, 0, 2, 4")
        assert key == [1, 2, 0, 2, 4]

    def test_parse_answer_key_invalid(self):
        """Test that invalid tokens raise ValueError."""
        with pytest.raises(ValueError):
            parse_answer_key("X,Y,Z")

        with pytest.raises(ValueError):
            parse_answer_key("10,20")

    def test_benchmark_run(self, capsys):
        """Test headless benchmark execution via CLI helper."""
        run_benchmark_test()
        captured = capsys.readouterr()
        assert "All benchmark sheets evaluated with 100% accuracy!" in captured.out

    def test_paths_validity(self):
        """Verify all default paths point to valid existing files and folders."""
        assert ASSETS_DIR.exists()
        assert (ASSETS_DIR / "demo_vid.mp4").exists()
        assert (ASSETS_DIR / "1.jpg").exists()
        assert DEFAULT_VIDEO_PATH.exists()
        assert DEFAULT_IMAGE_PATH.exists()
