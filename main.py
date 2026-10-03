"""
Main Entry Point for Optical Mark Recognition (OMR) MCQ Automated Grading.
"""
import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import OMRApp
from src.config import (
    ASSETS_DIR,
    BENCHMARK_EXPECTED,
    CHOICE_LABELS,
    DEFAULT_ANSWERS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    get_sample_images,
)
from src.grader import OMRGrader
import cv2


def parse_answer_key(key_str: str):
    """Parse answer key from either comma-separated letters ('B,C,A,C,E') or indices ('1,2,0,2,4')."""
    tokens = [t.strip().upper() for t in key_str.split(",") if t.strip()]
    label_map = {lbl: idx for idx, lbl in enumerate(CHOICE_LABELS)}
    parsed = []
    for token in tokens:
        if token in label_map:
            parsed.append(label_map[token])
        elif token.isdigit():
            val = int(token)
            if 0 <= val < len(CHOICE_LABELS):
                parsed.append(val)
            else:
                raise ValueError(f"Choice index {val} out of bounds (0-{len(CHOICE_LABELS)-1})")
        else:
            raise ValueError(f"Invalid answer key token: {token}")
    return parsed


def run_benchmark_test(answer_key=None):
    """Run headless verification across all 5 benchmark sheets and print results."""
    grader = OMRGrader(answer_key=answer_key)
    sample_images = get_sample_images()

    print("=" * 70)
    print(" RUNNING BENCHMARK EVALUATION ON SAMPLE EXAM SHEETS")
    print("=" * 70)
    print(f"{'Sheet':<12} | {'Detected Answers':<18} | {'Score':<8} | {'Grade':<6} | {'Status'}")
    print("-" * 70)

    all_passed = True
    for img_path in sample_images:
        if img_path.name not in BENCHMARK_EXPECTED:
            continue
        expected_answers, expected_score = BENCHMARK_EXPECTED[img_path.name]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"{img_path.name:<12} | FAILED TO LOAD IMAGE")
            all_passed = False
            continue

        result = grader.process(frame)
        if not result.success:
            print(f"{img_path.name:<12} | ALIGNMENT FAILED: {result.error_message}")
            all_passed = False
            continue

        ans_letters = "".join([CHOICE_LABELS[a] for a in result.student_answers])
        is_match = (result.score == expected_score and result.student_answers == expected_answers)
        status = "PASS [OK]" if is_match else "FAIL [MISMATCH]"
        if not is_match:
            all_passed = False

        print(
            f"{img_path.name:<12} | {ans_letters} ({result.student_answers}) "
            f"| {int(result.score)}%    | {result.letter_grade:<6} | {status}"
        )

    print("-" * 70)
    if all_passed:
        print("[SUCCESS] All benchmark sheets evaluated with 100% accuracy!")
    else:
        print("[WARNING] One or more benchmark evaluations failed.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Optical Mark Recognition (OMR) MCQ Automated Grading System"
    )
    parser.add_argument(
        "pos_mode",
        nargs="?",
        default=None,
        choices=["video", "image", "webcam"],
        help="Operating mode (video, image, or webcam)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["video", "image", "webcam"],
        default=None,
        help="Operating mode (video, image, or webcam)"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Custom video or image file path"
    )
    parser.add_argument(
        "--key",
        "-k",
        type=str,
        default=None,
        help="Answer key as comma-separated letters or indices (e.g. 'B,C,A,C,E' or '1,2,0,2,4')"
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        action="store_true",
        help="Run headless benchmark tests on benchmark sheets without opening GUI windows"
    )

    args = parser.parse_args()

    # Parse answer key if provided
    answer_key = DEFAULT_ANSWERS
    if args.key:
        try:
            answer_key = parse_answer_key(args.key)
            print(f"[INFO] Custom answer key loaded: {answer_key}")
        except Exception as e:
            print(f"[ERROR] Failed to parse custom answer key: {e}")
            sys.exit(1)

    # Headless benchmark test mode
    if args.benchmark:
        run_benchmark_test(answer_key)
        return

    # Determine mode
    mode = args.mode or args.pos_mode
    if not mode:
        if args.source and Path(args.source).suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            mode = "image"
        else:
            mode = "video"

    video_path = Path(args.source) if args.source else DEFAULT_VIDEO_PATH

    # Launch GUI application
    app = OMRApp(mode=mode, video_path=video_path, answer_key=answer_key)
    app.run()


if __name__ == "__main__":
    main()
