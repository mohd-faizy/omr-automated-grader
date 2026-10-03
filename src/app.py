"""
Interactive Application for OMR MCQ Automated Grading.
Manages video/camera/image capture, HUD overlay, diagnostic dashboard,
keyboard controls, and scan saving.
"""
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from src.config import (
    ASSETS_DIR,
    CHOICE_LABELS,
    DEFAULT_IMAGE_PATH,
    DEFAULT_VIDEO_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SCANNED_DIR,
    STAGE_LABELS,
    get_sample_images,
)
from src.grader import OMRGrader, OMRResult
from src.utils import stack_images


class OMRApp:
    """
    Interactive GUI Application for real-time OMR grading.
    """

    def __init__(
        self,
        mode: str = "video",
        video_path: Optional[Path] = None,
        answer_key: Optional[List[int]] = None,
        save_dir: Optional[Path] = None,
    ):
        self.mode = mode.lower()
        self.video_path = Path(video_path) if video_path else DEFAULT_VIDEO_PATH
        self.sample_images = get_sample_images()
        self.sample_idx = 0

        # If a custom image was passed, prioritize it in sample_images
        if video_path:
            vp = Path(video_path)
            if vp.is_file() and vp.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                if vp in self.sample_images:
                    self.sample_idx = self.sample_images.index(vp)
                else:
                    self.sample_images.insert(0, vp)
                    self.sample_idx = 0
        self.save_dir = Path(save_dir) if save_dir else SCANNED_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.grader = OMRGrader(answer_key=answer_key)
        self.cap: Optional[cv2.VideoCapture] = None

        # Toast status notification state
        self.status_msg = ""
        self.status_timer = 0
        self.saved_scan_count = 0

        # Registered resizable windows
        self._windows_initialized = False

    def init_capture(self, target_mode: str) -> bool:
        """Initialize or switch media capture source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if target_mode == "video":
            if self.video_path and self.video_path.exists():
                self.cap = cv2.VideoCapture(str(self.video_path))
                print(f"[INFO] Playing demo video: {self.video_path.name}")
                return True
            print(f"[WARN] Demo video not found at: {self.video_path}")
            return False

        elif target_mode == "webcam":
            for cam_id in [0, 1]:
                temp_cap = cv2.VideoCapture(cam_id)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        self.cap = temp_cap
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        print(f"[INFO] Webcam connected on device index {cam_id}!")
                        return True
                    temp_cap.release()
            print("[WARN] No physical webcam detected. Reverting to Demo Video.")
            return False

        return True

    def _setup_windows(self):
        """Create resizable OpenCV windows."""
        if not self._windows_initialized:
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Final Result", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Result", 1200, 600)
            cv2.resizeWindow("Final Result", 700, 700)
            self._windows_initialized = True

    def draw_hud(self, frame: np.ndarray, result: OMRResult, active_name: str = "") -> None:
        """Render HUD headers, score badges, answer breakdown, and controls footer."""
        h, w = frame.shape[:2]

        # Semi-transparent top and bottom banner bars
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 25), cv2.FILLED)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 25), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Mode Badge
        mode_text = f"MODE: {self.mode.upper()}"
        if active_name:
            mode_text += f" [{active_name}]"
        cv2.rectangle(frame, (10, 8), (min(w - 10, 310), 36), (60, 50, 40), cv2.FILLED)
        cv2.putText(frame, mode_text, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2, cv2.LINE_AA)

        # Score & Answers Badge
        if result.success and result.score is not None:
            score_str = (
                f"SCORE: {int(result.score)}% ({result.correct_count}/{result.total_questions}) "
                f"Grade: {result.letter_grade}"
            )
            cv2.putText(frame, score_str, (325, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, result.grade_color, 2, cv2.LINE_AA)

            # Question by question answer breakdown
            ans_parts = []
            for i in range(self.grader.questions):
                st_ans = (
                    CHOICE_LABELS[result.student_answers[i]]
                    if i < len(result.student_answers)
                    else "?"
                )
                is_correct = result.grading[i] == 1 if i < len(result.grading) else False
                if is_correct:
                    ans_parts.append(f"Q{i+1}:{st_ans}")
                else:
                    corr_ans = (
                        CHOICE_LABELS[self.grader.answer_key[i]]
                        if i < len(self.grader.answer_key)
                        else "?"
                    )
                    ans_parts.append(f"Q{i+1}:{st_ans}(Ans:{corr_ans})")
            ans_str = " | ".join(ans_parts)
            cv2.putText(frame, ans_str, (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Scanning for OMR Paper...", (325, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Searching for MCQ question box and score box corners", (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Footer Keyboard Shortcuts
        footer = "[V] Cam/Video | [M] Image Mode | [N]/[P] Sheet (1-5) | [S] Save | [Q] Quit"
        cv2.putText(frame, footer, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

    def save_scan(self, frame: np.ndarray) -> str:
        """Save graded scan to Scanned/ directory."""
        filename = f"myImage_{self.saved_scan_count}.jpg"
        save_path = self.save_dir / filename
        cv2.imwrite(str(save_path), frame)
        self.saved_scan_count += 1
        print(f"[INFO] Graded scan saved to: {save_path}")
        return str(save_path)

    def run(self) -> None:
        """Start the real-time OMR processing and display loop."""
        self._setup_windows()

        # Initial source setup
        if self.mode in ["video", "webcam"]:
            if not self.init_capture(self.mode):
                self.mode = "image"

        print("=" * 65)
        print(" OPTICAL MARK RECOGNITION (OMR) MCQ AUTOMATED GRADING")
        print("=" * 65)
        print(f" [INFO] Initial Mode: {self.mode.upper()}")
        if self.video_path and self.video_path.exists():
            print(f" [INFO] Demo Video: {self.video_path.name}")
        print(f" [INFO] Benchmark Sheets: {len(self.sample_images)} available")
        key_str = ", ".join([f"Q{i+1}:{CHOICE_LABELS[k]}" for i, k in enumerate(self.grader.answer_key)])
        print(f" [INFO] Answer Key: {key_str}")
        print(" [INFO] Controls:")
        print("   [V] Toggle Demo Video / Live Webcam")
        print("   [M] Switch to / Toggle Sample Images Mode")
        print("   [N] / [P] Next / Previous Sample Sheet (1.jpg - 5.jpg)")
        print("   [S] Save Graded Exam Sheet Scan to Scanned/")
        print("   [Q] or [Esc] Quit")
        print("=" * 65)

        try:
            while True:
                active_name = ""
                raw_frame = None

                if self.mode in ["video", "webcam"]:
                    if self.cap is not None and self.cap.isOpened():
                        success, raw_frame = self.cap.read()
                        if not success and self.mode == "video":
                            # Seamless loop for demo video
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            success, raw_frame = self.cap.read()
                        if success and raw_frame is not None:
                            active_name = self.video_path.name if self.mode == "video" else "Live Cam"

                    if raw_frame is None:
                        raw_frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
                        cv2.putText(raw_frame, "No Video / Camera Stream", (80, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Static image mode
                    if self.sample_images:
                        curr_path = self.sample_images[self.sample_idx % len(self.sample_images)]
                        active_name = f"{curr_path.name} ({self.sample_idx + 1}/{len(self.sample_images)})"
                        raw_frame = cv2.imread(str(curr_path))
                    if raw_frame is None:
                        raw_frame = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
                        cv2.putText(raw_frame, "No Sample Images Found", (80, 350),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Execute headless OMR grading pipeline
                result = self.grader.process(raw_frame)
                img_final = result.annotated_frame.copy()

                # Draw HUD on final output frame
                self.draw_hud(img_final, result, active_name)

                # Diagnostic 2x4 Stage Grid
                stage_matrix = self.grader.get_stages_matrix(result)
                stacked_stages = stack_images(stage_matrix, scale=0.5, labels=STAGE_LABELS)

                # Render active toast notification if timer is set
                if self.status_timer > 0:
                    w_mid = int(stacked_stages.shape[1] / 2) - 240
                    cv2.rectangle(stacked_stages, (w_mid, 20), (w_mid + 480, 80), (0, 180, 0), cv2.FILLED)
                    cv2.putText(stacked_stages, self.status_msg, (w_mid + 20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(img_final, (120, 100), (580, 160), (0, 180, 0), cv2.FILLED)
                    cv2.putText(img_final, self.status_msg, (140, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 255, 255), 2, cv2.LINE_AA)
                    self.status_timer -= 1

                # Display GUI windows
                cv2.imshow("Result", stacked_stages)
                cv2.imshow("Final Result", img_final)

                # Event handling
                wait_ms = 35 if self.mode in ["video", "webcam"] else 40
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # 'q' or Esc
                    print("[INFO] Quitting application.")
                    break

                elif key in [ord('v'), ord('V')]:
                    # Toggle Video <-> Webcam
                    if self.mode == "webcam":
                        self.mode = "video"
                        print("[INFO] Switched to Demo Video Mode.")
                        self.init_capture(self.mode)
                    else:
                        self.mode = "webcam"
                        print("[INFO] Switching to Webcam Mode...")
                        if not self.init_capture(self.mode):
                            self.mode = "video"
                            self.init_capture(self.mode)
                            self.status_msg = "No Webcam Detected!"
                            self.status_timer = 30

                elif key in [ord('m'), ord('M')]:
                    # Toggle Image <-> Video
                    if self.mode == "image":
                        self.mode = "video"
                        print("[INFO] Switched to Demo Video Mode.")
                        self.init_capture(self.mode)
                    else:
                        self.mode = "image"
                        if self.cap is not None:
                            self.cap.release()
                            self.cap = None
                        curr_name = self.sample_images[self.sample_idx % len(self.sample_images)].name
                        print(f"[INFO] Switched to Sample Image Mode: {curr_name}")
                        self.status_msg = f"Sheet: {curr_name}"
                        self.status_timer = 25

                elif key in [ord('n'), ord('N')]:
                    # Next sample image
                    self.mode = "image"
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self.sample_idx = (self.sample_idx + 1) % max(1, len(self.sample_images))
                    curr_name = self.sample_images[self.sample_idx].name
                    print(f"[INFO] Next sample sheet [{self.sample_idx + 1}/{len(self.sample_images)}]: {curr_name}")
                    self.status_msg = f"Sheet: {curr_name}"
                    self.status_timer = 25

                elif key in [ord('p'), ord('P')]:
                    # Previous sample image
                    self.mode = "image"
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self.sample_idx = (self.sample_idx - 1) % max(1, len(self.sample_images))
                    curr_name = self.sample_images[self.sample_idx].name
                    print(f"[INFO] Prev sample sheet [{self.sample_idx + 1}/{len(self.sample_images)}]: {curr_name}")
                    self.status_msg = f"Sheet: {curr_name}"
                    self.status_timer = 25

                elif key in [ord('s'), ord('S')]:
                    # Save graded scan
                    saved_path = self.save_scan(result.annotated_frame if result.annotated_frame is not None else img_final)
                    self.status_msg = "Scan Saved Successfully!"
                    self.status_timer = 35

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release camera/video and destroy all OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
