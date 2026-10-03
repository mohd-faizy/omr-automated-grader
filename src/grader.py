"""
Headless OMR Grading Engine.
Handles document alignment, perspective warping, bubble segmentation,
scoring against answer keys, and augmented reality visual projection.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    BINARY_INV_THRESH,
    CANNY_HIGH_THRESH,
    CANNY_LOW_THRESH,
    CHOICES_COUNT,
    DEFAULT_ANSWERS,
    GAUSSIAN_KERNEL,
    GAUSSIAN_SIGMA,
    GRADE_BOX_HEIGHT,
    GRADE_BOX_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    QUESTIONS_COUNT,
    get_grade_info,
)
from src.utils import draw_grid, get_corner_points, rect_contour, reorder, show_answers, split_boxes


@dataclass
class OMRResult:
    """Structured result returned by OMRGrader.process()."""
    success: bool
    score: Optional[float] = None
    letter_grade: Optional[str] = None
    grade_color: Tuple[int, int, int] = (180, 180, 180)
    student_answers: List[int] = field(default_factory=list)
    grading: List[int] = field(default_factory=list)
    raw_pixel_counts: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    grade_card: Optional[np.ndarray] = None
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def correct_count(self) -> int:
        return sum(self.grading) if self.grading else 0

    @property
    def total_questions(self) -> int:
        return len(self.grading) if self.grading else 0

    def get_summary_text(self) -> str:
        """Return formatted human-readable summary of the grading result."""
        if not self.success or self.score is None:
            return "Scanning / Aligning OMR Sheet..."
        letter = self.letter_grade or ""
        return (
            f"Score: {int(self.score)}% ({self.correct_count}/{self.total_questions}) "
            f"Grade: {letter}"
        )


class OMRGrader:
    """
    Automated Optical Mark Recognition (OMR) Grader.
    Provides headless, testable grading pipeline from raw images to structured results.
    """

    def __init__(
        self,
        answer_key: Optional[List[int]] = None,
        questions: int = QUESTIONS_COUNT,
        choices: int = CHOICES_COUNT,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
    ):
        self.answer_key = list(answer_key if answer_key is not None else DEFAULT_ANSWERS)
        self.questions = questions
        self.choices = choices
        self.width = width
        self.height = height

    @staticmethod
    def create_grade_card(
        score: Optional[float] = None,
        letter: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 200, 0),
        width: int = GRADE_BOX_WIDTH,
        height: int = GRADE_BOX_HEIGHT,
    ) -> np.ndarray:
        """Render a bold, high-contrast visual grade card for physical projection."""
        card = np.full((height, width, 3), 250, dtype=np.uint8)

        if score is not None:
            # Outer color border
            cv2.rectangle(card, (3, 3), (width - 4, height - 4), color, 4)

            # Percentage score with dark outline for contrast
            score_text = f"{int(score)}%"
            (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.9, 4)
            tx = (width - tw) // 2
            ty = int(height * 0.52)
            cv2.putText(card, score_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.9, (20, 20, 20), 7, cv2.LINE_AA)
            cv2.putText(card, score_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 1.9, color, 3, cv2.LINE_AA)

            # Letter grade label
            grade_text = f"GRADE: {letter}" if letter else "GRADE"
            (gw, _), _ = cv2.getTextSize(grade_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
            gx = (width - gw) // 2
            gy = int(height * 0.83)
            cv2.putText(card, grade_text, (gx, gy), cv2.FONT_HERSHEY_DUPLEX, 0.75, (40, 40, 40), 2, cv2.LINE_AA)
        else:
            # Placeholder card shown when scanning or aligning
            cv2.rectangle(card, (3, 3), (width - 4, height - 4), (180, 180, 180), 3)
            msg1 = "ALIGNING..."
            (mw1, _), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
            cv2.putText(card, msg1, ((width - mw1) // 2, int(height * 0.43)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 80, 80), 2, cv2.LINE_AA)
            msg2 = "GRADE BOX"
            (mw2, _), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.putText(card, msg2, ((width - mw2) // 2, int(height * 0.77)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)

        return card

    def process(self, frame: np.ndarray) -> OMRResult:
        """
        Execute full OMR grading pipeline on a single frame or image.
        Returns OMRResult with scores, student selections, and visual diagnostic stages.
        """
        # Resize input frame to standard dimensions
        img = cv2.resize(frame, (self.width, self.height))
        img_final = img.copy()
        img_blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Preprocessing: Grayscale -> Gaussian Blur -> Canny Edges
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
        img_canny = cv2.Canny(img_blur, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)

        # Find external contours
        img_contours = img.copy()
        img_big_contour = img.copy()
        contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 10)

        # Base fallback stages
        stages = {
            "original": img,
            "gray": img_gray,
            "edges": img_canny,
            "contours": img_contours,
            "biggest_contour": img_blank,
            "threshold": img_blank,
            "warped": img_blank,
            "final": img_final,
        }

        rect_con = rect_contour(contours)

        # Need at least question bubble grid and grade box (2 rectangular contours)
        if len(rect_con) < 2:
            return OMRResult(
                success=False,
                annotated_frame=img_final,
                grade_card=self.create_grade_card(None),
                stages=stages,
                error_message="Less than 2 rectangular contours detected."
            )

        biggest_points = get_corner_points(rect_con[0])
        grade_points = get_corner_points(rect_con[1])

        # Validate that both contours have exactly 4 corner points
        if biggest_points.size != 8 or grade_points.size != 8:
            return OMRResult(
                success=False,
                annotated_frame=img_final,
                grade_card=self.create_grade_card(None),
                stages=stages,
                error_message="Contours do not have 4 corner points."
            )

        try:
            # 1. Warp Question Bubble Grid
            biggest_points = reorder(biggest_points)
            cv2.drawContours(img_big_contour, biggest_points, -1, (0, 255, 0), 20)

            pts1 = np.float32(biggest_points)
            pts2 = np.float32([[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]])
            matrix_warp = cv2.getPerspectiveTransform(pts1, pts2)
            img_warp_colored = cv2.warpPerspective(img, matrix_warp, (self.width, self.height))

            # 2. Warp Grade Box
            grade_points = reorder(grade_points)
            cv2.drawContours(img_big_contour, grade_points, -1, (255, 0, 0), 20)

            pts_g1 = np.float32(grade_points)
            pts_g2 = np.float32([[0, 0], [GRADE_BOX_WIDTH, 0], [0, GRADE_BOX_HEIGHT], [GRADE_BOX_WIDTH, GRADE_BOX_HEIGHT]])
            matrix_grade = cv2.getPerspectiveTransform(pts_g1, pts_g2)

            # 3. Threshold Warped Question Grid
            img_warp_gray = cv2.cvtColor(img_warp_colored, cv2.COLOR_BGR2GRAY)
            img_thresh = cv2.threshold(img_warp_gray, BINARY_INV_THRESH, 255, cv2.THRESH_BINARY_INV)[1]

            # 4. Bubble Segmentation & Non-Zero Pixel Density Counting
            boxes = split_boxes(img_thresh, self.questions, self.choices)
            pixel_matrix = np.zeros((self.questions, self.choices), dtype=np.int32)
            r = 0
            c = 0
            for box_img in boxes:
                pixel_matrix[r][c] = cv2.countNonZero(box_img)
                c += 1
                if c == self.choices:
                    c = 0
                    r += 1

            # 5. Extract Student Selections
            student_answers = []
            for q_idx in range(self.questions):
                row_pixels = pixel_matrix[q_idx]
                selected_choice = int(np.argmax(row_pixels))
                student_answers.append(selected_choice)

            # 6. Grade Against Answer Key
            grading = [1 if self.answer_key[i] == student_answers[i] else 0 for i in range(self.questions)]
            score = (sum(grading) / float(self.questions)) * 100.0
            letter, grade_color = get_grade_info(score)

            # 7. Render Grade Box Card
            grade_card = self.create_grade_card(score, letter, grade_color)

            # 8. Draw Answers Overlay on Warped Image
            show_answers(img_warp_colored, student_answers, grading, self.answer_key, self.questions, self.choices)
            draw_grid(img_warp_colored, self.questions, self.choices)

            # 9. Inverse Perspective Projection for Question Answers
            img_drawings = np.zeros_like(img_warp_colored)
            show_answers(img_drawings, student_answers, grading, self.answer_key, self.questions, self.choices)
            inv_matrix_answers = cv2.getPerspectiveTransform(pts2, pts1)
            img_inv_warp = cv2.warpPerspective(img_drawings, inv_matrix_answers, (self.width, self.height))

            # 10. Inverse Perspective Projection for Grade Card onto Physical Exam Box
            inv_matrix_grade = cv2.getPerspectiveTransform(pts_g2, pts_g1)
            img_inv_grade = cv2.warpPerspective(grade_card, inv_matrix_grade, (self.width, self.height))
            mask_grade = cv2.warpPerspective(
                np.ones((GRADE_BOX_HEIGHT, GRADE_BOX_WIDTH), dtype=np.uint8) * 255,
                inv_matrix_grade,
                (self.width, self.height)
            )

            # 11. Blend Projected Overlays onto Original Frame
            img_final = cv2.addWeighted(img_final, 1, img_inv_warp, 1, 0)
            img_final[mask_grade > 128] = img_inv_grade[mask_grade > 128]

            # Populate diagnostic stages
            stages["biggest_contour"] = img_big_contour
            stages["threshold"] = img_thresh
            stages["warped"] = img_warp_colored
            stages["final"] = img_final

            return OMRResult(
                success=True,
                score=score,
                letter_grade=letter,
                grade_color=grade_color,
                student_answers=student_answers,
                grading=grading,
                raw_pixel_counts=pixel_matrix,
                annotated_frame=img_final,
                grade_card=grade_card,
                stages=stages
            )

        except Exception as exc:
            return OMRResult(
                success=False,
                annotated_frame=img_final,
                grade_card=self.create_grade_card(None),
                stages=stages,
                error_message=f"Processing exception: {exc}"
            )

    @staticmethod
    def get_stages_matrix(result: OMRResult) -> List[List[np.ndarray]]:
        """Return 2x4 image array suitable for stack_images diagnostic dashboard."""
        return [
            [
                result.stages.get("original"),
                result.stages.get("gray"),
                result.stages.get("edges"),
                result.stages.get("contours"),
            ],
            [
                result.stages.get("biggest_contour"),
                result.stages.get("threshold"),
                result.stages.get("warped"),
                result.stages.get("final"),
            ]
        ]
