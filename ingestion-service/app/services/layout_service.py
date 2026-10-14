"""Layout parsing service using Surya layout detection.

Detects bounding boxes, classifies regions (headings, text, equations, tables, diagrams),
infers heading hierarchy (H1-H3), and determines logical reading order.
Adheres strictly to the original-scan traceability requirement.
"""

from __future__ import annotations

import gc
import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    torch = importlib.import_module("torch")
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from PIL import Image

from app.config import get_settings
from app.db.models import BlockType

logger = logging.getLogger(__name__)


# Standard mapping from Surya layout labels to DrishtiVani semantic BlockType
SURYA_TO_BLOCK_TYPE: Dict[str, str] = {
    # Generic headers to be hierarchically resolved (H1-H3)
    "sectionheader": "heading",
    "section-header": "heading",
    "title": BlockType.HEADING_H1.value,
    # Body text
    "text": BlockType.TEXT.value,
    "footnote": BlockType.TEXT.value,
    "code": BlockType.TEXT.value,
    "code-block": BlockType.TEXT.value,
    # Mathematical and circuit equations
    "equation": BlockType.EQUATION.value,
    "equation-block": BlockType.EQUATION.value,
    "formula": BlockType.EQUATION.value,
    # Tabular data
    "table": BlockType.TABLE.value,
    # Visual diagrams, schematics, and pictures
    "picture": BlockType.DIAGRAM.value,
    "figure": BlockType.DIAGRAM.value,
    "diagram": BlockType.DIAGRAM.value,
    "image": BlockType.DIAGRAM.value,
    "complex-block": BlockType.DIAGRAM.value,
    # Captions
    "caption": BlockType.FIGURE_CAPTION.value,
    "figure-caption": BlockType.FIGURE_CAPTION.value,
    # Lists
    "listgroup": BlockType.LIST_ITEM.value,
    "list-group": BlockType.LIST_ITEM.value,
    "listitem": BlockType.LIST_ITEM.value,
    "list-item": BlockType.LIST_ITEM.value,
    # Headers, footers, and other peripheral regions
    "pageheader": BlockType.OTHER.value,
    "page-header": BlockType.OTHER.value,
    "pagefooter": BlockType.OTHER.value,
    "page-footer": BlockType.OTHER.value,
    "blankpage": BlockType.OTHER.value,
    "blank-page": BlockType.OTHER.value,
    "form": BlockType.OTHER.value,
    "tableofcontents": BlockType.OTHER.value,
    "table-of-contents": BlockType.OTHER.value,
    "bibliography": BlockType.OTHER.value,
}


@dataclass
class DetectedBlock:
    """Intermediate representation of a detected layout block on a page."""
    block_type: str
    reading_order: int
    bbox: List[float]  # [x0, y0, x1, y1] in pixel coordinates of page image
    layout_confidence: float
    needs_review: bool
    extra: Optional[Dict[str, Any]] = None


class LayoutService:
    """Service wrapping Surya layout detection and region classification."""

    def __init__(self) -> None:
        """Initialize the layout service configuration and lazily load ML models."""
        self.settings = get_settings()
        self._predictor: Optional[Any] = None
        self._device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        logger.info("LayoutService initialized (configured device: %s).", self._device)

    def _get_predictor(self) -> Any:
        """Load and return the Surya FastLayoutPredictor singleton."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available in the current environment.")
        if self._predictor is None:
            logger.info("Loading Surya FastLayoutPredictor model on device: %s...", self._device)
            try:
                fast_layout_mod = importlib.import_module("surya.fast_layout")
                FastLayoutPredictor = getattr(fast_layout_mod, "FastLayoutPredictor")
                self._predictor = FastLayoutPredictor(use_order=True)
                logger.info("Surya FastLayoutPredictor successfully loaded.")
            except Exception as e:
                logger.error("Failed to load Surya FastLayoutPredictor: %s", e)
                raise
        return self._predictor

    def detect_layout(
        self,
        image: Image.Image,
        width_px: int,
        height_px: int,
    ) -> List[DetectedBlock]:
        """Detect and classify all layout regions on a page image.

        Processes the image through Surya layout detection, normalizes bounding
        boxes to pixel space, maps model labels to semantic BlockType enum values,
        infers heading hierarchy (H1-H3), and determines logical reading order.

        Args:
            image: PIL Image object of the rendered page.
            width_px: Pixel width of the page image.
            height_px: Pixel height of the page image.

        Returns:
            List[DetectedBlock]: Detected blocks sorted in logical reading order.
        """
        raw_boxes: List[Dict[str, Any]] = []

        try:
            predictor = self._get_predictor()
            results = predictor([image], use_order=True)

            if results and not results[0].error and results[0].bboxes:
                for b in results[0].bboxes:
                    # Bounding box coordinates [x0, y0, x1, y1]
                    if hasattr(b, "bbox") and b.bbox:
                        box = list(b.bbox)
                    elif hasattr(b, "polygon") and b.polygon:
                        poly = b.polygon
                        if isinstance(poly[0], (list, tuple)):
                            xs = [pt[0] for pt in poly]
                            ys = [pt[1] for pt in poly]
                            box = [min(xs), min(ys), max(xs), max(ys)]
                        else:
                            box = list(poly[:4])
                    else:
                        continue

                    raw_boxes.append({
                        "bbox": box,
                        "label": getattr(b, "label", "Text") or "Text",
                        "confidence": float(getattr(b, "confidence", 1.0) or 1.0),
                        "position": getattr(b, "position", 0) or 0,
                    })

        except Exception as e:
            logger.warning(
                "Surya layout predictor failed for page (%s). Falling back to heuristic region extraction: %s",
                f"{width_px}x{height_px}",
                e,
            )
            raw_boxes = self._heuristic_fallback_layout(image, width_px, height_px)

        if not raw_boxes:
            logger.info("No layout boxes detected by Surya; generating default full-page text block.")
            raw_boxes = [{
                "bbox": [0.0, 0.0, float(width_px), float(height_px)],
                "label": "Text",
                "confidence": 0.50,
                "position": 0,
            }]

        # 1. Clamp bounding boxes to image dimensions and validate
        valid_boxes: List[Dict[str, Any]] = []
        for item in raw_boxes:
            box = item["bbox"]
            x0 = max(0.0, min(float(box[0]), float(width_px)))
            y0 = max(0.0, min(float(box[1]), float(height_px)))
            x1 = max(0.0, min(float(box[2]), float(width_px)))
            y1 = max(0.0, min(float(box[3]), float(height_px)))

            # Discard degenerated boxes
            if x1 <= x0 or y1 <= y0:
                continue

            valid_boxes.append({
                "bbox": [x0, y0, x1, y1],
                "label": item["label"],
                "confidence": item["confidence"],
                "position": item.get("position", 0),
            })

        if not valid_boxes:
            valid_boxes = [{
                "bbox": [0.0, 0.0, float(width_px), float(height_px)],
                "label": "Text",
                "confidence": 0.50,
                "position": 0,
            }]

        # 2. Sort into logical reading order
        sorted_boxes = self._sort_reading_order(valid_boxes, width_px, height_px)

        # 3. Infer heading hierarchy (H1-H3) and map to semantic BlockType
        detected_blocks = self._classify_and_infer_headings(sorted_boxes, height_px)

        # Free memory between pages
        gc.collect()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return detected_blocks

    def _sort_reading_order(
        self,
        boxes: List[Dict[str, Any]],
        width_px: int,
        height_px: int,
    ) -> List[Dict[str, Any]]:
        """Sort detected bounding boxes into natural top-to-bottom reading order.

        Handles multi-column layouts by detecting if content divides into distinct
        columns, otherwise orders primarily by vertical position with horizontal tolerance.
        """
        # If boxes already have valid sequential positions from model ordering
        if all("position" in b and b["position"] is not None for b in boxes):
            return sorted(boxes, key=lambda b: b["position"])

        mid_x = width_px / 2.0
        # Check if page is predominantly two-column:
        # Most boxes reside fully on left or right half
        left_boxes = [b for b in boxes if b["bbox"][2] <= mid_x * 1.1]
        right_boxes = [b for b in boxes if b["bbox"][0] >= mid_x * 0.9]

        if len(left_boxes) + len(right_boxes) >= len(boxes) * 0.75 and len(left_boxes) > 1 and len(right_boxes) > 1:
            # Multi-column sort: left column top-to-bottom, then right column
            left_sorted = sorted(left_boxes, key=lambda b: b["bbox"][1])
            right_sorted = sorted(right_boxes, key=lambda b: b["bbox"][1])
            # Full-width blocks (banners/titles) sorted by Y
            full_width = [b for b in boxes if b not in left_boxes and b not in right_boxes]
            all_sorted = sorted(full_width + left_sorted + right_sorted, key=lambda b: (round(b["bbox"][1] / 30.0), b["bbox"][0]))
            return all_sorted

        # Standard top-to-bottom, left-to-right reading order with 20px vertical tolerance
        return sorted(boxes, key=lambda b: (round(b["bbox"][1] / 20.0), b["bbox"][0]))

    def _classify_and_infer_headings(
        self,
        boxes: List[Dict[str, Any]],
        height_px: int,
    ) -> List[DetectedBlock]:
        """Classify each box into BlockType and infer heading levels (H1-H3).

        When Surya produces a generic 'section header' label:
        - Box height (font size indicator) and page position are analyzed.
        - Relative box heights are tiered into H1 (document/chapter title),
          H2 (section header), and H3 (subsection header).
        """
        threshold = self.settings.CONFIDENCE_THRESHOLD

        # Identify candidate heading boxes
        heading_candidates: List[Tuple[int, float, float]] = []  # (index, height, y0)
        body_text_heights: List[float] = []

        for idx, item in enumerate(boxes):
            norm_label = str(item["label"]).strip().lower().replace("_", "").replace("-", "")
            mapped_type = SURYA_TO_BLOCK_TYPE.get(norm_label, BlockType.OTHER.value)
            box_h = item["bbox"][3] - item["bbox"][1]

            if mapped_type == "heading":
                heading_candidates.append((idx, box_h, item["bbox"][1]))
            elif mapped_type == BlockType.TEXT.value:
                body_text_heights.append(box_h)

        # Compute median body text height for scale comparison
        median_body_h = (
            sorted(body_text_heights)[len(body_text_heights) // 2]
            if body_text_heights
            else (height_px * 0.02)
        )

        # Determine heading tiers
        heading_levels: Dict[int, int] = {}
        if heading_candidates:
            # Sort unique heights descending
            unique_heights = sorted({h for _, h, _ in heading_candidates}, reverse=True)

            if len(unique_heights) >= 3:
                # 3 or more distinct heading sizes: map to H1, H2, H3
                h1_threshold = unique_heights[0] * 0.90
                h2_threshold = unique_heights[1] * 0.85
                for idx, h, y0 in heading_candidates:
                    if h >= h1_threshold:
                        heading_levels[idx] = 1
                    elif h >= h2_threshold:
                        heading_levels[idx] = 2
                    else:
                        heading_levels[idx] = 3
            elif len(unique_heights) == 2:
                # 2 distinct heading sizes: map to H1 and H2
                for idx, h, y0 in heading_candidates:
                    if h >= unique_heights[0] * 0.90:
                        heading_levels[idx] = 1
                    else:
                        heading_levels[idx] = 2
            else:
                # Only 1 heading size: check relative height and vertical position
                for idx, h, y0 in heading_candidates:
                    if h >= median_body_h * 1.6 or y0 < (height_px * 0.25):
                        heading_levels[idx] = 1
                    elif h >= median_body_h * 1.2:
                        heading_levels[idx] = 2
                    else:
                        heading_levels[idx] = 3

        # Construct final DetectedBlock list
        detected_blocks: List[DetectedBlock] = []
        for order_idx, item in enumerate(boxes):
            norm_label = str(item["label"]).strip().lower().replace("_", "").replace("-", "")
            base_type = SURYA_TO_BLOCK_TYPE.get(norm_label, BlockType.OTHER.value)
            conf = float(item["confidence"])
            extra_payload: Optional[Dict[str, Any]] = None

            if base_type == "heading":
                level = heading_levels.get(order_idx, 2)
                block_type = f"heading_h{level}"
                extra_payload = {"heading_level": level}
            elif base_type == BlockType.HEADING_H1.value:
                block_type = BlockType.HEADING_H1.value
                extra_payload = {"heading_level": 1}
            else:
                block_type = base_type

            needs_review = conf < threshold

            detected_blocks.append(
                DetectedBlock(
                    block_type=block_type,
                    reading_order=order_idx,
                    bbox=item["bbox"],
                    layout_confidence=conf,
                    needs_review=needs_review,
                    extra=extra_payload,
                )
            )

        return detected_blocks

    def _heuristic_fallback_layout(
        self,
        image: Image.Image,
        width_px: int,
        height_px: int,
    ) -> List[Dict[str, Any]]:
        """Heuristic fallback layout detection using image brightness/contrasts."""
        boxes: List[Dict[str, Any]] = []
        try:
            import numpy as np
            gray = image.convert("L")
            arr = np.array(gray)

            # Detect rows with substantial dark pixels (text lines)
            dark_pixels_per_row = np.sum(arr < 200, axis=1)
            row_threshold = width_px * 0.02
            is_content_row = dark_pixels_per_row > row_threshold

            # Group continuous rows into vertical regions
            regions: List[Tuple[int, int]] = []
            in_region = False
            start_y = 0

            for y, has_content in enumerate(is_content_row):
                if has_content and not in_region:
                    in_region = True
                    start_y = y
                elif not has_content and in_region:
                    in_region = False
                    if (y - start_y) >= 10:  # Min 10px height
                        regions.append((start_y, y))

            if in_region and (height_px - start_y) >= 10:
                regions.append((start_y, height_px))

            for y0, y1 in regions:
                region_arr = arr[y0:y1, :]
                dark_cols = np.where(np.sum(region_arr < 200, axis=0) > 2)[0]
                if len(dark_cols) > 0:
                    x0 = float(max(0, dark_cols[0] - 5))
                    x1 = float(min(width_px, dark_cols[-1] + 5))
                    # If near top and tall, flag as potential header
                    label = "SectionHeader" if y0 < height_px * 0.2 and (y1 - y0) > 30 else "Text"
                    boxes.append({
                        "bbox": [x0, float(y0), x1, float(y1)],
                        "label": label,
                        "confidence": 0.65,
                        "position": len(boxes),
                    })
        except Exception as e:
            logger.warning("Heuristic fallback failed: %s", e)

        return boxes


_layout_service: Optional[LayoutService] = None


def get_layout_service() -> LayoutService:
    """Return a singleton instance of LayoutService."""
    global _layout_service
    if _layout_service is None:
        _layout_service = LayoutService()
    return _layout_service
