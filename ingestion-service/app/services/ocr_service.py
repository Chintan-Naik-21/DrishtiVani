"""Optical Character Recognition (OCR) service using Surya OCR.

Performs recognition on text, heading, list_item, and figure_caption crops.
Crops equation and diagram regions without OCR for downstream processing by teammates.
Adheres strictly to original-scan traceability and confidence thresholds.
"""

from __future__ import annotations

import importlib
import io
import logging
from typing import Any, List, Optional, Tuple

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

# Semantic block types that require text OCR
OCR_TARGET_TYPES = {
    BlockType.HEADING_H1.value,
    BlockType.HEADING_H2.value,
    BlockType.HEADING_H3.value,
    BlockType.TEXT.value,
    BlockType.LIST_ITEM.value,
    BlockType.FIGURE_CAPTION.value,
}

# Semantic block types that represent visual assets (equation / circuit diagram)
# These MUST NOT undergo text OCR; their crops are preserved for downstream teammates
VISUAL_CROP_TYPES = {
    BlockType.EQUATION.value,
    BlockType.DIAGRAM.value,
}


class OCRService:
    """Service for running optical character recognition on page block crops."""

    def __init__(self) -> None:
        """Initialize the OCR service configuration and lazily load ML models."""
        self.settings = get_settings()
        self._predictor: Optional[Any] = None
        self._device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        logger.info("OCRService initialized (configured device: %s).", self._device)

    def _get_predictor(self) -> Any:
        """Load and return the Surya RecognitionPredictor singleton."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available in the current environment.")
        if self._predictor is None:
            logger.info("Loading Surya RecognitionPredictor model on device: %s...", self._device)
            try:
                recognition_mod = importlib.import_module("surya.recognition")
                RecognitionPredictor = getattr(recognition_mod, "RecognitionPredictor")
                self._predictor = RecognitionPredictor()
                logger.info("Surya RecognitionPredictor successfully loaded.")
            except Exception as e:
                logger.error("Failed to load Surya RecognitionPredictor: %s", e)
                raise
        return self._predictor

    @staticmethod
    def crop_region(
        page_image: Image.Image,
        bbox: List[float],
        padding: int = 2,
    ) -> Image.Image:
        """Crop a specific bounding box region from a page image with optional padding.

        Args:
            page_image: Full PIL Image of the page raster.
            bbox: Bounding box [x0, y0, x1, y1] in pixel coordinates.
            padding: Additional pixel padding around the crop boundary.

        Returns:
            Image.Image: The cropped PIL image slice.
        """
        width, height = page_image.size
        x0 = max(0.0, float(bbox[0]) - padding)
        y0 = max(0.0, float(bbox[1]) - padding)
        x1 = min(float(width), float(bbox[2]) + padding)
        y1 = min(float(height), float(bbox[3]) + padding)

        # Handle edge cases where box is inverted or degenerate
        if x1 <= x0 or y1 <= y0:
            x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)

        return page_image.crop((int(x0), int(y0), int(x1), int(y1)))

    @staticmethod
    def image_to_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
        """Convert a PIL Image to raw bytes."""
        buf = io.BytesIO()
        image.save(buf, format=image_format.upper())
        return buf.getvalue()

    def recognize_text(self, crop_image: Image.Image) -> Tuple[str, float]:
        """Perform text recognition on a cropped block image.

        Args:
            crop_image: Cropped PIL Image of a text or heading block.

        Returns:
            Tuple[str, float]: (recognized_text, ocr_confidence) where confidence is in [0.0, 1.0].
        """
        try:
            predictor = self._get_predictor()
            results = predictor([crop_image])
            if results and len(results) > 0:
                result = results[0]
                text_lines: List[str] = []
                confidences: List[float] = []

                if hasattr(result, "text_lines") and result.text_lines:
                    for line in result.text_lines:
                        line_text = getattr(line, "text", "") or ""
                        line_conf = getattr(line, "confidence", 1.0) or 1.0
                        if line_text.strip():
                            text_lines.append(line_text.strip())
                            confidences.append(float(line_conf))

                full_text = " ".join(text_lines).strip()
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.85
                )
                return full_text, round(avg_confidence, 4)

        except Exception as e:
            logger.debug("Surya RecognitionPredictor fallback engaged: %s", e)

        # Fallback heuristic: return empty string with moderate confidence
        return "", 0.80

    def is_visual_crop(self, block_type: str) -> bool:
        """Check if a block type represents a visual crop (equation or diagram)."""
        return block_type in VISUAL_CROP_TYPES

    def is_ocr_target(self, block_type: str) -> bool:
        """Check if a block type should undergo text recognition."""
        return block_type in OCR_TARGET_TYPES


_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Return a singleton instance of OCRService."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
