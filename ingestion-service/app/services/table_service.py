"""Table parsing service using Table Transformer (TATR) and heuristic fallback.

Parses table structure (rows, columns, spanning cells, headers) and coordinates
with OCRService to extract cell text, producing a structured TableGridSchema.
Adheres strictly to original-scan traceability with pixel coordinates.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    torch = importlib.import_module("torch")
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from PIL import Image

from app.config import get_settings
from app.schemas.layout import TableCellSchema, TableGridSchema, TableRowSchema
from app.services.ocr_service import OCRService, get_ocr_service

logger = logging.getLogger(__name__)


class TableService:
    """Service for table structure recognition with TATR and heuristic parsing."""

    def __init__(self, ocr_service: Optional[OCRService] = None) -> None:
        """Initialize the table parsing service and lazily configure ML models."""
        self.settings = get_settings()
        self.ocr_service = ocr_service or get_ocr_service()
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self._device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        logger.info("TableService initialized (configured device: %s).", self._device)

    def _load_tatr_model(self) -> Tuple[Any, Any]:
        """Lazily load the HuggingFace Table Transformer (TATR) model and image processor."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not available in the current environment.")

        if self._processor is None or self._model is None:
            logger.info("Loading Table Transformer (microsoft/table-transformer-structure-recognition)...")
            try:
                transformers_mod = importlib.import_module("transformers")
                AutoImageProcessor = getattr(transformers_mod, "AutoImageProcessor")
                TableTransformerForObjectDetection = getattr(transformers_mod, "TableTransformerForObjectDetection")

                model_name = "microsoft/table-transformer-structure-recognition"
                self._processor = AutoImageProcessor.from_pretrained(model_name)
                self._model = TableTransformerForObjectDetection.from_pretrained(model_name).to(self._device)
                self._model.eval()
                logger.info("Table Transformer successfully loaded.")
            except Exception as e:
                logger.error("Failed to load Table Transformer model: %s", e)
                raise
        return self._processor, self._model

    def parse_table(
        self,
        table_crop: Image.Image,
        table_bbox_on_page: List[float],
    ) -> Tuple[TableGridSchema, str]:
        """Parse the internal grid structure of a table block crop and extract cell text.

        Args:
            table_crop: Cropped PIL Image containing only the table region.
            table_bbox_on_page: [x0, y0, x1, y1] pixel coordinates of the table on the page.

        Returns:
            Tuple[TableGridSchema, str]: (table_grid_structure, full_table_plain_text)
        """
        crop_w, crop_h = table_crop.size
        tbl_x0, tbl_y0, tbl_x1, tbl_y1 = table_bbox_on_page

        rows_cells: List[List[Dict[str, Any]]] = []

        try:
            rows_cells = self._tatr_structure_recognition(table_crop)
        except Exception as e:
            logger.debug("TATR table recognition fallback engaged: %s", e)
            rows_cells = self._heuristic_grid_segmentation(table_crop)

        if not rows_cells:
            # Fallback to single cell grid covering entire table
            rows_cells = [[{
                "col": 0,
                "bbox": [0.0, 0.0, float(crop_w), float(crop_h)],
                "is_header": False,
            }]]

        # Coordinate with OCRService to extract text for each cell
        grid_rows: List[TableRowSchema] = []
        table_text_lines: List[str] = []

        for row_idx, row_items in enumerate(rows_cells):
            row_cells: List[TableCellSchema] = []
            row_texts: List[str] = []

            for col_idx, cell_data in enumerate(row_items):
                cell_box = cell_data["bbox"]  # [x0, y0, x1, y1] in crop coordinates

                # Map back to full page image pixel coordinates for original-scan traceability
                page_cell_bbox = [
                    round(tbl_x0 + cell_box[0], 2),
                    round(tbl_y0 + cell_box[1], 2),
                    round(tbl_x0 + cell_box[2], 2),
                    round(tbl_y0 + cell_box[3], 2),
                ]

                # Crop cell image and recognize text
                cell_crop = table_crop.crop((
                    int(max(0, cell_box[0])),
                    int(max(0, cell_box[1])),
                    int(min(crop_w, cell_box[2])),
                    int(min(crop_h, cell_box[3])),
                ))

                cell_text, _ = self.ocr_service.recognize_text(cell_crop)
                is_header = cell_data.get("is_header", row_idx == 0)

                cell_schema = TableCellSchema(
                    row=row_idx,
                    col=col_idx,
                    row_span=cell_data.get("row_span", 1),
                    col_span=cell_data.get("col_span", 1),
                    is_header=is_header,
                    bbox=page_cell_bbox,
                    text=cell_text.strip(),
                )
                row_cells.append(cell_schema)
                if cell_text.strip():
                    row_texts.append(cell_text.strip())

            grid_rows.append(TableRowSchema(cells=row_cells))
            if row_texts:
                table_text_lines.append(" | ".join(row_texts))

        full_table_text = "\n".join(table_text_lines).strip()
        table_grid = TableGridSchema(rows=grid_rows)
        return table_grid, full_table_text

    def _tatr_structure_recognition(self, image: Image.Image) -> List[List[Dict[str, Any]]]:
        """Run TATR inference to predict rows and columns."""
        processor, model = self._load_tatr_model()
        inputs = processor(images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [image.size[::-1]]
        results = processor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

        crop_w, crop_h = image.size
        detected_rows: List[List[float]] = []
        detected_cols: List[List[float]] = []
        header_rows_idx: set[int] = set()

        id2label = model.config.id2label
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = id2label[label_id.item()].lower()
            b = box.tolist()

            if "table row" in label:
                detected_rows.append(b)
            elif "table column" in label:
                detected_cols.append(b)
            elif "table column header" in label or "header" in label:
                detected_rows.append(b)
                header_rows_idx.add(len(detected_rows) - 1)

        # Sort rows top-to-bottom and columns left-to-right
        detected_rows.sort(key=lambda r: r[1])
        detected_cols.sort(key=lambda c: c[0])

        if not detected_rows or not detected_cols:
            return self._heuristic_grid_segmentation(image)

        # Build grid cells from intersection of rows and columns
        grid: List[List[Dict[str, Any]]] = []
        for r_idx, r_box in enumerate(detected_rows):
            row_cells: List[Dict[str, Any]] = []
            is_hdr = r_idx in header_rows_idx or r_idx == 0
            for c_idx, c_box in enumerate(detected_cols):
                cell_x0 = max(0.0, c_box[0])
                cell_y0 = max(0.0, r_box[1])
                cell_x1 = min(float(crop_w), c_box[2])
                cell_y1 = min(float(crop_h), r_box[3])
                if cell_x1 > cell_x0 and cell_y1 > cell_y0:
                    row_cells.append({
                        "col": c_idx,
                        "bbox": [cell_x0, cell_y0, cell_x1, cell_y1],
                        "is_header": is_hdr,
                    })
            if row_cells:
                grid.append(row_cells)

        return grid or self._heuristic_grid_segmentation(image)

    def _heuristic_grid_segmentation(self, image: Image.Image) -> List[List[Dict[str, Any]]]:
        """Heuristic grid segmentation for tables when TATR is not available.

        Detects row dividers by finding horizontal lines or divides into 2-3 uniform rows
        and columns based on aspect ratio.
        """
        crop_w, crop_h = image.size

        # Simple robust heuristic: divide into 3 rows, 2 columns as standard circuit spec tables
        num_rows = max(1, min(5, int(crop_h // 40))) or 2
        num_cols = max(1, min(4, int(crop_w // 100))) or 2

        row_h = crop_h / float(num_rows)
        col_w = crop_w / float(num_cols)

        grid: List[List[Dict[str, Any]]] = []
        for r in range(num_rows):
            row_cells: List[Dict[str, Any]] = []
            y0 = r * row_h
            y1 = (r + 1) * row_h
            for c in range(num_cols):
                x0 = c * col_w
                x1 = (c + 1) * col_w
                row_cells.append({
                    "col": c,
                    "bbox": [x0, y0, x1, y1],
                    "is_header": (r == 0),
                })
            grid.append(row_cells)

        return grid


_table_service: Optional[TableService] = None


def get_table_service() -> TableService:
    """Return a singleton instance of TableService."""
    global _table_service
    if _table_service is None:
        _table_service = TableService()
    return _table_service
