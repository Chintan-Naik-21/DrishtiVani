"""PDF rendering service using PyMuPDF (fitz).

Extracts pages from uploaded PDF documents and renders them as optimized
images at the configured DPI for downstream layout and OCR processing.
Memory-conscious: processes pages one at a time and frees pixmaps immediately.
"""

import io
import logging
from typing import Generator, Tuple
from PIL import Image
import pymupdf

logger = logging.getLogger(__name__)


class PDFService:
    """Service for rendering PDF pages into raster images using PyMuPDF."""

    def __init__(self, default_dpi: int = 150, default_format: str = "PNG") -> None:
        """Initialize PDF service with default rendering options."""
        self.default_dpi = default_dpi
        self.default_format = default_format.upper()

    def get_page_count(self, pdf_bytes: bytes) -> int:
        """Return the total number of pages in the given PDF file.

        Args:
            pdf_bytes: Raw bytes of the PDF document.

        Returns:
            int: Page count of the document.

        Raises:
            ValueError: If the PDF cannot be opened or is corrupted.
        """
        try:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.error("Failed to read PDF page count: %s", e)
            raise ValueError(f"Invalid or corrupted PDF file: {e}") from e

    def render_page(
        self,
        pdf_bytes: bytes,
        page_number: int,
        dpi: int = 150,
        image_format: str = "PNG",
    ) -> Tuple[bytes, int, int]:
        """Render a single 1-indexed page of a PDF to optimized image bytes.

        Args:
            pdf_bytes: Raw bytes of the PDF document.
            page_number: 1-indexed page number to render.
            dpi: Resolution in dots per inch (default 150).
            image_format: Encoding format ('PNG' or 'JPEG').

        Returns:
            Tuple[bytes, int, int]: (image_bytes, width_px, height_px).
        """
        fmt = image_format.upper()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page_number < 1 or page_number > len(doc):
                raise IndexError(f"Page number {page_number} is out of range [1, {len(doc)}]")

            page = doc[page_number - 1]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            width_px, height_px = pix.width, pix.height

            # Encode and optimize using Pillow for maximum storage savings on Supabase Free Tier
            raw_png = pix.tobytes("png")
            del pix

            with Image.open(io.BytesIO(raw_png)) as img:
                buf = io.BytesIO()
                if fmt == "JPEG":
                    rgb_img = img.convert("RGB")
                    rgb_img.save(buf, format="JPEG", quality=95, optimize=True)
                else:
                    img.save(buf, format="PNG", optimize=True)
                image_bytes = buf.getvalue()

            return image_bytes, width_px, height_px
        finally:
            doc.close()

    def iter_pages(
        self,
        pdf_bytes: bytes,
        dpi: int = 150,
        image_format: str = "PNG",
    ) -> Generator[Tuple[int, bytes, int, int], None, None]:
        """Generator yielding rendered pages one at a time to minimize memory usage.

        Yields:
            Tuple[int, bytes, int, int]: (page_number, image_bytes, width_px, height_px)
        """
        fmt = image_format.upper()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            total_pages = len(doc)
            logger.info("Iterating %d pages for rendering at %d DPI (%s)...", total_pages, dpi, fmt)

            for idx in range(total_pages):
                page_number = idx + 1
                page = doc[idx]
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                width_px, height_px = pix.width, pix.height

                raw_png = pix.tobytes("png")
                del pix

                with Image.open(io.BytesIO(raw_png)) as img:
                    buf = io.BytesIO()
                    if fmt == "JPEG":
                        rgb_img = img.convert("RGB")
                        rgb_img.save(buf, format="JPEG", quality=95, optimize=True)
                    else:
                        img.save(buf, format="PNG", optimize=True)
                    image_bytes = buf.getvalue()

                yield page_number, image_bytes, width_px, height_px
        finally:
            doc.close()


_pdf_service = None


def get_pdf_service() -> PDFService:
    """Return a singleton instance of PDFService."""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = PDFService()
    return _pdf_service
