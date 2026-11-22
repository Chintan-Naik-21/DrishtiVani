"""
DrishtiVani - Member 2: Circuit Diagram to Accessible Netlist Engine
Diagnostic Visualization Overlay Generator
"""

from __future__ import annotations

import base64
from typing import Dict, List, Optional
import cv2
import numpy as np

from diagram_engine.detector import DetectedComponent
from diagram_engine.wire_tracer import JunctionClass, JunctionPoint


class CircuitVisualizer:
    """Renders visual inspection canvases with skeleton, junctions, pins, and badges."""

    def render_overlay(
        self,
        base_image: np.ndarray,
        skeleton: np.ndarray,
        components: List[DetectedComponent],
        junctions: List[JunctionPoint],
        confidence_score: float,
        needs_teacher_review: bool,
    ) -> np.ndarray:
        """Create an annotated visual canvas."""
        if len(base_image.shape) == 2:
            canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = base_image.copy()

        # 1. Overlay Skeleton Wires in Cyan [BGR: (230, 216, 50)]
        ys, xs = np.where(skeleton == 1)
        canvas[ys, xs] = [230, 216, 50]

        # 2. Draw Classified Junctions
        for j in junctions:
            pt = (j.x, j.y)
            if j.degree == 1:
                # Green Circle for Endpoints (N=1)
                cv2.circle(canvas, pt, 5, (50, 205, 50), -1)
                cv2.circle(canvas, pt, 6, (0, 100, 0), 1)
            elif j.degree == 3:
                # Yellow Circle for T-Junctions (N=3)
                cv2.circle(canvas, pt, 6, (0, 215, 255), -1)
                cv2.circle(canvas, pt, 7, (0, 140, 200), 1)
            elif j.degree >= 4:
                # Red Rectangle for Crossings (N>=4)
                color = (0, 0, 255) if j.has_solder_dot else (200, 50, 200)
                cv2.rectangle(canvas, (j.x - 5, j.y - 5), (j.x + 5, j.y + 5), color, -1)
                cv2.rectangle(canvas, (j.x - 6, j.y - 6), (j.x + 6, j.y + 6), (50, 50, 50), 1)

        # 3. Draw Detected Components, Pins, and Node Badges
        for comp in components:
            x1, y1, x2, y2 = comp.bbox
            box_color = (255, 120, 30) if comp.confidence >= 0.88 else (50, 50, 220)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)

            # Component label badge
            label = f"{comp.id}: {comp.class_name} ({comp.confidence:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), box_color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Terminals
            for t in comp.terminals:
                tx, ty = t.location
                pin_color = (0, 255, 0) if t.is_snapped else (0, 0, 255)
                cv2.circle(canvas, (tx, ty), 4, pin_color, -1)

                if t.connected_node:
                    cv2.putText(
                        canvas,
                        f"{t.connected_node}",
                        (tx + 4, ty - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (180, 0, 180),
                        1,
                        cv2.LINE_AA,
                    )

        # 4. Top Status Banner
        w = canvas.shape[1]
        banner_h = 36
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner_bg = (34, 139, 34) if not needs_teacher_review else (0, 69, 215)
        banner[:] = banner_bg

        status_text = (
            f"[APPROVED] Score: {confidence_score:.2f} -> Ready for Screen Reader"
            if not needs_teacher_review
            else f"[TEACHER REVIEW REQUIRED] Score: {confidence_score:.2f} (< 0.88)"
        )

        cv2.putText(
            banner,
            status_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return np.vstack([banner, canvas])

    def to_base64(self, image: np.ndarray) -> str:
        """Convert BGR image to base64 data URL string."""
        _, buffer = cv2.imencode(".png", image)
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
