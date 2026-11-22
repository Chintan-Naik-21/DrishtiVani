"""
DrishtiVani - Member 2: Circuit Diagram to Accessible Netlist Engine
Module 2: wire_tracer.py - OpenCV Preprocessing, Guo-Hall Skeletonization, and 8-Neighbor Junction Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from diagram_engine.detector import DetectedComponent

logger = logging.getLogger(__name__)


class JunctionClass(str, Enum):
    """Classification of skeleton wire points."""
    ENDPOINT = "endpoint"                    # Degree == 1 (Component lead or dangling wire)
    WIRE = "wire"                            # Degree == 2 (Continuous conductive trace)
    T_JUNCTION = "t_junction"                # Degree == 3 (Three-way electrical junction)
    CROSSING_CONNECTED = "crossing_dot"      # Degree >= 4 with solid solder dot (connected node)
    CROSSING_INSULATED = "crossing_insulated"# Degree >= 4 without solder dot (insulated bridge)


@dataclass
class JunctionPoint:
    """Classified junction or endpoint on the wire skeleton."""
    x: int
    y: int
    degree: int                              # 8-neighbor degree (1, 3, 4+)
    junction_type: JunctionClass             # Endpoint, T-junction, Crossing with/without solder dot
    has_solder_dot: bool = False             # True if circular solder dot detected at 4-way crossing
    cluster_id: int = -1                     # Cluster grouping index


@dataclass
class WireSegment:
    """A continuous 1-pixel-wide wire centerline segment between two junctions/endpoints."""
    segment_id: int
    points: List[Tuple[int, int]]            # List of (x, y) pixels along the wire path
    endpoint_a: Tuple[int, int]
    endpoint_b: Tuple[int, int]
    length_px: float = 0.0


class WireTracer:
    """
    OpenCV Preprocessing & Morphological Skeletonization Engine.
    Masks component interiors, computes Guo-Hall thinning, analyzes 8-neighbor
    convolutions, and resolves 4-way solder dots vs. insulated crossings.
    """

    def __init__(
        self,
        inward_mask_margin: int = 0,
        junction_cluster_radius: int = 5,
        solder_dot_radius: int = 8,
        solder_dot_fill_threshold: float = 0.55,
    ) -> None:
        """
        Args:
            inward_mask_margin: Inward padding in pixels when masking component interiors (0-2px).
                                Leaves terminal pins exposed at the boundary edges while severing internal symbols.
            junction_cluster_radius: Radius in pixels to merge adjacent junction pixels.
            solder_dot_radius: Search radius in pixels around degree-4 crossings for solder dots.
            solder_dot_fill_threshold: Pixel fill ratio threshold inside solder dot radius.
        """
        self.inward_mask_margin = max(0, min(4, inward_mask_margin))
        self.junction_cluster_radius = junction_cluster_radius
        self.solder_dot_radius = solder_dot_radius
        self.solder_dot_fill_threshold = solder_dot_fill_threshold

        # 8-neighbor convolution kernel
        self.neighbor_kernel = np.array(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]],
            dtype=np.uint8,
        )

    def preprocess_and_mask(
        self,
        image: np.ndarray,
        components: List[DetectedComponent],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to grayscale, apply Gaussian blur, binarize using Otsu's inverse
        thresholding, and mask component interiors with an inward margin.
        
        Returns:
            Tuple[binary_traces, raw_binary]:
              - binary_traces: Component interiors masked out (traces = 255, background = 0)
              - raw_binary: Unmasked binarized image (useful for solder dot detection)
        """
        # 1. Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Gaussian blur to suppress scanner / camera noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Detect background polarity: dark ink on white paper vs. dark background
        is_dark_background = float(np.mean(blurred)) < 127
        thresh_type = cv2.THRESH_BINARY if is_dark_background else cv2.THRESH_BINARY_INV

        # Otsu's thresholding (conductive traces = white 255, background = black 0)
        _, raw_binary = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

        # Morphological closing to bridge microscopic pencil/ink gaps
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_binary = cv2.morphologyEx(raw_binary, cv2.MORPH_CLOSE, bridge_kernel)

        # 4. Component Masking with Inward Margin:
        # Mask out interior of YOLO bounding boxes with 2-4px inward margin
        # so component lead wires entering the bounding box remain exposed.
        masked_binary = clean_binary.copy()
        h, w = masked_binary.shape[:2]
        margin = self.inward_mask_margin

        for comp in components:
            x1, y1, x2, y2 = comp.bbox
            mask_x1 = max(0, min(w, x1 + margin))
            mask_y1 = max(0, min(h, y1 + margin))
            mask_x2 = max(0, min(w, x2 - margin))
            mask_y2 = max(0, min(h, y2 - margin))

            if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                # Zero out internal component symbol artwork
                masked_binary[mask_y1:mask_y2, mask_x1:mask_x2] = 0

        return masked_binary, raw_binary

    def skeletonize(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply Guo-Hall morphological thinning (cv2.ximgproc.thinning)
        to reduce all wire traces to 1-pixel-wide centerlines.
        
        Returns:
            1-pixel-wide binary skeleton where skeleton pixels == 1 and background == 0.
        """
        thresh = (binary_mask > 0).astype(np.uint8) * 255

        # Attempt Guo-Hall thinning using OpenCV contrib ximgproc
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skeleton = cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        else:
            # Robust morphological fallback
            skeleton = np.zeros_like(thresh)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            temp = thresh.copy()
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
                subset = cv2.subtract(eroded, opened)
                skeleton = cv2.bitwise_or(skeleton, subset)
                temp = eroded.copy()
                if cv2.countNonZero(temp) == 0:
                    break

        return (skeleton > 0).astype(np.uint8)

    def detect_solder_dot(
        self,
        raw_binary: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> bool:
        """
        Check a local patch around a degree-4 crossing for a solid circular contour (solder dot).
        
        In circuit schematics:
        - A solder dot is a filled circular ink disk bridging the 4 intersecting lines.
        - An insulated crossing consists only of two thin perpendicular wires passing over each other
          without a solid circular blob.
        """
        h, w = raw_binary.shape[:2]
        r = self.solder_dot_radius

        x1 = max(0, center_x - r)
        y1 = max(0, center_y - r)
        x2 = min(w, center_x + r + 1)
        y2 = min(h, center_y + r + 1)

        patch = raw_binary[y1:y2, x1:x2]
        if patch.size == 0:
            return False

        # Measure circular fill density in a circular disk of radius r
        ph, pw = patch.shape
        cy, cx = center_y - y1, center_x - x1

        y_indices, x_indices = np.ogrid[:ph, :pw]
        dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        circle_mask = dist_from_center <= r

        total_disk_pixels = int(np.sum(circle_mask))
        if total_disk_pixels == 0:
            return False

        white_disk_pixels = int(np.sum((patch > 0) & circle_mask))
        fill_ratio = white_disk_pixels / float(total_disk_pixels)

        # Also inspect contours for circularity
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_circular_contour = False
        for c in contours:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            if peri > 0 and area > 15:
                circularity = 4.0 * np.pi * (area / (peri * peri))
                if circularity >= 0.50:
                    has_circular_contour = True
                    break

        # A solder dot has either substantial disk fill or a distinct circular blob
        return (fill_ratio >= self.solder_dot_fill_threshold) or (has_circular_contour and fill_ratio > 0.40)

    def classify_junctions(
        self,
        skeleton: np.ndarray,
        raw_binary: np.ndarray,
    ) -> List[JunctionPoint]:
        """
        Convolve the 1-pixel skeleton with the 8-neighbor kernel to classify:
          - Degree == 1: Wire endpoints (terminal leads or dangling wires)
          - Degree == 3: T-junctions (definite electrical nodes)
          - Degree >= 4: 4-way crossings (checked for solder dots vs. insulated crossings)
        """
        # Convolve skeleton with 8-neighbor kernel
        # cv2.filter2D on uint8 array
        neighbor_count = cv2.filter2D(skeleton, -1, self.neighbor_kernel, borderType=cv2.BORDER_CONSTANT)
        # Retain only skeleton pixels
        neighbor_count = neighbor_count * skeleton

        h, w = skeleton.shape[:2]

        # 1. Degree == 1: Endpoints
        endpoint_ys, endpoint_xs = np.where(neighbor_count == 1)
        raw_endpoints = list(zip(endpoint_xs, endpoint_ys))

        # 2. Degree == 3: T-Junctions
        t_ys, t_xs = np.where(neighbor_count == 3)
        raw_t_junctions = list(zip(t_xs, t_ys))

        # 3. Degree >= 4: Crossings
        cross_ys, cross_xs = np.where(neighbor_count >= 4)
        raw_crossings = list(zip(cross_xs, cross_ys))

        # Cluster raw adjacent junction pixels to avoid duplicate points around intersections
        clustered_junctions: List[JunctionPoint] = []

        # Process Endpoints (Degree == 1)
        clustered_endpoints = self._cluster_points(raw_endpoints, self.junction_cluster_radius)
        for x, y in clustered_endpoints:
            clustered_junctions.append(
                JunctionPoint(
                    x=x,
                    y=y,
                    degree=1,
                    junction_type=JunctionClass.ENDPOINT,
                    has_solder_dot=False,
                )
            )

        # Process T-Junctions (Degree == 3)
        clustered_t = self._cluster_points(raw_t_junctions, self.junction_cluster_radius)
        for x, y in clustered_t:
            clustered_junctions.append(
                JunctionPoint(
                    x=x,
                    y=y,
                    degree=3,
                    junction_type=JunctionClass.T_JUNCTION,
                    has_solder_dot=False,
                )
            )

        # Process Crossings (Degree >= 4)
        clustered_cross = self._cluster_points(raw_crossings, self.junction_cluster_radius + 2)
        for x, y in clustered_cross:
            has_dot = self.detect_solder_dot(raw_binary, x, y)
            j_type = (
                JunctionClass.CROSSING_CONNECTED
                if has_dot
                else JunctionClass.CROSSING_INSULATED
            )
            clustered_junctions.append(
                JunctionPoint(
                    x=x,
                    y=y,
                    degree=4,
                    junction_type=j_type,
                    has_solder_dot=has_dot,
                )
            )

        return clustered_junctions

    def _cluster_points(
        self,
        points: List[Tuple[int, int]],
        radius: int,
    ) -> List[Tuple[int, int]]:
        """Cluster proximate pixels into their spatial centroid."""
        if not points:
            return []

        clusters: List[List[Tuple[int, int]]] = []
        r_sq = radius * radius

        for x, y in points:
            matched = False
            for c in clusters:
                cx, cy = c[0]
                if (x - cx) ** 2 + (y - cy) ** 2 <= r_sq:
                    c.append((x, y))
                    matched = True
                    break
            if not matched:
                clusters.append([(x, y)])

        centroids = []
        for c in clusters:
            avg_x = int(round(sum(p[0] for p in c) / len(c)))
            avg_y = int(round(sum(p[1] for p in c) / len(c)))
            centroids.append((avg_x, avg_y))

        return centroids

    def trace_wire_paths(
        self,
        skeleton: np.ndarray,
        junctions: List[JunctionPoint],
    ) -> List[WireSegment]:
        """
        Trace continuous paths between endpoints and junctions along the skeleton.
        """
        h, w = skeleton.shape[:2]
        visited: Set[Tuple[int, int]] = set()
        segments: List[WireSegment] = []

        key_points = {(j.x, j.y): j for j in junctions}
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        seg_id = 0

        for junc in junctions:
            sx, sy = junc.x, junc.y

            for dx, dy in offsets:
                nx, ny = sx + dx, sy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if skeleton[ny, nx] != 1 or (nx, ny) in visited:
                    continue

                path: List[Tuple[int, int]] = [(sx, sy), (nx, ny)]
                visited.add((nx, ny))
                curr_x, curr_y = nx, ny

                while True:
                    # Check if reached another key point
                    found_end = False
                    for (kx, ky), _ in key_points.items():
                        if (curr_x - kx) ** 2 + (curr_y - ky) ** 2 <= (self.junction_cluster_radius + 1) ** 2:
                            if (kx, ky) != (sx, sy):
                                path.append((kx, ky))
                                found_end = True
                                break
                    if found_end:
                        break

                    next_step = None
                    for ndx, ndy in offsets:
                        step_x, step_y = curr_x + ndx, curr_y + ndy
                        if 0 <= step_x < w and 0 <= step_y < h:
                            if skeleton[step_y, step_x] == 1 and (step_x, step_y) not in visited:
                                next_step = (step_x, step_y)
                                break

                    if next_step is None:
                        break

                    curr_x, curr_y = next_step
                    path.append((curr_x, curr_y))
                    visited.add((curr_x, curr_y))

                if len(path) >= 3:
                    seg_id += 1
                    segments.append(
                        WireSegment(
                            segment_id=seg_id,
                            points=path,
                            endpoint_a=path[0],
                            endpoint_b=path[-1],
                            length_px=float(len(path)),
                        )
                    )

        return segments
