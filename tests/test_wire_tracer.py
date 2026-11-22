"""
Unit tests for Module 2: wire_tracer.py - Preprocessing, Masking, Skeletonization, and Junction Classification
"""

import cv2
import numpy as np
import pytest

from diagram_engine.detector import DetectedComponent
from diagram_engine.wire_tracer import JunctionClass, WireTracer


def test_component_masking_inward_margin():
    tracer = WireTracer(inward_mask_margin=3)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Draw component bounding box from (20, 20) to (60, 60)
    # Place ink inside the box
    img[25:55, 25:55] = 0
    # Place wire entering at x=10 to x=22 at y=40
    img[39:42, 10:23] = 0

    comp = DetectedComponent(
        id="R1",
        class_name="resistor",
        bbox=[20, 20, 60, 60],
        confidence=0.95,
    )

    masked_binary, raw_binary = tracer.preprocess_and_mask(img, [comp])

    # With inward margin of 3, the interior [23:57, 23:57] must be masked to 0
    assert np.all(masked_binary[25:55, 25:55] == 0)
    # The wire trace outside the inward margin at x=10-21 should be foreground (255)
    assert np.any(masked_binary[39:42, 10:20] == 255)


def test_skeletonization_and_8_neighbor_junction_classification():
    tracer = WireTracer()

    # Create a synthetic skeleton with:
    # 1. An Endpoint (Degree == 1)
    # 2. A T-junction (Degree == 3)
    # 3. A 4-way crossing (Degree == 4)
    skel = np.zeros((80, 80), dtype=np.uint8)

    # 4-way cross centered at (40, 40)
    skel[40, 20:61] = 1  # horizontal line
    skel[20:61, 40] = 1  # vertical line

    raw_binary = (skel * 255).astype(np.uint8)

    junctions = tracer.classify_junctions(skel, raw_binary)

    # Check for crossing at (40, 40)
    crossings = [j for j in junctions if j.degree == 4]
    assert len(crossings) >= 1
    c = crossings[0]
    assert abs(c.x - 40) <= 2
    assert abs(c.y - 40) <= 2

    # Check for endpoints at the 4 extremities: (20, 40), (60, 40), (40, 20), (40, 60)
    endpoints = [j for j in junctions if j.degree == 1]
    assert len(endpoints) == 4


def test_solder_dot_vs_insulated_crossing():
    tracer = WireTracer(solder_dot_radius=8, solder_dot_fill_threshold=0.50)

    # Scenario A: Insulated crossing (just thin 2px intersecting lines, no filled dot)
    insulated_raw = np.zeros((60, 60), dtype=np.uint8)
    insulated_raw[29:31, 10:50] = 255  # horizontal thin wire
    insulated_raw[10:50, 29:31] = 255  # vertical thin wire

    has_dot_insulated = tracer.detect_solder_dot(insulated_raw, 30, 30)
    assert has_dot_insulated is False

    # Scenario B: Connected crossing with solid circular solder dot
    connected_raw = insulated_raw.copy()
    cv2.circle(connected_raw, (30, 30), 7, 255, -1)  # Solid circular solder dot

    has_dot_connected = tracer.detect_solder_dot(connected_raw, 30, 30)
    assert has_dot_connected is True
