"""
Unit tests for Module 1: detector.py - Symbol Detection and Terminal Pin Estimation
"""

import numpy as np
import pytest

from diagram_engine.detector import ComponentDetector, DetectedComponent


def test_detector_initialization():
    detector = ComponentDetector(model_path=None)
    assert detector is not None
    assert detector.model is None


def test_terminal_pin_estimation_passive_horizontal():
    detector = ComponentDetector(model_path=None)
    # Horizontal resistor: bbox [100, 50, 200, 70] (w=100, h=20)
    terminals = detector.estimate_terminal_pins("resistor", [100, 50, 200, 70])
    assert len(terminals) == 2
    assert terminals[0].name == "pin_1"
    assert terminals[0].location == (100, 60)
    assert terminals[1].name == "pin_2"
    assert terminals[1].location == (200, 60)


def test_terminal_pin_estimation_passive_vertical():
    detector = ComponentDetector(model_path=None)
    # Vertical resistor: bbox [100, 50, 120, 150] (w=20, h=100)
    terminals = detector.estimate_terminal_pins("resistor", [100, 50, 120, 150])
    assert len(terminals) == 2
    assert terminals[0].name == "pin_1"
    assert terminals[0].location == (110, 50)
    assert terminals[1].name == "pin_2"
    assert terminals[1].location == (110, 150)


def test_terminal_pin_estimation_voltage_source():
    detector = ComponentDetector(model_path=None)
    # Vertical voltage source: bbox [50, 100, 90, 180]
    terminals = detector.estimate_terminal_pins("voltage_source", [50, 100, 90, 180])
    assert len(terminals) == 2
    assert terminals[0].name == "positive"
    assert terminals[0].location == (70, 100)
    assert terminals[1].name == "negative"
    assert terminals[1].location == (70, 180)


def test_terminal_pin_estimation_ground():
    detector = ComponentDetector(model_path=None)
    # Ground connects at top center: bbox [40, 200, 80, 240]
    terminals = detector.estimate_terminal_pins("ground", [40, 200, 80, 240])
    assert len(terminals) == 1
    assert terminals[0].name == "ground"
    assert terminals[0].location == (60, 200)


def test_terminal_pin_estimation_bjt_transistor():
    detector = ComponentDetector(model_path=None)
    # BJT: Base at (x1, cy), Collector at (x2, y1), Emitter at (x2, y2)
    terminals = detector.estimate_terminal_pins("transistor_bjt", [100, 100, 160, 160])
    assert len(terminals) == 3
    names = [t.name for t in terminals]
    assert "base" in names
    assert "collector" in names
    assert "emitter" in names
    b = next(t for t in terminals if t.name == "base")
    assert b.location == (100, 130)


def test_terminal_pin_estimation_opamp():
    detector = ComponentDetector(model_path=None)
    # Op-Amp: inverting, non-inverting, out
    terminals = detector.estimate_terminal_pins("opamp", [200, 100, 260, 160])
    assert len(terminals) == 3
    names = [t.name for t in terminals]
    assert "in_inverting" in names
    assert "in_non_inverting" in names
    assert "output" in names
    out = next(t for t in terminals if t.name == "output")
    assert out.location == (260, 130)


def test_detect_with_initial_detections():
    detector = ComponentDetector(model_path=None)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    raw_boxes = [
        {"class": "voltage_source", "bbox": [50, 100, 90, 180], "confidence": 0.96, "value": "10V"},
        {"class": "resistor", "bbox": [150, 80, 230, 100], "confidence": 0.93, "value": "1k"},
        {"class": "ground", "bbox": [150, 220, 190, 250], "confidence": 0.99},
    ]
    comps = detector.detect(img, initial_detections=raw_boxes)
    assert len(comps) == 3
    assert comps[0].id == "V1"
    assert comps[0].class_name == "voltage_source"
    assert comps[1].id == "R1"
    assert comps[1].class_name == "resistor"
    assert comps[2].id == "GND1"
    assert comps[2].class_name == "ground"
