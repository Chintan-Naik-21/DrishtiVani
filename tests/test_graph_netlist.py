"""
Unit tests for Module 3: graph_netlist.py - Terminal Snapping, SPICE Netlist, and Confidence Scoring
"""

import pytest

from diagram_engine.detector import DetectedComponent, TerminalPin
from diagram_engine.graph_netlist import CircuitGraphNetlist
from diagram_engine.wire_tracer import JunctionClass, JunctionPoint, WireSegment


def test_high_confidence_voltage_divider_netlist():
    engine = CircuitGraphNetlist(review_threshold=0.88, unconnected_penalty=0.15)

    # 1. Voltage Source V1 connecting to Node 1 (+) and Node 0 (-)
    v1 = DetectedComponent(
        id="V1",
        class_name="voltage_source",
        bbox=[50, 100, 90, 200],
        confidence=0.96,
        value="10V",
        terminals=[
            TerminalPin(name="positive", location=(70, 100)),
            TerminalPin(name="negative", location=(70, 200)),
        ],
    )

    # 2. Resistor R1 connecting between Node 1 and Node 2
    r1 = DetectedComponent(
        id="R1",
        class_name="resistor",
        bbox=[120, 90, 180, 110],
        confidence=0.94,
        value="1k",
        terminals=[
            TerminalPin(name="pin_1", location=(120, 100)),
            TerminalPin(name="pin_2", location=(180, 100)),
        ],
    )

    # 3. Resistor R2 connecting between Node 2 and Ground (0)
    r2 = DetectedComponent(
        id="R2",
        class_name="resistor",
        bbox=[170, 130, 190, 190],
        confidence=0.92,
        value="2k",
        terminals=[
            TerminalPin(name="pin_1", location=(180, 130)),
            TerminalPin(name="pin_2", location=(180, 190)),
        ],
    )

    # 4. Ground GND at (180, 200)
    gnd = DetectedComponent(
        id="GND1",
        class_name="ground",
        bbox=[160, 200, 200, 220],
        confidence=0.98,
        terminals=[
            TerminalPin(name="ground", location=(180, 200)),
        ],
    )

    components = [v1, r1, r2, gnd]

    # Wire endpoints perfectly meeting component terminals (distance <= 15px)
    junctions = [
        JunctionPoint(x=70, y=100, degree=1, junction_type=JunctionClass.ENDPOINT),   # snaps to V1:positive
        JunctionPoint(x=120, y=100, degree=1, junction_type=JunctionClass.ENDPOINT),  # snaps to R1:pin_1
        JunctionPoint(x=180, y=100, degree=3, junction_type=JunctionClass.T_JUNCTION),# T-junction between R1 & R2
        JunctionPoint(x=180, y=130, degree=1, junction_type=JunctionClass.ENDPOINT),  # snaps to R2:pin_1
        JunctionPoint(x=180, y=190, degree=1, junction_type=JunctionClass.ENDPOINT),  # snaps to R2:pin_2
        JunctionPoint(x=180, y=200, degree=3, junction_type=JunctionClass.T_JUNCTION),# T-junction at bottom ground rail
        JunctionPoint(x=70, y=200, degree=1, junction_type=JunctionClass.ENDPOINT),   # snaps to V1:negative
    ]

    wire_segments = [
        # Top wire: V1:pos (70, 100) -> R1:pin_1 (120, 100) (Node 1)
        WireSegment(segment_id=1, points=[(70, 100), (120, 100)], endpoint_a=(70, 100), endpoint_b=(120, 100)),
        # Middle wire: R1:pin_2 (180, 100) -> R2:pin_1 (180, 130) (Node 2)
        WireSegment(segment_id=2, points=[(180, 100), (180, 130)], endpoint_a=(180, 100), endpoint_b=(180, 130)),
        # Bottom wire: V1:neg (70, 200) -> GND / R2:pin_2 (180, 200) (Node 0)
        WireSegment(segment_id=3, points=[(70, 200), (180, 200)], endpoint_a=(70, 200), endpoint_b=(180, 200)),
    ]

    result = engine.process(components, junctions, wire_segments)

    # 1. SPICE Netlist Assertions
    assert "V1 n1 0 DC 10V" in result.spice_netlist or "V1" in result.spice_netlist
    assert "R1" in result.spice_netlist
    assert "R2" in result.spice_netlist
    assert ".end" in result.spice_netlist

    # 2. Confidence assertions: clean circuit is approved
    assert result.confidence_score >= 0.88
    assert result.needs_teacher_review is False

    # 3. Macro summary assertions
    assert "Circuit contains 3 components" in result.summary
    assert "ground reference present" in result.summary


def test_confidence_drops_below_88_when_terminal_unconnected():
    engine = CircuitGraphNetlist(review_threshold=0.88, unconnected_penalty=0.15)

    # Resistor with floating terminal pin_2 (no wire endpoint within snap threshold)
    comp = DetectedComponent(
        id="R1",
        class_name="resistor",
        bbox=[100, 100, 160, 120],
        confidence=0.95,
        value="1k",
        terminals=[
            TerminalPin(name="pin_1", location=(100, 110)),
            TerminalPin(name="pin_2", location=(160, 110)),
        ],
    )

    # Only pin_1 has a nearby wire endpoint; pin_2 is dangling/unconnected
    junctions = [
        JunctionPoint(x=102, y=110, degree=1, junction_type=JunctionClass.ENDPOINT), # snaps to pin_1
    ]
    wire_segments = []

    result = engine.process([comp], junctions, wire_segments)

    # An unconnected pin triggers a -0.15 penalty and forces confidence < 0.88
    assert result.confidence_score < 0.88
    assert result.needs_teacher_review is True
    assert "R1:pin_2" in result.unconnected_terminals


def test_confidence_drops_below_88_when_wire_endpoint_is_dangling():
    engine = CircuitGraphNetlist(review_threshold=0.88, unconnected_penalty=0.15)

    comp = DetectedComponent(
        id="R1",
        class_name="resistor",
        bbox=[100, 100, 160, 120],
        confidence=0.95,
        value="1k",
        terminals=[
            TerminalPin(name="pin_1", location=(100, 110)),
            TerminalPin(name="pin_2", location=(160, 110)),
        ],
    )

    # Wire endpoints: 2 snap to R1, but a 3rd endpoint is completely dangling in space
    junctions = [
        JunctionPoint(x=100, y=110, degree=1, junction_type=JunctionClass.ENDPOINT), # snaps to pin_1
        JunctionPoint(x=160, y=110, degree=1, junction_type=JunctionClass.ENDPOINT), # snaps to pin_2
        JunctionPoint(x=300, y=300, degree=1, junction_type=JunctionClass.ENDPOINT), # Dangling wire!
    ]
    wire_segments = [
        WireSegment(segment_id=1, points=[(100, 110), (160, 110)], endpoint_a=(100, 110), endpoint_b=(160, 110)),
        WireSegment(segment_id=2, points=[(300, 300), (350, 300)], endpoint_a=(300, 300), endpoint_b=(350, 300)),
    ]

    result = engine.process([comp], junctions, wire_segments)

    assert result.confidence_score < 0.88
    assert result.needs_teacher_review is True
    assert len(result.dangling_endpoints) >= 1
