"""
DrishtiVani - Member 2: Circuit Diagram to Accessible Netlist Engine
Module 1: detector.py - Symbol Detection & Terminal Pin Port Extraction using YOLO11
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Isolate Ultralytics settings cache inside the project to prevent Windows permission issues
_local_yolo_dir = Path(__file__).resolve().parent.parent / ".yolo"
_local_yolo_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_local_yolo_dir))
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Check for Ultralytics YOLO availability
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as _e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics package not loaded: %s", _e)


# Standard SPICE prefix and normalized class mapping
CLASS_MAP = {
    "resistor": ("resistor", "R", "1k"),
    "r": ("resistor", "R", "1k"),
    "capacitor": ("capacitor", "C", "10u"),
    "cap": ("capacitor", "C", "10u"),
    "c": ("capacitor", "C", "10u"),
    "inductor": ("inductor", "L", "1m"),
    "ind": ("inductor", "L", "1m"),
    "l": ("inductor", "L", "1m"),
    "voltage_source": ("voltage_source", "V", "10V"),
    "vsource": ("voltage_source", "V", "10V"),
    "battery": ("voltage_source", "V", "10V"),
    "v": ("voltage_source", "V", "10V"),
    "dc_voltage": ("voltage_source", "V", "10V"),
    "current_source": ("current_source", "I", "1A"),
    "isource": ("current_source", "I", "1A"),
    "i": ("current_source", "I", "1A"),
    "diode": ("diode", "D", "D1N4148"),
    "d": ("diode", "D", "D1N4148"),
    "ground": ("ground", "GND", "0"),
    "gnd": ("ground", "GND", "0"),
    "transistor": ("transistor_bjt", "Q", "2N2222"),
    "bjt": ("transistor_bjt", "Q", "2N2222"),
    "transistor_bjt": ("transistor_bjt", "Q", "2N2222"),
    "opamp": ("opamp", "X", "LM741"),
    "op_amp": ("opamp", "X", "LM741"),
    "switch": ("switch", "S", "SW"),
    "sw": ("switch", "S", "SW"),
}


@dataclass
class TerminalPin:
    """A connection pin or lead port on a circuit component."""
    name: str                                  # e.g., "pin_1", "pin_2", "positive", "negative", "ground"
    location: Tuple[int, int]                  # (x, y) coordinates in pixel space
    connected_node: Optional[str] = None       # Assigned electrical node, e.g., "n1", "0"
    is_snapped: bool = False                   # True if snapped to a valid wire endpoint
    snap_distance: Optional[float] = None      # Distance in pixels to the snapped wire endpoint


@dataclass
class DetectedComponent:
    """An electrical component detected by YOLO11 or layout extraction."""
    id: str                                    # e.g., "R1", "V1", "GND1"
    class_name: str                            # Canonical class: "resistor", "voltage_source", etc.
    bbox: List[int]                            # [x1, y1, x2, y2]
    confidence: float                          # Detection confidence (0.0 - 1.0)
    terminals: List[TerminalPin] = field(default_factory=list)
    value: Optional[str] = None                # e.g., "1k", "10V"
    orientation: str = "horizontal"            # "horizontal" or "vertical"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class ComponentDetector:
    """
    Symbol Detection Engine using Ultralytics YOLO11.
    Performs inference and geometric terminal pin extraction for circuit schematics.
    """

    def __init__(
        self,
        model_path: Optional[str] = "yolo11s.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        """
        Initialize the YOLO11 model detector.
        
        Args:
            model_path: Path to YOLO11 weights (e.g. 'yolo11s.pt', 'best.pt').
            conf_threshold: Minimum detection confidence threshold.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._counts: Dict[str, int] = {}

        if model_path and ULTRALYTICS_AVAILABLE:
            try:
                # Load custom or pre-trained YOLO11 model only if weights exist locally
                if Path(model_path).exists():
                    self.model = YOLO(model_path)
                    logger.info("Loaded custom YOLO11 model from %s", model_path)
                else:
                    logger.info("YOLO model weights '%s' not found locally. Running in layout/fallback mode.", model_path)
            except Exception as e:
                logger.warning("Failed to initialize YOLO model: %s", e)

    def reset_id_counters(self) -> None:
        """Reset component ID counters between runs."""
        self._counts = {}

    def _generate_id(self, prefix: str) -> str:
        """Generate consecutive component IDs (e.g., R1, R2, V1)."""
        idx = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = idx
        return f"{prefix}{idx}"

    def _normalize_class(self, raw_label: str) -> Tuple[str, str, str]:
        """Normalize raw class name to (canonical_class, spice_prefix, default_value)."""
        cleaned = str(raw_label).lower().strip().replace(" ", "_").replace("-", "_")
        return CLASS_MAP.get(cleaned, (cleaned, "U", "1"))

    def estimate_terminal_pins(
        self,
        class_name: str,
        bbox: List[int],
        orientation: Optional[str] = None,
    ) -> List[TerminalPin]:
        """
        Estimate the geometric terminal pin locations on the component boundary
        based on the bounding box geometry and orientation.
        
        Args:
            class_name: Canonical class name (e.g. 'resistor', 'voltage_source').
            bbox: Bounding box [x1, y1, x2, y2].
            orientation: Optional explicit 'horizontal' or 'vertical'.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if orientation is None:
            orientation = "horizontal" if w >= h else "vertical"

        terminals: List[TerminalPin] = []

        # 1. Ground Reference (single lead at top center)
        if class_name == "ground":
            terminals.append(TerminalPin(name="ground", location=(cx, y1)))

        # 2. Two-terminal passive or directional component (Resistor, Capacitor, Inductor, Diode, Switch)
        elif class_name in ("resistor", "capacitor", "inductor", "diode", "switch"):
            if orientation == "horizontal":
                terminals.append(TerminalPin(name="pin_1", location=(x1, cy)))
                terminals.append(TerminalPin(name="pin_2", location=(x2, cy)))
            else:
                terminals.append(TerminalPin(name="pin_1", location=(cx, y1)))
                terminals.append(TerminalPin(name="pin_2", location=(cx, y2)))

        # 3. Two-terminal active source (Voltage Source, Current Source)
        elif class_name in ("voltage_source", "current_source"):
            if orientation == "horizontal":
                terminals.append(TerminalPin(name="positive", location=(x1, cy)))
                terminals.append(TerminalPin(name="negative", location=(x2, cy)))
            else:
                terminals.append(TerminalPin(name="positive", location=(cx, y1)))
                terminals.append(TerminalPin(name="negative", location=(cx, y2)))

        # 4. Transistor BJT (3 terminals: Base, Collector, Emitter)
        elif class_name == "transistor_bjt":
            terminals.append(TerminalPin(name="base", location=(x1, cy)))
            terminals.append(TerminalPin(name="collector", location=(x2, y1)))
            terminals.append(TerminalPin(name="emitter", location=(x2, y2)))

        # 5. Operational Amplifier (3 terminals: Inverting, Non-Inverting, Output)
        elif class_name == "opamp":
            terminals.append(TerminalPin(name="in_inverting", location=(x1, y1 + h // 3)))
            terminals.append(TerminalPin(name="in_non_inverting", location=(x1, y1 + (2 * h) // 3)))
            terminals.append(TerminalPin(name="output", location=(x2, cy)))

        # Fallback default two-terminal component
        else:
            if orientation == "horizontal":
                terminals.append(TerminalPin(name="pin_1", location=(x1, cy)))
                terminals.append(TerminalPin(name="pin_2", location=(x2, cy)))
            else:
                terminals.append(TerminalPin(name="pin_1", location=(cx, y1)))
                terminals.append(TerminalPin(name="pin_2", location=(cx, y2)))

        return terminals

    def detect(
        self,
        image: Union[str, np.ndarray, Image.Image],
        initial_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[DetectedComponent]:
        """
        Execute symbol detection and terminal pin extraction.
        
        Args:
            image: Image file path, numpy ndarray (BGR), or PIL Image.
            initial_detections: Optional pre-extracted bounding boxes from Member 1's layout stage:
                [{"class": "resistor", "bbox": [x1, y1, x2, y2], "confidence": 0.95, "value": "1k"}, ...]
                
        Returns:
            List[DetectedComponent] with derived terminal pin locations.
        """
        self.reset_id_counters()
        components: List[DetectedComponent] = []

        # Convert image input to numpy BGR
        if isinstance(image, str):
            cv_img = cv2.imread(image)
            if cv_img is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, Image.Image):
            cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_img = image.copy()
            if len(cv_img.shape) == 2:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError("Unsupported image format. Provide filepath, numpy ndarray, or PIL Image.")

        # Case 1: Initial bounding boxes passed explicitly (e.g. from layout parser or integration tests)
        if initial_detections:
            for item in initial_detections:
                raw_cls = item.get("class", item.get("class_name", item.get("label", "unknown")))
                canonical_cls, prefix, default_val = self._normalize_class(raw_cls)
                bbox_raw = item.get("bbox", [0, 0, 50, 50])
                bbox = [int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3])]
                confidence = float(item.get("confidence", 0.95))
                orientation = item.get("orientation")
                comp_id = item.get("id") or self._generate_id(prefix)
                value = item.get("value") or default_val

                terminals = self.estimate_terminal_pins(canonical_cls, bbox, orientation)
                orient = orientation or ("horizontal" if (bbox[2] - bbox[0]) >= (bbox[3] - bbox[1]) else "vertical")

                components.append(
                    DetectedComponent(
                        id=comp_id,
                        class_name=canonical_cls,
                        bbox=bbox,
                        confidence=confidence,
                        terminals=terminals,
                        value=value,
                        orientation=orient,
                    )
                )
            return components

        # Case 2: Run YOLO11 inference if model is loaded
        if self.model is not None:
            results = self.model.predict(cv_img, conf=self.conf_threshold, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                names = self.model.names
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    raw_name = names.get(cls_id, "unknown")
                    canonical_cls, prefix, default_val = self._normalize_class(raw_name)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    conf = float(box.conf[0].item())

                    comp_id = self._generate_id(prefix)
                    value = default_val
                    terminals = self.estimate_terminal_pins(canonical_cls, bbox)
                    orient = "horizontal" if (bbox[2] - bbox[0]) >= (bbox[3] - bbox[1]) else "vertical"

                    components.append(
                        DetectedComponent(
                            id=comp_id,
                            class_name=canonical_cls,
                            bbox=bbox,
                            confidence=conf,
                            terminals=terminals,
                            value=value,
                            orientation=orient,
                        )
                    )

        return components
