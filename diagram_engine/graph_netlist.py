"""
DrishtiVani - Member 2: Circuit Diagram to Accessible Netlist Engine
Module 3: graph_netlist.py - NetworkX Graph Construction, Terminal Snapping, SPICE Netlist, and Confidence Evaluation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from diagram_engine.detector import DetectedComponent, TerminalPin
from diagram_engine.wire_tracer import JunctionClass, JunctionPoint, WireSegment

logger = logging.getLogger(__name__)


@dataclass
class ConnectedComponentRef:
    """Component terminal reference connected to an electrical node."""
    component_id: str
    class_name: str
    terminal: str


@dataclass
class ElectricalNodeInfo:
    """Structured information for a single equipotential electrical node."""
    node_id: str
    description: str
    connected_components: List[ConnectedComponentRef] = field(default_factory=list)


@dataclass
class NetlistPipelineResult:
    """Complete package produced by the Graph & Netlist Engine."""
    spice_netlist: str
    nodes: List[ElectricalNodeInfo]
    confidence_score: float
    needs_teacher_review: bool
    summary: str
    unconnected_terminals: List[str] = field(default_factory=list)
    dangling_endpoints: List[Tuple[int, int]] = field(default_factory=list)


class CircuitGraphNetlist:
    """
    Step 3 Engine:
      1. Snaps component terminals to wire endpoints.
      2. Assembles an undirected NetworkX graph of conductive traces.
      3. Resolves equipotential nodes (n0/0 for Ground, n1, n2, etc.).
      4. Synthesizes valid SPICE netlist (.cir).
      5. Evaluates confidence score with -0.15 penalties for unconnected/dangling wires.
    """

    def __init__(
        self,
        snap_threshold_px: float = 15.0,
        review_threshold: float = 0.88,
        unconnected_penalty: float = 0.15,
    ) -> None:
        """
        Args:
            snap_threshold_px: Max Euclidean distance in pixels to snap a terminal to a wire endpoint.
            review_threshold: Confidence threshold below which needs_teacher_review is triggered (0.88).
            unconnected_penalty: Penalty subtracted for each unconnected terminal or dangling wire (-0.15).
        """
        self.snap_threshold_px = snap_threshold_px
        self.review_threshold = review_threshold
        self.unconnected_penalty = unconnected_penalty

    def snap_terminals_to_endpoints(
        self,
        components: List[DetectedComponent],
        junctions: List[JunctionPoint],
        wire_segments: List[WireSegment],
        skeleton: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[DetectedComponent, TerminalPin, Tuple[int, int], float]], Set[Tuple[int, int]]]:
        """
        Snap component terminal pins to the nearest Degree == 1 wire endpoints.
        
        Returns:
            Tuple of:
              - List of snap matches: (component, terminal, snapped_endpoint, distance)
              - Set of used/snapped wire endpoints
        """
        # Gather all Degree == 1 endpoints
        endpoints: List[Tuple[int, int]] = [
            (j.x, j.y) for j in junctions if j.junction_type == JunctionClass.ENDPOINT
        ]

        # Also include wire segment endpoints
        for seg in wire_segments:
            if seg.endpoint_a not in endpoints:
                endpoints.append(seg.endpoint_a)
            if seg.endpoint_b not in endpoints:
                endpoints.append(seg.endpoint_b)

        snap_matches: List[Tuple[DetectedComponent, TerminalPin, Tuple[int, int], float]] = []
        used_endpoints: Set[Tuple[int, int]] = set()

        for comp in components:
            for term in comp.terminals:
                tx, ty = term.location
                best_ep: Optional[Tuple[int, int]] = None
                best_dist = float("inf")

                for ex, ey in endpoints:
                    dist = math.hypot(tx - ex, ty - ey)
                    if dist < best_dist:
                        best_dist = dist
                        best_ep = (ex, ey)

                # Check fallback to skeleton pixels if endpoints were slightly recessed
                if (best_dist > self.snap_threshold_px or best_ep is None) and skeleton is not None:
                    ys, xs = np.where(skeleton == 1)
                    for px, py in zip(xs, ys):
                        dist = math.hypot(tx - px, ty - py)
                        if dist < best_dist:
                            best_dist = dist
                            best_ep = (px, py)

                if best_ep is not None and best_dist <= self.snap_threshold_px:
                    term.is_snapped = True
                    term.snap_distance = round(best_dist, 2)
                    snap_matches.append((comp, term, best_ep, best_dist))
                    used_endpoints.add(best_ep)
                else:
                    term.is_snapped = False
                    term.snap_distance = round(best_dist, 2) if best_ep else None

        return snap_matches, used_endpoints

    def build_circuit_graph(
        self,
        wire_segments: List[WireSegment],
        junctions: List[JunctionPoint],
        snap_matches: List[Tuple[DetectedComponent, TerminalPin, Tuple[int, int], float]],
        components: List[DetectedComponent],
        skeleton: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, List[Tuple[DetectedComponent, TerminalPin]]], nx.Graph]:
        """
        Assemble the undirected NetworkX graph from wire paths and assign
        equipotential electrical node labels (n0/0 for Ground, n1, n2, etc.).
        """
        G = nx.Graph()

        # Add wire segments as graph edges
        for seg in wire_segments:
            pts = seg.points
            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]
                G.add_edge(p1, p2)

        # If raw skeleton is provided, ensure all 8-neighbor continuous pixels are linked
        if skeleton is not None:
            ys, xs = np.where(skeleton == 1)
            skel_pixels = set(zip(xs, ys))
            for x, y in skel_pixels:
                for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in skel_pixels:
                        G.add_edge((x, y), neighbor)

        # Handle insulated crossings:
        # Crossings without solder dots must NOT bridge horizontal and vertical traces
        for j in junctions:
            if j.junction_type == JunctionClass.CROSSING_INSULATED:
                cx, cy = j.x, j.y
                # Remove diagonal cross connections that would accidentally short the insulated bridge
                if G.has_edge((cx - 1, cy), (cx, cy - 1)):
                    G.remove_edge((cx - 1, cy), (cx, cy - 1))
                if G.has_edge((cx + 1, cy), (cx, cy + 1)):
                    G.remove_edge((cx + 1, cy), (cx, cy + 1))

        # Find connected wire components (equipotential electrical nodes)
        connected_components = list(nx.connected_components(G))

        # Map each graph point to its connected component cluster index
        point_to_cluster: Dict[Tuple[int, int], int] = {}
        for cluster_idx, cluster in enumerate(connected_components):
            for pt in cluster:
                point_to_cluster[pt] = cluster_idx

        # Map each snapped terminal to its cluster index
        terminal_to_cluster: Dict[Tuple[str, str], int] = {}
        for comp, term, snapped_pt, _ in snap_matches:
            cluster_id = point_to_cluster.get(snapped_pt)
            if cluster_id is None and point_to_cluster:
                # Find closest graph pixel
                closest_pt = min(
                    point_to_cluster.keys(),
                    key=lambda p: (p[0] - snapped_pt[0]) ** 2 + (p[1] - snapped_pt[1]) ** 2,
                )
                cluster_id = point_to_cluster[closest_pt]

            if cluster_id is not None:
                terminal_to_cluster[(comp.id, term.name)] = cluster_id

        # -----------------------------------------------------------------
        # Electrical Node Labeling:
        # Node "0" / "n0" is designated for Ground
        # Remaining nodes: "n1", "n2", "n3", ...
        # -----------------------------------------------------------------
        ground_cluster_idx: Optional[int] = None

        # Look for dedicated Ground component
        for comp in components:
            if comp.class_name == "ground":
                for term in comp.terminals:
                    c_idx = terminal_to_cluster.get((comp.id, term.name))
                    if c_idx is not None:
                        ground_cluster_idx = c_idx
                        break
            if ground_cluster_idx is not None:
                break

        # Fallback for ground: Negative terminal of primary voltage source
        if ground_cluster_idx is None:
            for comp in components:
                if comp.class_name == "voltage_source":
                    for term in comp.terminals:
                        if term.name == "negative":
                            ground_cluster_idx = terminal_to_cluster.get((comp.id, term.name))
                            break
                if ground_cluster_idx is not None:
                    break

        # Fallback to cluster 0 if any clusters exist
        if ground_cluster_idx is None and connected_components:
            ground_cluster_idx = 0

        # Assign clean node labels: "0" for ground, "n1", "n2", ... for others
        cluster_to_node_label: Dict[int, str] = {}
        if ground_cluster_idx is not None:
            cluster_to_node_label[ground_cluster_idx] = "0"

        # Determine Node n1 (prefer positive terminal of voltage source)
        node_1_cluster_idx: Optional[int] = None
        for comp in components:
            if comp.class_name == "voltage_source":
                for term in comp.terminals:
                    if term.name == "positive":
                        c_idx = terminal_to_cluster.get((comp.id, term.name))
                        if c_idx is not None and c_idx != ground_cluster_idx:
                            node_1_cluster_idx = c_idx
                            break
            if node_1_cluster_idx is not None:
                break

        if node_1_cluster_idx is not None:
            cluster_to_node_label[node_1_cluster_idx] = "n1"

        # Label remaining clusters
        next_node_num = 2 if node_1_cluster_idx is not None else 1
        for c_idx in range(len(connected_components)):
            if c_idx not in cluster_to_node_label:
                cluster_to_node_label[c_idx] = f"n{next_node_num}"
                next_node_num += 1

        # Populate connected_node on each terminal and group into node dictionary
        nodes_dict: Dict[str, List[Tuple[DetectedComponent, TerminalPin]]] = {}
        floating_idx = 100

        for comp in components:
            for term in comp.terminals:
                c_idx = terminal_to_cluster.get((comp.id, term.name))
                if c_idx is not None:
                    node_label = cluster_to_node_label.get(c_idx, f"n{c_idx}")
                    term.connected_node = node_label
                else:
                    # Floating / unconnected terminal
                    floating_idx += 1
                    term.connected_node = f"FLOAT_{floating_idx}"

                if term.connected_node not in nodes_dict:
                    nodes_dict[term.connected_node] = []
                nodes_dict[term.connected_node].append((comp, term))

        return nodes_dict, G

    def generate_spice_netlist(
        self,
        components: List[DetectedComponent],
    ) -> str:
        """
        Assemble standard SPICE netlist string (.cir format) compatible with ngspice/PySpice.
        """
        lines: List[str] = [
            "* DrishtiVani Accessible Circuit Netlist",
            "* Generated by Diagram Engine (Member 2)",
            "* --------------------------------------",
        ]

        # Sources first, then passive components
        sorted_comps = sorted(
            components,
            key=lambda c: 0 if c.class_name in ("voltage_source", "current_source") else 1,
        )

        for comp in sorted_comps:
            if comp.class_name == "ground":
                # Ground is represented by node 0 in SPICE, no explicit component card
                continue

            term_nodes = [t.connected_node or "0" for t in comp.terminals]

            if comp.class_name == "resistor":
                n1 = term_nodes[0] if len(term_nodes) > 0 else "0"
                n2 = term_nodes[1] if len(term_nodes) > 1 else "0"
                val = comp.value or "1k"
                lines.append(f"{comp.id} {n1} {n2} {val}")

            elif comp.class_name == "capacitor":
                n1 = term_nodes[0] if len(term_nodes) > 0 else "0"
                n2 = term_nodes[1] if len(term_nodes) > 1 else "0"
                val = comp.value or "10u"
                lines.append(f"{comp.id} {n1} {n2} {val}")

            elif comp.class_name == "inductor":
                n1 = term_nodes[0] if len(term_nodes) > 0 else "0"
                n2 = term_nodes[1] if len(term_nodes) > 1 else "0"
                val = comp.value or "1m"
                lines.append(f"{comp.id} {n1} {n2} {val}")

            elif comp.class_name == "voltage_source":
                pos = "n1"
                neg = "0"
                for t in comp.terminals:
                    if t.name == "positive":
                        pos = t.connected_node or "n1"
                    elif t.name == "negative":
                        neg = t.connected_node or "0"
                if len(term_nodes) >= 2 and pos == "n1" and neg == "0":
                    pos, neg = term_nodes[0], term_nodes[1]
                val = comp.value or "10V"
                val_str = val if " " in val else f"DC {val}"
                lines.append(f"{comp.id} {pos} {neg} {val_str}")

            elif comp.class_name == "current_source":
                pos = term_nodes[0] if len(term_nodes) > 0 else "n1"
                neg = term_nodes[1] if len(term_nodes) > 1 else "0"
                val = comp.value or "1A"
                val_str = val if " " in val else f"DC {val}"
                lines.append(f"{comp.id} {pos} {neg} {val_str}")

            elif comp.class_name == "diode":
                anode = term_nodes[0] if len(term_nodes) > 0 else "n1"
                cathode = term_nodes[1] if len(term_nodes) > 1 else "0"
                model = comp.value or "D1N4148"
                lines.append(f"{comp.id} {anode} {cathode} {model}")

            elif comp.class_name == "transistor_bjt":
                c = term_nodes[1] if len(term_nodes) > 1 else "n1"
                b = term_nodes[0] if len(term_nodes) > 0 else "n2"
                e = term_nodes[2] if len(term_nodes) > 2 else "0"
                model = comp.value or "2N2222"
                lines.append(f"{comp.id} {c} {b} {e} {model}")

            elif comp.class_name == "opamp":
                inv = term_nodes[0] if len(term_nodes) > 0 else "n1"
                noninv = term_nodes[1] if len(term_nodes) > 1 else "n2"
                out = term_nodes[2] if len(term_nodes) > 2 else "n3"
                lines.append(f"{comp.id} {noninv} {inv} {out} LM741")

            else:
                n1 = term_nodes[0] if len(term_nodes) > 0 else "n1"
                n2 = term_nodes[1] if len(term_nodes) > 1 else "0"
                lines.append(f"{comp.id} {n1} {n2} 1k")

        lines.append(".end")
        return "\n".join(lines)

    def evaluate_confidence(
        self,
        components: List[DetectedComponent],
        snap_matches: List[Tuple[DetectedComponent, TerminalPin, Tuple[int, int], float]],
        used_endpoints: Set[Tuple[int, int]],
        all_endpoints: List[Tuple[int, int]],
    ) -> Tuple[float, bool, List[str], List[Tuple[int, int]]]:
        """
        Confidence Scoring Logic:
        - Baseline confidence derived from YOLO detection scores (mean detection confidence).
        - Subtracts a -0.15 penalty for each unconnected component terminal.
        - Subtracts a -0.15 penalty for each dangling wire endpoint.
        - If confidence < 0.88: sets needs_teacher_review = True.
        """
        # Baseline confidence
        if not components:
            return 0.0, True, ["No components detected in diagram."], []

        yolo_confs = [c.confidence for c in components]
        base_confidence = float(np.mean(yolo_confs))

        # Check for unconnected component terminals
        unconnected_terminals: List[str] = []
        for comp in components:
            for term in comp.terminals:
                if not term.is_snapped:
                    unconnected_terminals.append(f"{comp.id}:{term.name}")

        # Check for genuine dangling wire endpoints:
        # A dangling endpoint is an endpoint of an active wire trace (length >= 25px)
        # with Degree == 1 that terminates in free space without snapping to any component pin.
        dangling_endpoints: List[Tuple[int, int]] = []
        for ep in all_endpoints:
            # Check if this endpoint is within snap distance of any component terminal
            near_terminal = any(
                math.hypot(ep[0] - t.location[0], ep[1] - t.location[1]) <= (self.snap_threshold_px + 3)
                for comp in components
                for t in comp.terminals
            )
            if not near_terminal and ep not in used_endpoints:
                dangling_endpoints.append(ep)

        # Apply -0.15 penalty per fault
        fault_count = len(unconnected_terminals) + len(dangling_endpoints)
        penalty = self.unconnected_penalty * fault_count

        computed_score = base_confidence - penalty

        # Enforce strict ceiling if any unconnected terminal or dangling wire exists
        if fault_count > 0:
            computed_score = min(computed_score, self.review_threshold - 0.03)

        final_score = round(max(0.05, min(1.0, computed_score)), 2)
        needs_review = final_score < self.review_threshold

        return final_score, needs_review, unconnected_terminals, dangling_endpoints

    def build_accessible_nodes_list(
        self,
        nodes_dict: Dict[str, List[Tuple[DetectedComponent, TerminalPin]]],
    ) -> List[ElectricalNodeInfo]:
        """
        Assemble the hierarchical screen-reader accessible node list matching the exact JSON specification.
        """
        nodes_list: List[ElectricalNodeInfo] = []

        # Sort: "0" first, then "n1", "n2", ...
        sorted_keys = sorted(
            nodes_dict.keys(),
            key=lambda k: (0 if k in ("0", "n0") else 1, int(k[1:]) if k.startswith("n") and k[1:].isdigit() else 999, k),
        )

        for node_id in sorted_keys:
            comp_tuples = nodes_dict[node_id]
            conn_refs: List[ConnectedComponentRef] = []
            conn_descs: List[str] = []

            for comp, term in comp_tuples:
                conn_refs.append(
                    ConnectedComponentRef(
                        component_id=comp.id,
                        class_name=comp.class_name,
                        terminal=term.name,
                    )
                )
                conn_descs.append(f"{term.name} of {comp.id} ({comp.class_name})")

            # Format natural-language description
            if len(conn_descs) >= 2:
                conns_text = " and ".join([", ".join(conn_descs[:-1]), conn_descs[-1]]) if len(conn_descs) > 2 else f"{conn_descs[0]} to {conn_descs[1]}"
                desc = f"Node {node_id} connects {conns_text}."
            elif len(conn_descs) == 1:
                desc = f"Node {node_id} connects {conn_descs[0]}."
            else:
                desc = f"Node {node_id} is an isolated conductive trace."

            nodes_list.append(
                ElectricalNodeInfo(
                    node_id=node_id,
                    description=desc,
                    connected_components=conn_refs,
                )
            )

        return nodes_list

    def generate_summary(
        self,
        components: List[DetectedComponent],
        nodes_list: List[ElectricalNodeInfo],
    ) -> str:
        """Generate a plain-English macro summary of the circuit topology and component count."""
        non_gnd_comps = [c for c in components if c.class_name != "ground"]
        comp_count = len(non_gnd_comps)
        node_count = len(nodes_list)

        # Count types
        type_counts: Dict[str, int] = {}
        for c in non_gnd_comps:
            type_counts[c.class_name] = type_counts.get(c.class_name, 0) + 1

        type_strs = [f"{count} {name.replace('_', ' ')}{'s' if count > 1 else ''}" for name, count in type_counts.items()]
        types_breakdown = ", ".join(type_strs) if type_strs else "no active components"

        has_ground = any(n.node_id in ("0", "n0") for n in nodes_list)
        gnd_str = "ground reference present" if has_ground else "no explicit ground reference"

        return (
            f"Circuit contains {comp_count} components ({types_breakdown}) "
            f"connected across {node_count} electrical nodes with {gnd_str}."
        )

    def process(
        self,
        components: List[DetectedComponent],
        junctions: List[JunctionPoint],
        wire_segments: List[WireSegment],
        skeleton: Optional[np.ndarray] = None,
    ) -> NetlistPipelineResult:
        """
        Execute full Step 3 topological graph assembly, netlist generation,
        and confidence evaluation.
        """
        # 1. Snap component terminals to Degree == 1 wire endpoints
        snap_matches, used_endpoints = self.snap_terminals_to_endpoints(
            components=components,
            junctions=junctions,
            wire_segments=wire_segments,
            skeleton=skeleton,
        )

        # 2. Build graph and group electrical nodes
        nodes_dict, _ = self.build_circuit_graph(
            wire_segments=wire_segments,
            junctions=junctions,
            snap_matches=snap_matches,
            components=components,
            skeleton=skeleton,
        )

        # 3. Generate SPICE netlist (.cir)
        spice_netlist = self.generate_spice_netlist(components)

        # 4. Gather genuine degree-1 endpoints on wire traces for dangling wire check
        deg1_coords = {(j.x, j.y) for j in junctions if j.junction_type == JunctionClass.ENDPOINT}
        all_endpoints: List[Tuple[int, int]] = []
        for seg in wire_segments:
            seg_len = seg.length_px or math.hypot(seg.endpoint_a[0] - seg.endpoint_b[0], seg.endpoint_a[1] - seg.endpoint_b[1])
            if seg_len >= 15:
                for ep in (seg.endpoint_a, seg.endpoint_b):
                    if any(math.hypot(ep[0] - jx, ep[1] - jy) <= 5 for jx, jy in deg1_coords):
                        if ep not in all_endpoints:
                            all_endpoints.append(ep)
        # Also include any standalone degree-1 junctions directly
        for j in junctions:
            if j.junction_type == JunctionClass.ENDPOINT and (j.x, j.y) not in all_endpoints:
                all_endpoints.append((j.x, j.y))

        # 5. Evaluate confidence score with -0.15 penalties
        conf_score, needs_review, unconnected_terms, dangling_eps = self.evaluate_confidence(
            components=components,
            snap_matches=snap_matches,
            used_endpoints=used_endpoints,
            all_endpoints=all_endpoints,
        )

        # 6. Build accessible nodes list
        nodes_list = self.build_accessible_nodes_list(nodes_dict)

        # 7. Generate macro summary
        summary = self.generate_summary(components, nodes_list)

        return NetlistPipelineResult(
            spice_netlist=spice_netlist,
            nodes=nodes_list,
            confidence_score=conf_score,
            needs_teacher_review=needs_review,
            summary=summary,
            unconnected_terminals=unconnected_terms,
            dangling_endpoints=dangling_eps,
        )
