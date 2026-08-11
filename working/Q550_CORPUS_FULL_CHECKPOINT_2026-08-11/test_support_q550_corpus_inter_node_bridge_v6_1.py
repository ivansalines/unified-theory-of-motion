#!/usr/bin/env python3
"""Q=550 corpus inter-node bridge (v6).

Purpose
-------
The v5 dense-start audit rigorously certified all selected dense nodes through
endpoint nesting

    T_2P(s) = G_edge(s) T_O(s),
    D(s)    = T_O(s) - T_2P(s) = C(s) T(s),
    C(s)    = I - G_edge(s),
    T(s)    = T_O(s).

This v6 pilot attacks the still-open continuous *inter-node* coordinate without
subtracting two independently uncertain long transports.  Instead it evolves
exactly the two corpus factors in the start coordinate s:

    T' = B1 T - T B0,
    C' = (B1 - B2) + B2 C - C B1,

where

    B0 = B(s),
    B1 = B(s + P_oriented),
    B2 = B(s + 2 P_line).

The forcing B1-B2 is enclosed in a *paired* way on the same start subinterval,
so the tiny endpoint separation DeltaP = 2 P_line - P_oriented is not destroyed
by subtracting two coarse independent generator hulls.

Each dense cell is covered by two independently anchored half-cells: one grows
forward from the left rigorous v5 node, the other backward from the right v5
node.  Inside each half-cell, a validated Euler/Picard tube is propagated in
spectral-norm balls.  At every tube step the active endpoint difference is
certified from the product C*T.  No uncertainty is chained from one dense cell
to the next.

Default mode is a diagnostic pilot on automatically selected hard cells.  Use
--all-cells after the pilot passes to cover all 80 dense cells.

Scope
-----
Frozen natural-cubic-spline linear cocycle only.  This script is designed to
close continuous start-coordinate coverage of the endpoint distinction.  It
does not interval-validate the nonlinear orbit cache against an exact nonlinear
trajectory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


SCRIPT_VERSION = "2026-08-10-q550-corpus-inter-node-bridge-v6.1-full-path-scaling"
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_4.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_V5_ROWS = Path("support_q550_endpoint_nesting_dense_span_0_80_rows.csv")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_corpus_inter_node_bridge_v6")


@dataclass
class MatrixBall:
    center: np.ndarray
    radius: float


@dataclass
class Anchor:
    index: int
    start: float
    t: MatrixBall
    c: MatrixBall
    recorded_lower: float
    recorded_sep: float
    center_alignment: float


def load_module(path: Path, name: str) -> ModuleType:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def outward(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def downward(x: float) -> float:
    return float(np.nextafter(float(x), -np.inf))


def n2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float), ord=2))


def n2_up(a: np.ndarray) -> float:
    return outward(n2(a))


def parse_cells(text: str) -> list[int]:
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    return values


def read_v5_rows(path: Path, subdivision: int) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if int(raw["residual_subdivisions"]) != subdivision:
                continue
            idx = int(raw["start_index"])
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key in ("endpoint_pass",):
                    row[key] = 1.0 if value.strip().lower() == "true" else 0.0
                elif key in ("start_index", "residual_subdivisions"):
                    row[key] = float(int(value))
                else:
                    row[key] = float(value)
            rows[idx] = row
    if not rows:
        raise RuntimeError(
            f"No v5 rows at residual_subdivisions={subdivision} in {path}"
        )
    failed = [idx for idx, row in rows.items() if not bool(row["endpoint_pass"])]
    if failed:
        raise RuntimeError(f"Input v5 certificate contains failed nodes: {failed}")
    return rows


def guide_transport_center(
    envelope: Any,
    *,
    start: float,
    step: float,
    steps: int,
) -> np.ndarray:
    """Reproduce the RK4 guide center used by v5, without interval residual work."""
    center = np.eye(9, dtype=float)
    for index in range(steps):
        s0 = start + index * step
        sm = s0 + 0.5 * step
        s1 = s0 + step
        b0 = envelope.point(s0)
        bm = envelope.point(sm)
        b1 = envelope.point(s1)
        y0 = center
        k1 = b0 @ y0
        k2 = bm @ (y0 + 0.5 * step * k1)
        k3 = bm @ (y0 + 0.5 * step * k2)
        k4 = b1 @ (y0 + step * k3)
        center = y0 + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return center


def product_radius(a: MatrixBall, b: MatrixBall) -> float:
    """Spectral radius for (A+dA)(B+dB) around A B."""
    return outward(
        n2_up(a.center) * b.radius
        + a.radius * n2_up(b.center)
        + a.radius * b.radius
    )


def active_product_lower(c: MatrixBall, t: MatrixBall) -> tuple[float, float, float]:
    center = c.center @ t.center
    sep = n2(center[:2, :2])
    radius = product_radius(c, t)
    return sep, radius, downward(sep - radius)


def roundoff_budget(*scales: float, dimension: int = 9, factor: float = 4096.0) -> float:
    eps = np.finfo(float).eps
    scale = 1.0 + sum(abs(float(x)) for x in scales)
    return outward(factor * eps * max(1, dimension) * scale)


def generator_radius_about(
    base: ModuleType,
    pilot: ModuleType,
    box: Any,
    center: np.ndarray,
) -> float:
    delta = base.isub(box, base.exact_box(np.asarray(center, dtype=float)))
    return outward(pilot.box_spectral_upper(base, delta))


def paired_generator_difference_box(
    base: ModuleType,
    envelope: Any,
    *,
    left: float,
    right: float,
    p_oriented: float,
    p_two_line: float,
    subdivisions: int,
) -> Any:
    """Enclose B(s+P_o)-B(s+2P) while retaining the shared s coordinate."""
    if subdivisions < 1:
        raise ValueError("paired subdivisions must be positive")
    pieces: list[Any] = []
    for j in range(subdivisions):
        a = left + (right - left) * (j / subdivisions)
        b = left + (right - left) * ((j + 1) / subdivisions)
        b1 = envelope.box(a + p_oriented, b + p_oriented)
        b2 = envelope.box(a + p_two_line, b + p_two_line)
        pieces.append(base.isub(b1, b2))
    return base.hull(pieces)


def derivative_ball_radii(
    *,
    t_center: np.ndarray,
    c_center: np.ndarray,
    t_tube_radius: float,
    c_tube_radius: float,
    b0_center: np.ndarray,
    b1_center: np.ndarray,
    b2_center: np.ndarray,
    b0_radius: float,
    b1_radius: float,
    b2_radius: float,
    q_radius: float,
) -> tuple[float, float]:
    """Deviation radii around the chosen point-center corpus derivatives."""
    nt = n2_up(t_center)
    nc = n2_up(c_center)
    nb0 = n2_up(b0_center)
    nb1 = n2_up(b1_center)
    nb2 = n2_up(b2_center)

    rt = outward(
        nb1 * t_tube_radius
        + b1_radius * nt
        + b1_radius * t_tube_radius
        + nb0 * t_tube_radius
        + b0_radius * nt
        + b0_radius * t_tube_radius
    )
    rc = outward(
        q_radius
        + nb2 * c_tube_radius
        + b2_radius * nc
        + b2_radius * c_tube_radius
        + nb1 * c_tube_radius
        + b1_radius * nc
        + b1_radius * c_tube_radius
    )
    return rt, rc


def validate_corpus_step(
    base: ModuleType,
    pilot: ModuleType,
    envelope: Any,
    *,
    physical_left: float,
    physical_right: float,
    direction: int,
    t: MatrixBall,
    c: MatrixBall,
    p_oriented: float,
    p_two_line: float,
    paired_subdivisions: int,
    picard_iterations: int,
    picard_rtol: float,
) -> tuple[MatrixBall, MatrixBall, dict[str, float]]:
    """Validated one-step tube in the start coordinate.

    ``direction`` is +1 for left->right and -1 for right->left.  The physical
    generator interval is always [physical_left, physical_right].
    """
    if not physical_right > physical_left:
        raise ValueError("empty corpus step")
    if direction not in (-1, +1):
        raise ValueError("direction must be +/-1")
    h = physical_right - physical_left
    mid = 0.5 * (physical_left + physical_right)

    b0_box = envelope.box(physical_left, physical_right)
    b1_box = envelope.box(
        physical_left + p_oriented, physical_right + p_oriented
    )
    b2_box = envelope.box(
        physical_left + p_two_line, physical_right + p_two_line
    )
    q_box = paired_generator_difference_box(
        base,
        envelope,
        left=physical_left,
        right=physical_right,
        p_oriented=p_oriented,
        p_two_line=p_two_line,
        subdivisions=paired_subdivisions,
    )

    b0 = envelope.point(mid)
    b1 = envelope.point(mid + p_oriented)
    b2 = envelope.point(mid + p_two_line)
    q = b1 - b2

    rb0 = generator_radius_about(base, pilot, b0_box, b0)
    rb1 = generator_radius_about(base, pilot, b1_box, b1)
    rb2 = generator_radius_about(base, pilot, b2_box, b2)
    rq = generator_radius_about(base, pilot, q_box, q)

    f_t = direction * (b1 @ t.center - t.center @ b0)
    f_c = direction * (q + b2 @ c.center - c.center @ b1)
    nft = n2_up(f_t)
    nfc = n2_up(f_c)

    # Account conservatively for floating products used only as guide centers.
    f_t_round = roundoff_budget(
        n2_up(b1) * n2_up(t.center), n2_up(t.center) * n2_up(b0)
    )
    f_c_round = roundoff_budget(
        n2_up(q), n2_up(b2) * n2_up(c.center), n2_up(c.center) * n2_up(b1)
    )

    rtube = outward(t.radius + h * (nft + f_t_round))
    ctube = outward(c.radius + h * (nfc + f_c_round))
    converged = False
    final_rft = math.inf
    final_rfc = math.inf

    for iteration in range(1, picard_iterations + 1):
        rft, rfc = derivative_ball_radii(
            t_center=t.center,
            c_center=c.center,
            t_tube_radius=rtube,
            c_tube_radius=ctube,
            b0_center=b0,
            b1_center=b1,
            b2_center=b2,
            b0_radius=rb0,
            b1_radius=rb1,
            b2_radius=rb2,
            q_radius=rq,
        )
        rft = outward(rft + f_t_round)
        rfc = outward(rfc + f_c_round)
        new_rtube = outward(t.radius + h * (nft + rft))
        new_ctube = outward(c.radius + h * (nfc + rfc))

        tol_t = max(1e-15, picard_rtol * max(new_rtube, rtube, 1e-300))
        tol_c = max(1e-15, picard_rtol * max(new_ctube, ctube, 1e-300))
        if new_rtube <= rtube + tol_t and new_ctube <= ctube + tol_c:
            rtube = max(rtube, new_rtube)
            ctube = max(ctube, new_ctube)
            final_rft = rft
            final_rfc = rfc
            converged = True
            break

        rtube = outward(max(rtube, new_rtube))
        ctube = outward(max(ctube, new_ctube))
        final_rft = rft
        final_rfc = rfc

        if not np.isfinite(rtube + ctube) or rtube > 1e6 or ctube > 1e6:
            break

    if not converged:
        raise RuntimeError(
            "Picard tube did not close: "
            f"h={h:.3e} rtube={rtube:.3e} ctube={ctube:.3e} "
            f"after {picard_iterations} iterations"
        )

    tube_t = MatrixBall(t.center, rtube)
    tube_c = MatrixBall(c.center, ctube)
    sep, active_radius, lower = active_product_lower(tube_c, tube_t)

    # Recenter the endpoint at the midpoint-generator Euler guide.  Since the
    # full derivative differs from f_center by final_rF throughout the tube,
    # the endpoint error is r_start + h*rF (plus explicit roundoff budget).
    t_next_center = t.center + h * f_t
    c_next_center = c.center + h * f_c
    t_update_round = roundoff_budget(n2_up(t.center), h * nft)
    c_update_round = roundoff_budget(n2_up(c.center), h * nfc)
    t_next_radius = outward(t.radius + h * final_rft + t_update_round)
    c_next_radius = outward(c.radius + h * final_rfc + c_update_round)

    diag = {
        "h": h,
        "rb0": rb0,
        "rb1": rb1,
        "rb2": rb2,
        "rq": rq,
        "fT_center_norm": nft,
        "fC_center_norm": nfc,
        "T_tube_radius": rtube,
        "C_tube_radius": ctube,
        "T_endpoint_radius": t_next_radius,
        "C_endpoint_radius": c_next_radius,
        "active_center_separation": sep,
        "active_product_radius": active_radius,
        "active_lower": lower,
        "picard_iterations_used": float(iteration),
    }
    return (
        MatrixBall(t_next_center, t_next_radius),
        MatrixBall(c_next_center, c_next_radius),
        diag,
    )


def validate_half_cell(
    base: ModuleType,
    pilot: ModuleType,
    envelope: Any,
    *,
    cell_index: int,
    side: str,
    anchor: Anchor,
    target: float,
    p_oriented: float,
    p_two_line: float,
    tube_steps: int,
    paired_subdivisions: int,
    picard_iterations: int,
    picard_rtol: float,
) -> tuple[bool, float, list[dict[str, Any]]]:
    if side not in ("left", "right"):
        raise ValueError("side must be left/right")
    direction = +1 if target > anchor.start else -1
    edges = np.linspace(anchor.start, target, tube_steps + 1)
    t = MatrixBall(anchor.t.center.copy(), anchor.t.radius)
    c = MatrixBall(anchor.c.center.copy(), anchor.c.radius)
    rows: list[dict[str, Any]] = []
    minimum = math.inf

    for k in range(tube_steps):
        a = float(edges[k])
        b = float(edges[k + 1])
        physical_left = min(a, b)
        physical_right = max(a, b)
        t, c, diag = validate_corpus_step(
            base,
            pilot,
            envelope,
            physical_left=physical_left,
            physical_right=physical_right,
            direction=direction,
            t=t,
            c=c,
            p_oriented=p_oriented,
            p_two_line=p_two_line,
            paired_subdivisions=paired_subdivisions,
            picard_iterations=picard_iterations,
            picard_rtol=picard_rtol,
        )
        minimum = min(minimum, diag["active_lower"])
        rows.append(
            {
                "cell_index": cell_index,
                "side": side,
                "step_index": k,
                "physical_left": physical_left,
                "physical_right": physical_right,
                "direction": direction,
                **diag,
                "step_pass": bool(diag["active_lower"] > 0.0),
            }
        )
        # v6.1 diagnostic mode: do NOT stop at the first loss of positive
        # separation.  The corpus ODE enclosure itself remains valid, so we
        # continue to the half-cell midpoint.  This makes scaling runs at
        # different paired_subdivisions directly comparable at identical
        # physical locations.

    passed = all(bool(r["step_pass"]) for r in rows)
    return passed, minimum, rows


def choose_pilot_cells(
    starts: np.ndarray,
    archived_oriented: np.ndarray,
    archived_two_line: np.ndarray,
    v5_rows: dict[int, dict[str, float]],
) -> tuple[list[int], dict[str, int]]:
    indices = sorted(v5_rows)
    worst_lower_node = min(indices, key=lambda i: v5_rows[i]["endpoint_separation_lower"])
    worst_radius_node = max(indices, key=lambda i: v5_rows[i]["active_difference_radius"])

    d = archived_oriented[:, :2, :2] - archived_two_line[:, :2, :2]
    slopes = []
    for i in range(len(starts) - 1):
        h = starts[i + 1] - starts[i]
        slopes.append(n2((d[i + 1] - d[i]) / h))
    max_slope_cell = int(np.argmax(slopes))

    cells = {0, len(starts) - 2, max_slope_cell}
    for node in (worst_lower_node, worst_radius_node):
        if node > 0:
            cells.add(node - 1)
        if node < len(starts) - 1:
            cells.add(node)
    cells = {i for i in cells if 0 <= i < len(starts) - 1}
    reasons = {
        "worst_v5_lower_node": int(worst_lower_node),
        "worst_v5_radius_node": int(worst_radius_node),
        "largest_archived_D_finite_slope_cell": int(max_slope_cell),
    }
    return sorted(cells), reasons


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_self_test() -> None:
    rng = np.random.default_rng(20260810)
    for _ in range(1000):
        a = rng.normal(size=(4, 4))
        b = rng.normal(size=(4, 4))
        ra = 1e-3 * (1.0 + rng.random())
        rb = 1e-3 * (1.0 + rng.random())
        ball_a = MatrixBall(a, ra)
        ball_b = MatrixBall(b, rb)
        bound = product_radius(ball_a, ball_b)
        for _j in range(10):
            da = rng.normal(size=(4, 4))
            db = rng.normal(size=(4, 4))
            da *= ra / max(n2(da), 1e-300) * rng.random()
            db *= rb / max(n2(db), 1e-300) * rng.random()
            actual = n2((a + da) @ (b + db) - a @ b)
            if actual > bound * (1.0 + 1e-11):
                raise AssertionError(f"product ball escaped: {actual} > {bound}")
    print("corpus spectral-ball algebra self-test passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--pilot-script", type=Path, default=DEFAULT_PILOT)
    p.add_argument("--dense-raw", type=Path, default=DEFAULT_DENSE)
    p.add_argument("--v5-rows", type=Path, default=DEFAULT_V5_ROWS)
    p.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT)
    p.add_argument("--q", type=float, default=550.0)
    p.add_argument("--v5-subdivision", type=int, default=32)
    p.add_argument("--body-steps", type=int, default=1308)
    p.add_argument("--cap-steps", type=int, default=16)
    p.add_argument("--tube-steps", type=int, default=16,
                   help="validated corpus steps per half dense cell")
    p.add_argument("--paired-subdivisions", type=int, default=4,
                   help="paired B1-B2 pieces inside each corpus step")
    p.add_argument("--picard-iterations", type=int, default=40)
    p.add_argument("--picard-rtol", type=float, default=1e-12)
    p.add_argument("--cells", default="",
                   help="comma-separated cell indices; overrides automatic pilot selection")
    p.add_argument("--all-cells", action="store_true")
    p.add_argument("--tail-periods", type=int, default=70)
    p.add_argument("--steps-per-period", type=int, default=160)
    p.add_argument("--samples-per-rotation", type=int, default=720)
    p.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    p.add_argument("--gamma-rho", type=float, default=0.3)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.tube_steps < 1 or args.paired_subdivisions < 1:
        p.error("tube and paired subdivisions must be positive")

    base = load_module(args.base_script, "corpus_v6_base")
    pilot = load_module(args.pilot_script, "corpus_v6_pilot")
    audit = load_module(args.audit_script, "corpus_v6_audit")

    with np.load(args.dense_raw, allow_pickle=True) as d:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])
        archived_oriented = np.asarray(d["fine_oriented"], dtype=float)
        archived_two_line = np.asarray(d["fine_two_line"], dtype=float)

    p_two_line = 2.0 * p_line
    cap_length = p_two_line - p_oriented
    v5_rows = read_v5_rows(args.v5_rows, args.v5_subdivision)
    if set(v5_rows) != set(range(starts.size)):
        missing = sorted(set(range(starts.size)) - set(v5_rows))
        raise RuntimeError(f"v5 certificate does not cover all dense starts; missing={missing}")

    auto_cells, pilot_reasons = choose_pilot_cells(
        starts, archived_oriented, archived_two_line, v5_rows
    )
    if args.all_cells:
        cells = list(range(starts.size - 1))
        selection_mode = "all cells"
    elif args.cells.strip():
        cells = parse_cells(args.cells)
        selection_mode = "explicit cells"
    else:
        cells = auto_cells
        selection_mode = "automatic hard-cell pilot"
    if not cells or min(cells) < 0 or max(cells) >= starts.size - 1:
        p.error(f"invalid cell selection for {starts.size - 1} cells: {cells}")

    params = audit.ModelParameters(gamma_rho=args.gamma_rho)
    orbit = audit.load_orbit(args.q, args.orbit)
    tail = audit.collect_uniform_tail(
        orbit,
        tail_periods=args.tail_periods,
        steps_per_period=args.steps_per_period,
        samples_per_rotation=args.samples_per_rotation,
        initial_x_perturbation=args.initial_x_perturbation,
        params=params,
        progress_every=100,
    )
    spline = audit.build_state_spline(tail)
    inverse, inverse_box = base.mp_inverse_enclosure(basis)
    basis_box = base.constant_matrix_box(basis)
    kappa0 = (55.0 / 6.0) * (10.0 / args.q)

    domain_left = float(starts[min(cells)])
    domain_right = float(starts[max(cells) + 1] + p_two_line)
    envelope = pilot.LocalCubicGeneratorEnvelope(
        base,
        audit,
        spline,
        np.asarray(tail.s, dtype=float),
        domain_left=domain_left,
        domain_right=domain_right,
        kappa0=kappa0,
        params=params,
        basis=basis,
        basis_box=basis_box,
        left_matrix=inverse,
        left_box=inverse_box,
    )

    body_step = p_oriented / args.body_steps
    cap_step = cap_length / args.cap_steps
    anchor_cache: dict[int, Anchor] = {}

    def get_anchor(idx: int) -> Anchor:
        cached = anchor_cache.get(idx)
        if cached is not None:
            return cached
        row = v5_rows[idx]
        s = float(starts[idx])
        tc = guide_transport_center(
            envelope, start=s, step=body_step, steps=args.body_steps
        )
        gc = guide_transport_center(
            envelope,
            start=s + p_oriented,
            step=cap_step,
            steps=args.cap_steps,
        )
        cc = np.eye(9) - gc
        rt = float(row["body_absolute_radius"])
        rc = float(row["edge_absolute_radius"])

        # Strong center-alignment checks against the consumed v5 certificate.
        sep = n2((cc @ tc)[:2, :2])
        recorded_sep = float(row["propagated_center_separation"])
        align = abs(sep - recorded_sep)
        rt_rebuilt = n2(tc) * float(row["body_relative_eps"])
        rc_rebuilt = n2(gc) * float(row["edge_relative_eps"])
        if align > 5e-10:
            raise RuntimeError(
                f"anchor {idx}: recomputed center not aligned with v5: {align:.3e}"
            )
        if abs(rt_rebuilt - rt) > 5e-10 * max(1.0, rt):
            raise RuntimeError(
                f"anchor {idx}: body radius alignment failed: "
                f"rebuilt={rt_rebuilt:.16e} recorded={rt:.16e}"
            )
        if abs(rc_rebuilt - rc) > 5e-10 * max(1.0, rc):
            raise RuntimeError(
                f"anchor {idx}: edge radius alignment failed: "
                f"rebuilt={rc_rebuilt:.16e} recorded={rc:.16e}"
            )
        anchor = Anchor(
            index=idx,
            start=s,
            t=MatrixBall(tc, outward(rt)),
            c=MatrixBall(cc, outward(rc)),
            recorded_lower=float(row["endpoint_separation_lower"]),
            recorded_sep=recorded_sep,
            center_alignment=align,
        )
        anchor_cache[idx] = anchor
        return anchor

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"Q                    = {args.q:g}")
    print(f"dense starts         = {starts.size}")
    print(f"dense cells          = {starts.size - 1}")
    print(f"selection mode       = {selection_mode}")
    print(f"selected cells       = {cells}")
    print(f"pilot reasons        = {pilot_reasons}")
    print(f"P_oriented           = {p_oriented:.15e}")
    print(f"2 P_line             = {p_two_line:.15e}")
    print(f"edge cap DeltaP      = {cap_length:.15e}")
    print(f"v5 subdivision       = {args.v5_subdivision}")
    print(f"tube steps / half    = {args.tube_steps}")
    print(f"paired B1-B2 pieces  = {args.paired_subdivisions}")
    print("corpus factorization = D=C*T, C=I-G_edge")
    print("start ODE T           = B1*T - T*B0")
    print("start ODE C           = (B1-B2) + B2*C - C*B1")
    print("=" * 78, flush=True)

    step_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for count, cell in enumerate(cells, start=1):
        left_anchor = get_anchor(cell)
        right_anchor = get_anchor(cell + 1)
        midpoint = 0.5 * (left_anchor.start + right_anchor.start)

        left_pass, left_min, left_rows = validate_half_cell(
            base,
            pilot,
            envelope,
            cell_index=cell,
            side="left",
            anchor=left_anchor,
            target=midpoint,
            p_oriented=p_oriented,
            p_two_line=p_two_line,
            tube_steps=args.tube_steps,
            paired_subdivisions=args.paired_subdivisions,
            picard_iterations=args.picard_iterations,
            picard_rtol=args.picard_rtol,
        )
        right_pass, right_min, right_rows = validate_half_cell(
            base,
            pilot,
            envelope,
            cell_index=cell,
            side="right",
            anchor=right_anchor,
            target=midpoint,
            p_oriented=p_oriented,
            p_two_line=p_two_line,
            tube_steps=args.tube_steps,
            paired_subdivisions=args.paired_subdivisions,
            picard_iterations=args.picard_iterations,
            picard_rtol=args.picard_rtol,
        )
        step_rows.extend(left_rows)
        step_rows.extend(right_rows)
        minimum = min(left_min, right_min)
        passed = bool(left_pass and right_pass and minimum > 0.0)
        max_tube_t = max(r["T_tube_radius"] for r in left_rows + right_rows)
        max_tube_c = max(r["C_tube_radius"] for r in left_rows + right_rows)
        max_rq = max(r["rq"] for r in left_rows + right_rows)
        combined_rows = left_rows + right_rows
        max_picard = max(r["picard_iterations_used"] for r in combined_rows)
        first_bad = next((r for r in combined_rows if not r["step_pass"]), None)
        left_bad = next((r for r in left_rows if not r["step_pass"]), None)
        right_bad = next((r for r in right_rows if not r["step_pass"]), None)
        cell_row = {
            "cell_index": cell,
            "start_left": left_anchor.start,
            "start_right": right_anchor.start,
            "cell_width": right_anchor.start - left_anchor.start,
            "midpoint": midpoint,
            "left_anchor_v5_lower": left_anchor.recorded_lower,
            "right_anchor_v5_lower": right_anchor.recorded_lower,
            "left_minimum_continuous_lower": left_min,
            "right_minimum_continuous_lower": right_min,
            "minimum_continuous_lower": minimum,
            "maximum_T_tube_radius": max_tube_t,
            "maximum_C_tube_radius": max_tube_c,
            "maximum_paired_generator_radius": max_rq,
            "maximum_picard_iterations": max_picard,
            "left_steps_completed": len(left_rows),
            "right_steps_completed": len(right_rows),
            "first_bad_side": "" if first_bad is None else first_bad["side"],
            "first_bad_step": -1 if first_bad is None else int(first_bad["step_index"]),
            "first_bad_left_step": -1 if left_bad is None else int(left_bad["step_index"]),
            "first_bad_right_step": -1 if right_bad is None else int(right_bad["step_index"]),
            "cell_pass": passed,
        }
        cell_rows.append(cell_row)
        print(
            f"[{count:>3}/{len(cells)}] cell={cell:>2} "
            f"width={cell_row['cell_width']:.6e} "
            f"left={left_min:+.6e} right={right_min:+.6e} "
            f"min={minimum:+.6e} "
            f"maxRT={max_tube_t:.3e} maxRC={max_tube_c:.3e} "
            f"rq={max_rq:.3e} picard={int(max_picard)} "
            f"badL={cell_row['first_bad_left_step']} "
            f"badR={cell_row['first_bad_right_step']} pass={passed}",
            flush=True,
        )

    passed_count = sum(bool(r["cell_pass"]) for r in cell_rows)
    worst = min(cell_rows, key=lambda r: r["minimum_continuous_lower"])
    classification = (
        "CORPUS INTER-NODE SELECTED CELLS PASSED"
        if passed_count == len(cell_rows)
        else "CORPUS INTER-NODE COVERAGE STILL OPEN"
    )
    if args.all_cells and passed_count == len(cell_rows):
        classification = "CORPUS CONTINUOUS DENSE-SPAN COVERAGE PASSED"

    print("=" * 78)
    print(
        f"summary: passed={passed_count}/{len(cell_rows)} "
        f"minContinuousLower={worst['minimum_continuous_lower']:+.6e} "
        f"at cell={int(worst['cell_index'])}"
    )
    print(f"classification = {classification}")

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(prefix.with_name(prefix.name + "_steps.csv"), step_rows)
    write_csv(prefix.with_name(prefix.name + "_cells.csv"), cell_rows)

    payload = {
        "script_version": SCRIPT_VERSION,
        "q": args.q,
        "scope": "continuous start-coordinate endpoint distinction in frozen-spline linear cocycle",
        "selection_mode": selection_mode,
        "selected_cells": cells,
        "pilot_reasons": pilot_reasons,
        "p_line": p_line,
        "p_oriented": p_oriented,
        "p_two_line": p_two_line,
        "edge_cap_length": cap_length,
        "v5_subdivision": args.v5_subdivision,
        "tube_steps_per_half": args.tube_steps,
        "paired_generator_subdivisions": args.paired_subdivisions,
        "cells_passed": passed_count,
        "cells_tested": len(cell_rows),
        "minimum_continuous_lower": float(worst["minimum_continuous_lower"]),
        "minimum_continuous_lower_cell": int(worst["cell_index"]),
        "classification": classification,
        "cells": cell_rows,
    }
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")

    report = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Q=550 CORPUS INTER-NODE BRIDGE",
        "================================",
        "",
        f"selection mode                    = {selection_mode}",
        f"selected cells                    = {cells}",
        f"P_oriented                        = {p_oriented:.12e}",
        f"2 P_line                          = {p_two_line:.12e}",
        f"edge cap DeltaP                   = {cap_length:.12e}",
        f"v5 anchor subdivision             = {args.v5_subdivision}",
        f"tube steps per half-cell          = {args.tube_steps}",
        f"paired B1-B2 pieces per step      = {args.paired_subdivisions}",
        "",
        "Corpus architecture",
        "-------------------",
        "D = C T, with C = I-G_edge and T=T_O.",
        "T' = B1 T - T B0.",
        "C' = (B1-B2) + B2 C - C B1.",
        "B1-B2 is evaluated on paired start-coordinate subintervals.",
        "Each dense cell is covered from both endpoints; no uncertainty is",
        "chained from one dense cell to the next.",
        "",
        "Cell summaries",
        "--------------",
    ]
    for row in cell_rows:
        report.append(
            f"cell={int(row['cell_index']):>2} "
            f"minLower={row['minimum_continuous_lower']:+.6e} "
            f"maxRT={row['maximum_T_tube_radius']:.6e} "
            f"maxRC={row['maximum_C_tube_radius']:.6e} "
            f"rq={row['maximum_paired_generator_radius']:.6e} "
            f"pass={bool(row['cell_pass'])}"
        )
    report.extend(
        [
            "",
            f"classification = {classification}",
            "",
            "Scope: frozen natural-cubic-spline linear cocycle only; exact",
            "nonlinear-orbit interval validation remains open.",
        ]
    )
    prefix.with_name(prefix.name + "_report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
