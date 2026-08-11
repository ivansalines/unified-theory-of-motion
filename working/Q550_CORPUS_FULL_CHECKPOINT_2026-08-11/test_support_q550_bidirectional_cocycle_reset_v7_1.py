#!/usr/bin/env python3
"""Q=550 bidirectional cocycle-reset corpus bridge (v7.1).

This is the natural continuation of v7.

v7 reconstructs every start x in a cell from the LEFT rigorous v5 anchor:
    T(x) = H1(x) T(a) H0(x)^(-1)
    G(x) = H2(x) G(a) H1(x)^(-1)

The v7 scaling shows the cocycle centers are correct but the short-transport
balls H still accumulate across the whole dense cell.

v7.1 uses BOTH rigorous v5 endpoint anchors.

Left half [a,m]:
    Hk(x) = Phi(x+shift_k, a+shift_k)
    T(x) = H1 T(a) H0^(-1)
    G(x) = H2 G(a) H1^(-1)

Right half [m,b]:
    Kk(x) = Phi(b+shift_k, x+shift_k)   (forward transport x -> b)
    T(x) = K1^(-1) T(b) K0
    G(x) = K2^(-1) G(b) K1

Each local slab is covered with the existing v6.1 one-step Picard tube, but
the endpoint ball is discarded.  The next slab is rebuilt from its nearest
rigorous v5 endpoint anchor through validated short cocycle transports.

If both half-cell covers pass, the whole dense cell is continuously covered.
No uncertainty is chained from one dense cell to another.

Scope: frozen natural-cubic-spline linear cocycle only.  Exact nonlinear-orbit
interval validation remains open.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_VERSION = "2026-08-10-q550-bidirectional-cocycle-reset-corpus-v7.1"

DEFAULT_CORPUS = Path("test_support_q550_corpus_inter_node_bridge_v6_1.py")
DEFAULT_RESET = Path("test_support_q550_cocycle_reset_corpus_bridge_v7.py")
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_4.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_V5_ROWS = Path("support_q550_endpoint_nesting_dense_span_0_80_rows.csv")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_bidirectional_cocycle_reset_v7_1")


def load_module(path: Path, name: str) -> Any:
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


def n2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float), ord=2))


def outward(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def parse_sweep(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or vals[0] < 1:
        raise ValueError("half-step sweep must contain positive integers")
    return vals


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-script", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--reset-script", type=Path, default=DEFAULT_RESET)
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
    p.add_argument("--cells", default="39")
    p.add_argument("--half-step-sweep", default="64,128,256")
    p.add_argument("--transport-residual-subdivisions", type=int, default=8)
    p.add_argument("--paired-subdivisions", type=int, default=8)
    p.add_argument("--picard-iterations", type=int, default=40)
    p.add_argument("--picard-rtol", type=float, default=1e-12)
    p.add_argument("--tail-periods", type=int, default=70)
    p.add_argument("--steps-per-period", type=int, default=160)
    p.add_argument("--samples-per-rotation", type=int, default=720)
    p.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    p.add_argument("--gamma-rho", type=float, default=0.3)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    corpus = load_module(args.corpus_script, "bidir_v71_corpus")
    reset = load_module(args.reset_script, "bidir_v71_reset")

    if args.self_test:
        reset.run_self_test(corpus)
        print("bidirectional cocycle-reset algebra self-test passed")
        return

    base = load_module(args.base_script, "bidir_v71_base")
    pilot = load_module(args.pilot_script, "bidir_v71_pilot")
    audit = load_module(args.audit_script, "bidir_v71_audit")

    cells = sorted({int(x.strip()) for x in args.cells.split(",") if x.strip()})
    sweep = parse_sweep(args.half_step_sweep)
    if not cells:
        p.error("--cells must contain at least one cell")
    if args.transport_residual_subdivisions < 1 or args.paired_subdivisions < 1:
        p.error("subdivision counts must be positive")

    with np.load(args.dense_raw, allow_pickle=True) as d:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])

    if min(cells) < 0 or max(cells) >= starts.size - 1:
        p.error(f"invalid cells for {starts.size - 1} dense cells: {cells}")

    p_two_line = 2.0 * p_line
    cap_length = p_two_line - p_oriented
    v5_rows = corpus.read_v5_rows(args.v5_rows, args.v5_subdivision)
    if set(v5_rows) != set(range(starts.size)):
        missing = sorted(set(range(starts.size)) - set(v5_rows))
        raise RuntimeError(f"v5 certificate missing dense starts: {missing}")

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
    anchor_cache: dict[int, Any] = {}

    def get_anchor(idx: int) -> Any:
        if idx in anchor_cache:
            return anchor_cache[idx]
        row = v5_rows[idx]
        s = float(starts[idx])
        tc = corpus.guide_transport_center(
            envelope, start=s, step=body_step, steps=args.body_steps
        )
        gc = corpus.guide_transport_center(
            envelope,
            start=s + p_oriented,
            step=cap_step,
            steps=args.cap_steps,
        )
        cc = np.eye(9) - gc
        rt = outward(float(row["body_absolute_radius"]))
        rc = outward(float(row["edge_absolute_radius"]))

        sep = n2((cc @ tc)[:2, :2])
        recorded_sep = float(row["propagated_center_separation"])
        align = abs(sep - recorded_sep)
        if align > 5e-10:
            raise RuntimeError(
                f"anchor {idx}: center mismatch against v5: {align:.3e}"
            )
        anchor = corpus.Anchor(
            index=idx,
            start=s,
            t=corpus.MatrixBall(tc, rt),
            c=corpus.MatrixBall(cc, rc),
            recorded_lower=float(row["endpoint_separation_lower"]),
            recorded_sep=recorded_sep,
            center_alignment=align,
        )
        anchor_cache[idx] = anchor
        return anchor

    print(f"Script version          = {SCRIPT_VERSION}")
    print(f"Q                       = {args.q:g}")
    print(f"selected cells          = {cells}")
    print(f"half-step sweep         = {sweep}")
    print(f"P_oriented              = {p_oriented:.15e}")
    print(f"2 P_line                = {p_two_line:.15e}")
    print(f"edge cap DeltaP         = {cap_length:.15e}")
    print(f"v5 subdivision          = {args.v5_subdivision}")
    print(f"H/K residual subdivisions= {args.transport_residual_subdivisions}")
    print(f"local paired pieces     = {args.paired_subdivisions}")
    print("left reset T            = H1*T_left*H0^{-1}")
    print("left reset G            = H2*G_left*H1^{-1}")
    print("right reset T           = K1^{-1}*T_right*K0")
    print("right reset G           = K2^{-1}*G_right*K1")
    print("coverage                = left anchor -> midpoint <- right anchor")

    step_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for n_half in sweep:
        print("=" * 78, flush=True)
        print(f"steps / half-cell={n_half}", flush=True)

        for cell in cells:
            left_anchor = get_anchor(cell)
            right_anchor = get_anchor(cell + 1)
            a = float(starts[cell])
            b = float(starts[cell + 1])
            mid = 0.5 * (a + b)
            h = (mid - a) / n_half

            # ---------- left half ----------
            eye = corpus.MatrixBall(np.eye(9), 0.0)
            h0 = corpus.MatrixBall(np.eye(9), 0.0)
            h1 = corpus.MatrixBall(np.eye(9), 0.0)
            h2 = corpus.MatrixBall(np.eye(9), 0.0)
            g_left = corpus.MatrixBall(np.eye(9) - left_anchor.c.center, left_anchor.c.radius)

            left_min = math.inf
            left_bad = -1
            max_left_reset_t = max_left_reset_c = 0.0
            max_left_tube_t = max_left_tube_c = 0.0
            max_left_short = 0.0
            max_left_rq = 0.0
            max_left_invq = 0.0

            for j in range(n_half):
                x0 = a + j * h
                x1 = a + (j + 1) * h

                inv_h0, q0 = reset.ball_inv(corpus, h0)
                inv_h1, q1 = reset.ball_inv(corpus, h1)
                t_reset = reset.ball_mul(
                    corpus, reset.ball_mul(corpus, h1, left_anchor.t), inv_h0
                )
                g_reset = reset.ball_mul(
                    corpus, reset.ball_mul(corpus, h2, g_left), inv_h1
                )
                c_reset = reset.identity_minus(corpus, g_reset)

                _tn, _cn, diag = corpus.validate_corpus_step(
                    base,
                    pilot,
                    envelope,
                    physical_left=x0,
                    physical_right=x1,
                    direction=+1,
                    t=t_reset,
                    c=c_reset,
                    p_oriented=p_oriented,
                    p_two_line=p_two_line,
                    paired_subdivisions=args.paired_subdivisions,
                    picard_iterations=args.picard_iterations,
                    picard_rtol=args.picard_rtol,
                )
                lower = float(diag["active_lower"])
                left_min = min(left_min, lower)
                if lower <= 0.0 and left_bad < 0:
                    left_bad = j + 1

                max_left_reset_t = max(max_left_reset_t, t_reset.radius)
                max_left_reset_c = max(max_left_reset_c, c_reset.radius)
                max_left_tube_t = max(max_left_tube_t, float(diag["T_tube_radius"]))
                max_left_tube_c = max(max_left_tube_c, float(diag["C_tube_radius"]))
                max_left_short = max(max_left_short, h0.radius, h1.radius, h2.radius)
                max_left_rq = max(max_left_rq, float(diag["rq"]))
                max_left_invq = max(max_left_invq, q0, q1)

                step_rows.append({
                    "half_steps": n_half, "cell_index": cell, "side": "left",
                    "step_index": j + 1, "physical_left": x0, "physical_right": x1,
                    "reset_T_radius": t_reset.radius, "reset_C_radius": c_reset.radius,
                    "local_T_tube_radius": float(diag["T_tube_radius"]),
                    "local_C_tube_radius": float(diag["C_tube_radius"]),
                    "active_center_separation": float(diag["active_center_separation"]),
                    "active_product_radius": float(diag["active_product_radius"]),
                    "active_lower": lower, "local_rq": float(diag["rq"]),
                    "transport_radius_0": h0.radius,
                    "transport_radius_1": h1.radius,
                    "transport_radius_2": h2.radius,
                    "inverse_q_0": q0, "inverse_q_1": q1,
                    "step_pass": bool(lower > 0.0),
                })

                inc0, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"L-H0-c{cell}-n{n_half}-j{j+1}",
                )
                inc1, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0 + p_oriented, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"L-H1-c{cell}-n{n_half}-j{j+1}",
                )
                inc2, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0 + p_two_line, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"L-H2-c{cell}-n{n_half}-j{j+1}",
                )
                h0 = reset.ball_mul(corpus, inc0, h0)
                h1 = reset.ball_mul(corpus, inc1, h1)
                h2 = reset.ball_mul(corpus, inc2, h2)

            # Reconstruct LEFT midpoint.
            inv_h0, _ = reset.ball_inv(corpus, h0)
            inv_h1, _ = reset.ball_inv(corpus, h1)
            t_mid_left = reset.ball_mul(
                corpus, reset.ball_mul(corpus, h1, left_anchor.t), inv_h0
            )
            g_mid_left = reset.ball_mul(
                corpus, reset.ball_mul(corpus, h2, g_left), inv_h1
            )
            c_mid_left = reset.identity_minus(corpus, g_mid_left)

            # ---------- right half ----------
            k0 = corpus.MatrixBall(np.eye(9), 0.0)
            k1 = corpus.MatrixBall(np.eye(9), 0.0)
            k2 = corpus.MatrixBall(np.eye(9), 0.0)
            g_right = corpus.MatrixBall(np.eye(9) - right_anchor.c.center, right_anchor.c.radius)

            right_min = math.inf
            right_bad = -1
            max_right_reset_t = max_right_reset_c = 0.0
            max_right_tube_t = max_right_tube_c = 0.0
            max_right_short = 0.0
            max_right_rq = 0.0
            max_right_invq = 0.0

            for j in range(n_half):
                xr = b - j * h
                xl = b - (j + 1) * h

                inv_k1, q1 = reset.ball_inv(corpus, k1)
                inv_k2, q2 = reset.ball_inv(corpus, k2)
                t_reset = reset.ball_mul(
                    corpus, reset.ball_mul(corpus, inv_k1, right_anchor.t), k0
                )
                g_reset = reset.ball_mul(
                    corpus, reset.ball_mul(corpus, inv_k2, g_right), k1
                )
                c_reset = reset.identity_minus(corpus, g_reset)

                _tn, _cn, diag = corpus.validate_corpus_step(
                    base,
                    pilot,
                    envelope,
                    physical_left=xl,
                    physical_right=xr,
                    direction=-1,
                    t=t_reset,
                    c=c_reset,
                    p_oriented=p_oriented,
                    p_two_line=p_two_line,
                    paired_subdivisions=args.paired_subdivisions,
                    picard_iterations=args.picard_iterations,
                    picard_rtol=args.picard_rtol,
                )
                lower = float(diag["active_lower"])
                right_min = min(right_min, lower)
                if lower <= 0.0 and right_bad < 0:
                    right_bad = j + 1

                max_right_reset_t = max(max_right_reset_t, t_reset.radius)
                max_right_reset_c = max(max_right_reset_c, c_reset.radius)
                max_right_tube_t = max(max_right_tube_t, float(diag["T_tube_radius"]))
                max_right_tube_c = max(max_right_tube_c, float(diag["C_tube_radius"]))
                max_right_short = max(max_right_short, k0.radius, k1.radius, k2.radius)
                max_right_rq = max(max_right_rq, float(diag["rq"]))
                max_right_invq = max(max_right_invq, q1, q2)

                step_rows.append({
                    "half_steps": n_half, "cell_index": cell, "side": "right",
                    "step_index": j + 1, "physical_left": xl, "physical_right": xr,
                    "reset_T_radius": t_reset.radius, "reset_C_radius": c_reset.radius,
                    "local_T_tube_radius": float(diag["T_tube_radius"]),
                    "local_C_tube_radius": float(diag["C_tube_radius"]),
                    "active_center_separation": float(diag["active_center_separation"]),
                    "active_product_radius": float(diag["active_product_radius"]),
                    "active_lower": lower, "local_rq": float(diag["rq"]),
                    "transport_radius_0": k0.radius,
                    "transport_radius_1": k1.radius,
                    "transport_radius_2": k2.radius,
                    "inverse_q_1": q1, "inverse_q_2": q2,
                    "step_pass": bool(lower > 0.0),
                })

                # Extend K(x)=Phi(b,x) one slab to the LEFT:
                # Phi(b,xl) = Phi(b,xr) Phi(xr,xl).
                inc0, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=xl, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"R-K0-c{cell}-n{n_half}-j{j+1}",
                )
                inc1, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=xl + p_oriented, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"R-K1-c{cell}-n{n_half}-j{j+1}",
                )
                inc2, _, _ = reset.transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=xl + p_two_line, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"R-K2-c{cell}-n{n_half}-j{j+1}",
                )
                k0 = reset.ball_mul(corpus, k0, inc0)
                k1 = reset.ball_mul(corpus, k1, inc1)
                k2 = reset.ball_mul(corpus, k2, inc2)

            # Reconstruct RIGHT midpoint.
            inv_k1, _ = reset.ball_inv(corpus, k1)
            inv_k2, _ = reset.ball_inv(corpus, k2)
            t_mid_right = reset.ball_mul(
                corpus, reset.ball_mul(corpus, inv_k1, right_anchor.t), k0
            )
            g_mid_right = reset.ball_mul(
                corpus, reset.ball_mul(corpus, inv_k2, g_right), k1
            )
            c_mid_right = reset.identity_minus(corpus, g_mid_right)

            t_mid_gap = n2(t_mid_left.center - t_mid_right.center)
            c_mid_gap = n2(c_mid_left.center - c_mid_right.center)
            t_mid_overlap = (
                t_mid_left.radius + t_mid_right.radius - t_mid_gap
            )
            c_mid_overlap = (
                c_mid_left.radius + c_mid_right.radius - c_mid_gap
            )

            cell_min = min(left_min, right_min)
            passed = bool(left_bad < 0 and right_bad < 0)
            summary = {
                "half_steps": n_half,
                "cell_index": cell,
                "cell_width": b - a,
                "minimum_continuous_lower": cell_min,
                "left_minimum_lower": left_min,
                "right_minimum_lower": right_min,
                "left_first_bad_step": left_bad,
                "right_first_bad_step": right_bad,
                "maximum_reset_T_radius": max(max_left_reset_t, max_right_reset_t),
                "maximum_reset_C_radius": max(max_left_reset_c, max_right_reset_c),
                "maximum_local_T_tube_radius": max(max_left_tube_t, max_right_tube_t),
                "maximum_local_C_tube_radius": max(max_left_tube_c, max_right_tube_c),
                "maximum_short_transport_radius": max(max_left_short, max_right_short),
                "maximum_local_rq": max(max_left_rq, max_right_rq),
                "maximum_inverse_q": max(max_left_invq, max_right_invq),
                "midpoint_T_center_gap": t_mid_gap,
                "midpoint_C_center_gap": c_mid_gap,
                "midpoint_T_overlap_margin": t_mid_overlap,
                "midpoint_C_overlap_margin": c_mid_overlap,
                "cell_pass": passed,
            }
            summaries.append(summary)

            print(
                f"cell={cell:2d} min={cell_min:+.6e} "
                f"badL={left_bad:4d} badR={right_bad:4d} "
                f"resetRT={summary['maximum_reset_T_radius']:.3e} "
                f"tubeRT={summary['maximum_local_T_tube_radius']:.3e} "
                f"resetRC={summary['maximum_reset_C_radius']:.3e} "
                f"tubeRC={summary['maximum_local_C_tube_radius']:.3e} "
                f"rShort={summary['maximum_short_transport_radius']:.3e} "
                f"rq={summary['maximum_local_rq']:.3e} "
                f"midGapT={t_mid_gap:.2e} midGapC={c_mid_gap:.2e} "
                f"midOvT={t_mid_overlap:+.3e} midOvC={c_mid_overlap:+.3e} "
                f"pass={passed}",
                flush=True,
            )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(str(prefix) + "_steps.csv"), step_rows)
    write_csv(Path(str(prefix) + "_summary.csv"), summaries)

    largest = max(sweep)
    largest_rows = [r for r in summaries if int(r["half_steps"]) == largest]
    passed_count = sum(bool(r["cell_pass"]) for r in largest_rows)
    classification = (
        "BIDIRECTIONAL COCYCLE-RESET CORPUS PILOT PASSED AT LARGEST HALF-STEP RESOLUTION"
        if passed_count == len(largest_rows)
        else "BIDIRECTIONAL COCYCLE-RESET CORPUS PILOT STILL OPEN"
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "q": args.q,
        "scope": "continuous start-coordinate endpoint distinction in frozen-spline linear cocycle",
        "cells": cells,
        "half_step_sweep": sweep,
        "transport_residual_subdivisions": args.transport_residual_subdivisions,
        "paired_subdivisions": args.paired_subdivisions,
        "summaries": summaries,
        "classification": classification,
    }
    Path(str(prefix) + ".json").write_text(json.dumps(payload, indent=2) + "\n")

    with Path(str(prefix) + "_report.txt").open("w") as handle:
        handle.write(f"Script version: {SCRIPT_VERSION}\n\n")
        handle.write("Q=550 BIDIRECTIONAL COCYCLE-RESET CORPUS PILOT\n")
        handle.write("==============================================\n\n")
        handle.write(f"cells                         = {cells}\n")
        handle.write(f"half-step sweep               = {sweep}\n")
        handle.write(
            f"H/K residual subdivisions     = {args.transport_residual_subdivisions}\n"
        )
        handle.write(f"local paired pieces           = {args.paired_subdivisions}\n\n")
        handle.write("Architecture\n------------\n")
        handle.write("Left:  T=H1*T_left*H0^-1, G=H2*G_left*H1^-1\n")
        handle.write("Right: T=K1^-1*T_right*K0, G=K2^-1*G_right*K1\n")
        handle.write("Each endpoint anchor covers only its nearest half-cell.\n")
        handle.write("Local Picard endpoint balls are discarded after each slab.\n\n")
        handle.write("Summaries\n---------\n")
        for r in summaries:
            handle.write(
                f"Nhalf={int(r['half_steps']):4d} cell={int(r['cell_index']):2d} "
                f"min={r['minimum_continuous_lower']:+.6e} "
                f"badL={int(r['left_first_bad_step']):4d} "
                f"badR={int(r['right_first_bad_step']):4d} "
                f"resetRT={r['maximum_reset_T_radius']:.6e} "
                f"tubeRT={r['maximum_local_T_tube_radius']:.6e} "
                f"rShort={r['maximum_short_transport_radius']:.6e} "
                f"rq={r['maximum_local_rq']:.6e} "
                f"pass={bool(r['cell_pass'])}\n"
            )
        handle.write(f"\nclassification = {classification}\n\n")
        handle.write(
            "Scope: frozen natural-cubic-spline linear cocycle only; exact nonlinear-orbit "
            "interval validation remains open.\n"
        )

    print("=" * 78)
    print(
        f"largest half-step summary: passed={passed_count}/{len(largest_rows)} "
        f"classification={classification}"
    )


if __name__ == "__main__":
    main()
