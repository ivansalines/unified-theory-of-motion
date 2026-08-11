#!/usr/bin/env python3
"""Q=550 cocycle-reset corpus inter-node bridge (v7).

Purpose
-------
v6/v6.1 showed that the continuous inter-node failure is not caused by the
paired endpoint forcing B1-B2: its radius tightens cleanly under refinement.
The limiting term is the cumulative Euler/Picard radius carried by the long
body T in the start coordinate.

This v7 keeps the same exact corpus factorization

    D(s) = C(s) T(s),            C(s) = I - G(s),

but DOES NOT chain the long-body Euler endpoint ball from one local step to
the next.  Instead each local step is rigorously re-anchored to the left v5
node through exact cocycle shift identities.

Let a be the left dense-node anchor and x >= a. Define the three short forward
transports

    H0(x) = Phi(x, a),
    H1(x) = Phi(x + P_oriented, a + P_oriented),
    H2(x) = Phi(x + 2 P_line, a + 2 P_line).

Then exactly

    T(x) = H1(x) T(a) H0(x)^(-1),

and, for G = I-C,

    G(x) = H2(x) G(a) H1(x)^(-1).

The Hk are accumulated from rigorously validated *short* relative-defect
increments.  Physical motion lives in their centers; only validation error is
carried in their radii.  At every local cell slab we rebuild T(x), C(x) from
the original rigorous v5 anchor and the current Hk balls, then use the v6.1
one-step Picard tube only to cover that slab.  Its endpoint radius is discarded:
the next slab is reset from the cocycle identity again.

This directly tests whether the v6.1 obstruction was artificial numerical
memory in the body coordinate.

Scope
-----
Frozen natural-cubic-spline linear cocycle only.  This script does not
interval-validate the nonlinear orbit cache against an exact nonlinear
trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np


SCRIPT_VERSION = "2026-08-10-q550-cocycle-reset-corpus-bridge-v7"

DEFAULT_CORPUS = Path("test_support_q550_corpus_inter_node_bridge_v6_1.py")
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_4.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_V5_ROWS = Path("support_q550_endpoint_nesting_dense_span_0_80_rows.csv")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_cocycle_reset_corpus_bridge_v7")


def outward(x: float) -> float:
    return float(np.nextafter(float(x), np.inf))


def downward(x: float) -> float:
    return float(np.nextafter(float(x), -np.inf))


def n2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float), ord=2))


def upward_expm1(base: Any, alpha: float) -> float:
    with mp.workdps(base.MP_DPS):
        value = mp.expm1(mp.mpf(alpha))
        _, hi = base.mp_bracket(value, ulps=4)
    return float(hi)


def ball_mul(corpus: Any, a: Any, b: Any) -> Any:
    return corpus.MatrixBall(
        a.center @ b.center,
        corpus.product_radius(a, b),
    )


def ball_inv(corpus: Any, a: Any) -> tuple[Any, float]:
    """Rigorous spectral ball inverse.

    If ||A0^{-1}|| r < 1 then
      ||(A0+dA)^{-1} - A0^{-1}||
      <= ||A0^{-1}||^2 r / (1 - ||A0^{-1}|| r).
    """
    center_inv = np.linalg.inv(a.center)
    ni = outward(n2(center_inv))
    q = outward(ni * a.radius)
    if not q < 1.0:
        raise RuntimeError(f"cocycle reset inverse ball is singular/too wide: q={q:.6e}")
    radius = outward(ni * q / (1.0 - q))
    return corpus.MatrixBall(center_inv, radius), q


def identity_minus(corpus: Any, g: Any) -> Any:
    return corpus.MatrixBall(np.eye(g.center.shape[0]) - g.center, g.radius)


def transport_increment_ball(
    *,
    base: Any,
    pilot: Any,
    corpus: Any,
    envelope: Any,
    start: float,
    step: float,
    residual_subdivisions: int,
    label: str,
) -> tuple[Any, float, float]:
    """Validate one short forward transport and return an absolute matrix ball."""
    # The pilot is deliberately chatty even for a one-step transport.
    # Suppress only those progress lines; all failures still propagate.
    with contextlib.redirect_stdout(io.StringIO()):
        val = pilot.validate_transport_relative_defect(
            base,
            envelope,
            start=start,
            step=step,
            steps=1,
            residual_subdivisions=residual_subdivisions,
            label=label,
        )
    eps = upward_expm1(base, val.accumulated_relative_defect)
    radius = outward(n2(val.center) * eps)
    return corpus.MatrixBall(np.asarray(val.center, dtype=float), radius), eps, float(
        val.accumulated_relative_defect
    )


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


def parse_sweep(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or vals[0] < 1:
        raise ValueError("reset sweep must contain positive integers")
    return vals


def run_self_test(corpus: Any) -> None:
    rng = np.random.default_rng(20260810)
    for _ in range(500):
        a0 = np.eye(4) + 0.05 * rng.normal(size=(4, 4))
        r = 2e-3 * rng.random()
        a = corpus.MatrixBall(a0, r)
        ainv, _q = ball_inv(corpus, a)
        for _j in range(20):
            da = rng.normal(size=(4, 4))
            da *= (r * rng.random()) / max(n2(da), 1e-300)
            exact = np.linalg.inv(a0 + da)
            err = n2(exact - ainv.center)
            if err > ainv.radius * (1.0 + 1e-11):
                raise AssertionError(f"inverse ball escaped: {err:.3e} > {ainv.radius:.3e}")
    print("cocycle-reset corpus ball algebra self-test passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-script", type=Path, default=DEFAULT_CORPUS)
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
    p.add_argument("--cells", default="39",
                   help="comma-separated dense cell indices")
    p.add_argument("--reset-sweep", default="32,64,128",
                   help="full-cell reset steps to test in one run")
    p.add_argument("--transport-residual-subdivisions", type=int, default=8,
                   help="relative-defect subdivisions for each short H increment")
    p.add_argument("--paired-subdivisions", type=int, default=8,
                   help="paired B1-B2 pieces inside each local slab")
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

    # Load helper architecture first; it contains the already-tested v6.1 ball
    # algebra, anchor reconstruction, paired generator enclosure, and one-step
    # Picard tube.
    import importlib.util
    import sys

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

    corpus = load_module(args.corpus_script, "corpus_v7_parent")
    if args.self_test:
        run_self_test(corpus)
        return

    base = load_module(args.base_script, "corpus_v7_base")
    pilot = load_module(args.pilot_script, "corpus_v7_pilot")
    audit = load_module(args.audit_script, "corpus_v7_audit")

    sweep = parse_sweep(args.reset_sweep)
    cells = sorted({int(x.strip()) for x in args.cells.split(",") if x.strip()})
    if not cells:
        p.error("--cells must contain at least one cell index")
    if args.transport_residual_subdivisions < 1 or args.paired_subdivisions < 1:
        p.error("subdivision counts must be positive")

    with np.load(args.dense_raw, allow_pickle=True) as d:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])

    if min(cells) < 0 or max(cells) >= starts.size - 1:
        p.error(f"invalid cell selection for {starts.size - 1} cells: {cells}")

    p_two_line = 2.0 * p_line
    cap_length = p_two_line - p_oriented
    v5_rows = corpus.read_v5_rows(args.v5_rows, args.v5_subdivision)
    if set(v5_rows) != set(range(starts.size)):
        missing = sorted(set(range(starts.size)) - set(v5_rows))
        raise RuntimeError(f"v5 certificate does not cover all dense starts; missing={missing}")

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
                f"anchor {idx}: recomputed center not aligned with v5: {align:.3e}"
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

    print(f"Script version         = {SCRIPT_VERSION}")
    print(f"Q                      = {args.q:g}")
    print(f"selected cells         = {cells}")
    print(f"reset sweep            = {sweep}")
    print(f"P_oriented             = {p_oriented:.15e}")
    print(f"2 P_line               = {p_two_line:.15e}")
    print(f"edge cap DeltaP        = {cap_length:.15e}")
    print(f"v5 subdivision         = {args.v5_subdivision}")
    print(f"H residual subdivisions= {args.transport_residual_subdivisions}")
    print(f"local paired pieces    = {args.paired_subdivisions}")
    print("reset identity T       = H1*T_anchor*H0^{-1}")
    print("reset identity G       = H2*G_anchor*H1^{-1}")
    print("local tube             = v6.1 one-step only; endpoint ball discarded")

    all_step_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for reset_steps in sweep:
        print("=" * 78, flush=True)
        print(f"reset steps / full cell={reset_steps}", flush=True)

        for cell in cells:
            anchor = get_anchor(cell)
            right_anchor = get_anchor(cell + 1)
            a = float(starts[cell])
            b = float(starts[cell + 1])
            h = (b - a) / reset_steps

            eye_ball = corpus.MatrixBall(np.eye(9), 0.0)
            h0 = eye_ball
            h1 = corpus.MatrixBall(np.eye(9), 0.0)
            h2 = corpus.MatrixBall(np.eye(9), 0.0)
            g_anchor = corpus.MatrixBall(np.eye(9) - anchor.c.center, anchor.c.radius)

            cell_min = math.inf
            first_bad = -1
            max_reset_t_radius = 0.0
            max_reset_c_radius = 0.0
            max_local_t_tube = 0.0
            max_local_c_tube = 0.0
            max_h_radius = 0.0
            max_h_inverse_q = 0.0
            max_increment_eps = 0.0
            max_local_rq = 0.0

            for j in range(reset_steps):
                x0 = a + j * h
                x1 = a + (j + 1) * h

                inv_h0, q0 = ball_inv(corpus, h0)
                inv_h1, q1 = ball_inv(corpus, h1)
                t_reset = ball_mul(corpus, ball_mul(corpus, h1, anchor.t), inv_h0)
                g_reset = ball_mul(corpus, ball_mul(corpus, h2, g_anchor), inv_h1)
                c_reset = identity_minus(corpus, g_reset)

                # Cover only [x0,x1] with the local Picard tube.  Crucially its
                # endpoint ball is NOT inherited by the next slab.
                _tnext, _cnext, diag = corpus.validate_corpus_step(
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
                cell_min = min(cell_min, lower)
                if lower <= 0.0 and first_bad < 0:
                    first_bad = j + 1

                max_reset_t_radius = max(max_reset_t_radius, t_reset.radius)
                max_reset_c_radius = max(max_reset_c_radius, c_reset.radius)
                max_local_t_tube = max(max_local_t_tube, float(diag["T_tube_radius"]))
                max_local_c_tube = max(max_local_c_tube, float(diag["C_tube_radius"]))
                max_h_radius = max(max_h_radius, h0.radius, h1.radius, h2.radius)
                max_h_inverse_q = max(max_h_inverse_q, q0, q1)
                max_local_rq = max(max_local_rq, float(diag["rq"]))

                all_step_rows.append({
                    "reset_steps": reset_steps,
                    "cell_index": cell,
                    "step_index": j + 1,
                    "physical_left": x0,
                    "physical_right": x1,
                    "reset_T_radius": t_reset.radius,
                    "reset_C_radius": c_reset.radius,
                    "local_T_tube_radius": float(diag["T_tube_radius"]),
                    "local_C_tube_radius": float(diag["C_tube_radius"]),
                    "active_center_separation": float(diag["active_center_separation"]),
                    "active_product_radius": float(diag["active_product_radius"]),
                    "active_lower": lower,
                    "local_rq": float(diag["rq"]),
                    "H0_radius": h0.radius,
                    "H1_radius": h1.radius,
                    "H2_radius": h2.radius,
                    "inverse_q_H0": q0,
                    "inverse_q_H1": q1,
                    "step_pass": bool(lower > 0.0),
                })

                # Advance the three short cocycle transports rigorously.
                inc0, eps0, _ = transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"H0-c{cell}-n{reset_steps}-j{j+1}",
                )
                inc1, eps1, _ = transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0 + p_oriented, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"H1-c{cell}-n{reset_steps}-j{j+1}",
                )
                inc2, eps2, _ = transport_increment_ball(
                    base=base, pilot=pilot, corpus=corpus, envelope=envelope,
                    start=x0 + p_two_line, step=h,
                    residual_subdivisions=args.transport_residual_subdivisions,
                    label=f"H2-c{cell}-n{reset_steps}-j{j+1}",
                )
                max_increment_eps = max(max_increment_eps, eps0, eps1, eps2)
                h0 = ball_mul(corpus, inc0, h0)
                h1 = ball_mul(corpus, inc1, h1)
                h2 = ball_mul(corpus, inc2, h2)

            # Reconstruct the right node once more from the left anchor and the
            # full short cocycle shift; compare with the independent v5 right
            # node as a diagnostic (not used to prove containment).
            inv_h0, q0 = ball_inv(corpus, h0)
            inv_h1, q1 = ball_inv(corpus, h1)
            t_right_from_left = ball_mul(
                corpus, ball_mul(corpus, h1, anchor.t), inv_h0
            )
            g_right_from_left = ball_mul(
                corpus, ball_mul(corpus, h2, g_anchor), inv_h1
            )
            c_right_from_left = identity_minus(corpus, g_right_from_left)

            t_center_gap = n2(t_right_from_left.center - right_anchor.t.center)
            c_center_gap = n2(c_right_from_left.center - right_anchor.c.center)
            t_overlap_margin = (
                t_right_from_left.radius + right_anchor.t.radius - t_center_gap
            )
            c_overlap_margin = (
                c_right_from_left.radius + right_anchor.c.radius - c_center_gap
            )

            passed = bool(first_bad < 0)
            summary = {
                "reset_steps": reset_steps,
                "cell_index": cell,
                "cell_width": b - a,
                "minimum_continuous_lower": cell_min,
                "first_bad_step": first_bad,
                "first_bad_fraction": -1.0 if first_bad < 0 else first_bad / reset_steps,
                "maximum_reset_T_radius": max_reset_t_radius,
                "maximum_reset_C_radius": max_reset_c_radius,
                "maximum_local_T_tube_radius": max_local_t_tube,
                "maximum_local_C_tube_radius": max_local_c_tube,
                "maximum_short_H_radius": max_h_radius,
                "maximum_H_inverse_q": max_h_inverse_q,
                "maximum_increment_relative_eps": max_increment_eps,
                "maximum_local_rq": max_local_rq,
                "right_T_center_gap": t_center_gap,
                "right_C_center_gap": c_center_gap,
                "right_T_overlap_margin": t_overlap_margin,
                "right_C_overlap_margin": c_overlap_margin,
                "cell_pass": passed,
            }
            summary_rows.append(summary)

            print(
                f"cell={cell:2d} min={cell_min:+.6e} "
                f"bad={first_bad:4d} "
                f"resetRT={max_reset_t_radius:.3e} "
                f"tubeRT={max_local_t_tube:.3e} "
                f"resetRC={max_reset_c_radius:.3e} "
                f"tubeRC={max_local_c_tube:.3e} "
                f"rH={max_h_radius:.3e} "
                f"rq={max_local_rq:.3e} "
                f"overlapT={t_overlap_margin:+.3e} "
                f"overlapC={c_overlap_margin:+.3e} "
                f"pass={passed}",
                flush=True,
            )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(str(prefix) + "_steps.csv"), all_step_rows)
    write_csv(Path(str(prefix) + "_summary.csv"), summary_rows)

    largest = max(sweep)
    largest_rows = [r for r in summary_rows if int(r["reset_steps"]) == largest]
    largest_passed = sum(bool(r["cell_pass"]) for r in largest_rows)
    classification = (
        "COCYCLE-RESET CORPUS PILOT PASSED AT LARGEST RESET RESOLUTION"
        if largest_passed == len(largest_rows)
        else "COCYCLE-RESET CORPUS PILOT STILL OPEN"
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "q": args.q,
        "scope": "continuous start-coordinate endpoint distinction in frozen-spline linear cocycle",
        "cells": cells,
        "reset_sweep": sweep,
        "transport_residual_subdivisions": args.transport_residual_subdivisions,
        "paired_subdivisions": args.paired_subdivisions,
        "summaries": summary_rows,
        "largest_reset_cells_passed": largest_passed,
        "largest_reset_cells_tested": len(largest_rows),
        "classification": classification,
    }
    Path(str(prefix) + ".json").write_text(json.dumps(payload, indent=2) + "\n")

    with Path(str(prefix) + "_report.txt").open("w") as handle:
        handle.write(f"Script version: {SCRIPT_VERSION}\n\n")
        handle.write("Q=550 COCYCLE-RESET CORPUS INTER-NODE PILOT\n")
        handle.write("===========================================\n\n")
        handle.write(f"cells                         = {cells}\n")
        handle.write(f"reset sweep                   = {sweep}\n")
        handle.write(
            f"H residual subdivisions       = {args.transport_residual_subdivisions}\n"
        )
        handle.write(f"local paired pieces           = {args.paired_subdivisions}\n\n")
        handle.write("Architecture\n------------\n")
        handle.write("T(x) = H1(x) T(anchor) H0(x)^(-1)\n")
        handle.write("G(x) = H2(x) G(anchor) H1(x)^(-1)\n")
        handle.write("C(x) = I-G(x), D(x)=C(x)T(x)\n")
        handle.write(
            "Each local Picard tube is discarded at its endpoint; the next slab is "
            "rebuilt from the original v5 anchor through short validated cocycle transports.\n\n"
        )
        handle.write("Summaries\n---------\n")
        for r in summary_rows:
            handle.write(
                f"N={int(r['reset_steps']):4d} cell={int(r['cell_index']):2d} "
                f"min={r['minimum_continuous_lower']:+.6e} "
                f"bad={int(r['first_bad_step']):4d} "
                f"resetRT={r['maximum_reset_T_radius']:.6e} "
                f"tubeRT={r['maximum_local_T_tube_radius']:.6e} "
                f"rH={r['maximum_short_H_radius']:.6e} "
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
        f"largest reset summary: passed={largest_passed}/{len(largest_rows)} "
        f"classification={classification}"
    )


if __name__ == "__main__":
    main()
