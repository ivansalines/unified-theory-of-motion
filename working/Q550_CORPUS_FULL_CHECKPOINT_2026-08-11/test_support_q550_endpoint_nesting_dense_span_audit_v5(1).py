#!/usr/bin/env python3
"""Q=550 endpoint-nesting dense-start span audit (v5).

Purpose
-------
The v4 one-cell pilot showed that at Q=550 the endpoint difference is best
validated in the nested/common-body form

    T_2P(s) = G_edge(s) T_O(s),
    D(s)    = T_O(s) - T_2P(s) = (I-G_edge(s)) T_O(s),

where

    G_edge(s) = Phi(s + 2 P_line, s + P_oriented)

is only the tiny cap of length

    DeltaP = 2 P_line - P_oriented = 1/720.

This script applies that same architecture independently at every selected
start in the dense Q=550 grid.  It deliberately does NOT chain uncertainty
from one start to the next: each start is directly validated in the frozen
natural-cubic-spline linear cocycle.  That avoids introducing artificial
history through scalar error accumulation while testing whether the endpoint-
nesting mechanism is uniform across the grid.

At each start s:

  1. validate T_O(s) directly with the relative-defect moving frame;
  2. validate the tiny edge cap G_edge(s);
  3. certify only the active 2x2 part of (I-G_edge) T_O;
  4. compare the guide center with the archived dense-grid difference.

The result is a rigorous audit of the SELECTED DENSE STARTS only.  It does not
by itself fill the continuous intervals between adjacent starts.  Scope also
remains the frozen spline linear cocycle; the nonlinear orbit cache is not yet
interval-validated against an exact nonlinear trajectory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mpmath as mp
import numpy as np


SCRIPT_VERSION = "2026-08-07-q550-endpoint-nesting-dense-span-audit-v5"
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_3.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_endpoint_nesting_dense_span_audit")


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


def parse_sweep(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or vals[0] < 1:
        raise ValueError("subdivision sweep must contain positive integers")
    return vals


def n2(a: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float), ord=2))


def upward_expm1(base: ModuleType, alpha: float) -> float:
    with mp.workdps(base.MP_DPS):
        value = mp.expm1(mp.mpf(alpha))
        _, hi = base.mp_bracket(value, ulps=4)
    return float(hi)


def active_nested_bound(
    *,
    body_center: np.ndarray,
    body_eps: float,
    edge_center: np.ndarray,
    edge_eps: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Bound the active 2x2 block of D=(I-G)T.

    Relative frames:
        T = U_T (I+E_T), ||E_T|| <= body_eps
        G = U_G (I+E_G), ||E_G|| <= edge_eps

    Write C0=I-U_G.  Then

        (I-G)T = (C0 - U_G E_G)(U_T + U_T E_T).

    A spectral-norm ball is used only AFTER multiplication by the small first-
    two-row factor C0.  The returned radius bounds the first two rows, hence in
    particular the active 2x2 block.
    """
    n = body_center.shape[0]
    if edge_center.shape != (n, n):
        raise ValueError("body/edge center shape mismatch")

    c0 = np.eye(n) - edge_center
    center_full = c0 @ body_center
    active_center = center_full[:2, :2]

    body_norm = n2(body_center)
    body_abs_radius = outward(body_norm * body_eps)

    edge_norm = n2(edge_center)
    edge_abs_radius = outward(edge_norm * edge_eps)

    edge_rows_norm = n2(c0[:2, :])

    # (C0+dC)(T0+dT) - C0 T0
    # = C0 dT + dC T0 + dC dT, with ||dC|| <= ||U_G|| eps_G.
    body_through_edge = outward(edge_rows_norm * body_abs_radius)
    edge_on_body = outward(edge_abs_radius * body_norm)
    cross = outward(edge_abs_radius * body_abs_radius)
    active_radius = outward(body_through_edge + edge_on_body + cross)

    return active_center, active_radius, {
        "body_center_norm": body_norm,
        "body_absolute_radius": body_abs_radius,
        "edge_center_norm": edge_norm,
        "edge_absolute_radius": edge_abs_radius,
        "edge_factor_rows_norm": edge_rows_norm,
        "body_through_edge_term": body_through_edge,
        "edge_on_body_term": edge_on_body,
        "cross_term": cross,
    }


def run_self_test() -> None:
    """Monte-Carlo algebra sanity check for the direct nested endpoint bound."""
    rng = np.random.default_rng(20260807)
    for _ in range(400):
        n = 4
        ut = np.eye(n) + 0.10 * rng.normal(size=(n, n))
        ug = np.eye(n) + 0.004 * rng.normal(size=(n, n))
        et = 0.02
        eg = 2e-5
        center, radius, _ = active_nested_bound(
            body_center=ut,
            body_eps=et,
            edge_center=ug,
            edge_eps=eg,
        )

        def perturb(r: float) -> np.ndarray:
            e = rng.normal(size=(n, n))
            norm = n2(e)
            return e * (r / max(norm, 1e-300))

        for _j in range(30):
            t = ut @ (np.eye(n) + perturb(et))
            g = ug @ (np.eye(n) + perturb(eg))
            exact = ((np.eye(n) - g) @ t)[:2, :2]
            actual = n2(exact - center)
            if actual > radius * (1.0 + 1e-10):
                raise AssertionError(
                    f"dense nested endpoint bound escaped: {actual:.3e} > {radius:.3e}"
                )
    print("endpoint-nesting dense-span algebra self-test passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--pilot-script", type=Path, default=DEFAULT_PILOT)
    p.add_argument("--dense-raw", type=Path, default=DEFAULT_DENSE)
    p.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT)
    p.add_argument("--q", type=float, default=550.0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--end-index", type=int, default=-1,
                   help="inclusive; -1 means the final dense start")
    p.add_argument("--body-steps", type=int, default=1308)
    p.add_argument("--cap-steps", type=int, default=16)
    p.add_argument("--subdivision-sweep", default="8,16,32")
    p.add_argument("--mp-dps", type=int, default=80)
    p.add_argument("--tail-periods", type=int, default=70)
    p.add_argument("--steps-per-period", type=int, default=160)
    p.add_argument("--samples-per-rotation", type=int, default=720)
    p.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    p.add_argument("--gamma-rho", type=float, default=0.3)
    p.add_argument("--progress-every", type=int, default=5)
    p.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        run_self_test()
        return
    if args.body_steps < 20:
        p.error("--body-steps must be at least 20")
    if args.cap_steps < 1:
        p.error("--cap-steps must be positive")
    if args.progress_every < 1:
        p.error("--progress-every must be positive")

    base = load_module(args.base_script, "dense_v5_base")
    base.MP_DPS = int(args.mp_dps)
    pilot = load_module(args.pilot_script, "dense_v5_pilot")
    audit = load_module(args.audit_script, "dense_v5_audit")
    sweep = parse_sweep(args.subdivision_sweep)

    with np.load(args.dense_raw, allow_pickle=True) as d:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])
        archived_oriented = np.asarray(d["fine_oriented"], dtype=float)
        archived_two_line = np.asarray(d["fine_two_line"], dtype=float)

    end_index = starts.size - 1 if args.end_index < 0 else args.end_index
    if not (0 <= args.start_index <= end_index < starts.size):
        p.error(
            f"invalid start/end indexes for {starts.size} starts: "
            f"{args.start_index}..{end_index}"
        )

    p_two_line = 2.0 * p_line
    cap_length = p_two_line - p_oriented
    if not cap_length > 0.0:
        raise ValueError(
            "v5 endpoint nesting requires 2*P_line > P_oriented; "
            f"got DeltaP={cap_length:+.16e}"
        )

    body_step = p_oriented / args.body_steps
    cap_step = cap_length / args.cap_steps

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

    domain_left = float(starts[args.start_index])
    domain_right = float(starts[end_index] + p_two_line)
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

    selected = list(range(args.start_index, end_index + 1))
    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"Q                    = {args.q:g}")
    print(f"dense starts         = {args.start_index} .. {end_index} ({len(selected)} starts)")
    print(f"P_oriented           = {p_oriented:.15e}")
    print(f"2 P_line             = {p_two_line:.15e}")
    print(f"edge cap DeltaP      = {cap_length:.15e}")
    print(f"body steps           = {args.body_steps}")
    print(f"body step            = {body_step:.15e}")
    print(f"cap steps            = {args.cap_steps}")
    print(f"cap step             = {cap_step:.15e}")
    print(f"subdivision sweep    = {sweep}")
    print("architecture         = direct T_O + tiny G_edge; D=(I-G_edge)T_O")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for subdivisions in sweep:
        print("=" * 78, flush=True)
        print(f"residual subdivisions={subdivisions}", flush=True)
        level_rows: list[dict[str, Any]] = []

        for count, idx in enumerate(selected, start=1):
            s = float(starts[idx])
            body = pilot.validate_transport_relative_defect(
                base,
                envelope,
                start=s,
                step=body_step,
                steps=args.body_steps,
                residual_subdivisions=subdivisions,
                label=f"body-i{idx}-m{subdivisions}",
            )
            edge = pilot.validate_transport_relative_defect(
                base,
                envelope,
                start=s + p_oriented,
                step=cap_step,
                steps=args.cap_steps,
                residual_subdivisions=subdivisions,
                label=f"edge-i{idx}-m{subdivisions}",
            )

            body_eps = upward_expm1(base, body.accumulated_relative_defect)
            edge_eps = upward_expm1(base, edge.accumulated_relative_defect)

            active_center, active_radius, diag = active_nested_bound(
                body_center=body.center,
                body_eps=body_eps,
                edge_center=edge.center,
                edge_eps=edge_eps,
            )
            sep_center = n2(active_center)
            lower = downward(sep_center - active_radius)

            archived_diff = (
                archived_oriented[idx][:2, :2]
                - archived_two_line[idx][:2, :2]
            )
            archived_sep = n2(archived_diff)
            center_discrepancy = float(
                np.linalg.norm(active_center - archived_diff, ord="fro")
                / max(np.linalg.norm(archived_diff, ord="fro"), 1e-300)
            )

            row = {
                "residual_subdivisions": subdivisions,
                "start_index": idx,
                "start": s,
                "body_relative_eps": body_eps,
                "body_absolute_radius": diag["body_absolute_radius"],
                "edge_relative_eps": edge_eps,
                "edge_absolute_radius": diag["edge_absolute_radius"],
                "edge_factor_rows_norm": diag["edge_factor_rows_norm"],
                "body_through_edge_term": diag["body_through_edge_term"],
                "edge_on_body_term": diag["edge_on_body_term"],
                "cross_term": diag["cross_term"],
                "active_difference_radius": active_radius,
                "propagated_center_separation": sep_center,
                "archived_center_separation": archived_sep,
                "center_relative_discrepancy": center_discrepancy,
                "endpoint_separation_lower": lower,
                "endpoint_pass": bool(lower > 0.0),
            }
            rows.append(row)
            level_rows.append(row)

            if (
                count == 1
                or count == len(selected)
                or count % args.progress_every == 0
                or not row["endpoint_pass"]
            ):
                print(
                    f"  [{count:>3}/{len(selected)}] i={idx:>2} "
                    f"epsT={body_eps:.3e} epsG={edge_eps:.3e} "
                    f"edgeRows={diag['edge_factor_rows_norm']:.3e} "
                    f"activeR={active_radius:.3e} sep={sep_center:.3e} "
                    f"lower={lower:+.3e} pass={lower > 0.0} "
                    f"disc={center_discrepancy:.2e}",
                    flush=True,
                )

        min_row = min(level_rows, key=lambda r: r["endpoint_separation_lower"])
        max_radius_row = max(level_rows, key=lambda r: r["active_difference_radius"])
        max_disc_row = max(level_rows, key=lambda r: r["center_relative_discrepancy"])
        passed = sum(bool(r["endpoint_pass"]) for r in level_rows)
        summary = {
            "residual_subdivisions": subdivisions,
            "starts_tested": len(level_rows),
            "starts_passed": passed,
            "all_starts_passed": bool(passed == len(level_rows)),
            "minimum_endpoint_lower": float(min_row["endpoint_separation_lower"]),
            "minimum_endpoint_lower_index": int(min_row["start_index"]),
            "maximum_active_radius": float(max_radius_row["active_difference_radius"]),
            "maximum_active_radius_index": int(max_radius_row["start_index"]),
            "maximum_center_relative_discrepancy": float(max_disc_row["center_relative_discrepancy"]),
            "maximum_center_relative_discrepancy_index": int(max_disc_row["start_index"]),
        }
        summaries.append(summary)
        print(
            f"  summary m={subdivisions}: passed={passed}/{len(level_rows)} "
            f"minLower={summary['minimum_endpoint_lower']:+.6e} "
            f"at i={summary['minimum_endpoint_lower_index']} "
            f"maxR={summary['maximum_active_radius']:.6e} "
            f"maxDisc={summary['maximum_center_relative_discrepancy']:.3e}",
            flush=True,
        )

    largest = summaries[-1]
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "selected dense starts of frozen-spline endpoint-nesting active difference",
        "q": args.q,
        "start_index": args.start_index,
        "end_index": end_index,
        "starts_tested": len(selected),
        "p_line": p_line,
        "p_oriented": p_oriented,
        "p_two_line": p_two_line,
        "edge_cap_length": cap_length,
        "body_steps": args.body_steps,
        "cap_steps": args.cap_steps,
        "subdivision_sweep": sweep,
        "summaries": summaries,
        "rows": rows,
        "largest_subdivision_all_starts_passed": bool(largest["all_starts_passed"]),
        "classification": (
            "ENDPOINT-NESTING DENSE-START SPAN PASSED AT LARGEST SUBDIVISION"
            if largest["all_starts_passed"]
            else "ENDPOINT-NESTING DENSE-START SPAN STILL OPEN AT LARGEST SUBDIVISION"
        ),
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_name(prefix.name + "_rows.csv")
    summary_csv_path = prefix.with_name(prefix.name + "_summary.csv")
    report_path = prefix.with_name(prefix.name + "_report.txt")

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with summary_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Q=550 ENDPOINT-NESTING DENSE-START SPAN AUDIT",
        "=================================================",
        "",
        f"dense starts                         = {args.start_index} .. {end_index} ({len(selected)})",
        f"P_oriented                           = {p_oriented:.12e}",
        f"2 P_line                             = {p_two_line:.12e}",
        f"edge cap DeltaP                      = {cap_length:.12e}",
        f"body steps                           = {args.body_steps}",
        f"cap steps                            = {args.cap_steps}",
        f"subdivision sweep                    = {sweep}",
        "",
        "Architecture",
        "------------",
        "For each dense start independently:",
        "  T_2P = G_edge T_O, hence D=(I-G_edge)T_O.",
        "  The long-body uncertainty is charged once and attenuated by I-G_edge.",
        "  No uncertainty is chained from one start to the next.",
        "",
        "Summaries",
        "---------",
    ]
    for summary in summaries:
        lines.append(
            f"m={summary['residual_subdivisions']:>4} "
            f"passed={summary['starts_passed']:>3}/{summary['starts_tested']} "
            f"minLower={summary['minimum_endpoint_lower']:+.6e} "
            f"at i={summary['minimum_endpoint_lower_index']:>2} "
            f"maxR={summary['maximum_active_radius']:.6e} "
            f"maxDisc={summary['maximum_center_relative_discrepancy']:.3e} "
            f"all={summary['all_starts_passed']}"
        )
    lines += [
        "",
        f"classification = {payload['classification']}",
        "",
        "Scope: dense start nodes only in the frozen-spline linear cocycle;",
        "continuous inter-node coverage and exact nonlinear-orbit validation remain open.",
    ]
    report_path.write_text("\n".join(lines) + "\n")

    print("=" * 78)
    print("\n".join(lines))
    print(f"report      = {report_path.resolve()}")
    print(f"JSON        = {json_path.resolve()}")
    print(f"rows CSV    = {csv_path.resolve()}")
    print(f"summary CSV = {summary_csv_path.resolve()}")


if __name__ == "__main__":
    main()
