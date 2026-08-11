#!/usr/bin/env python3
"""Q=550 forward head-tail cocycle bridge pilot (v2).

This replaces the symmetric start-coordinate/lognorm bridge by an oriented
cocycle update.  For a long transport

    T_L(s) = Phi(s+L, s)

and a forward start increment delta > 0,

    T_L(s+delta)
      = H_L(s,delta) T_L(s) K(s,delta)^(-1),

where

    H_L = Phi(s+L+delta, s+L)     (the head advances),
    K   = Phi(s+delta, s)         (the tail advances).

Both short transports are validated in the same moving-frame relative-defect
architecture that already passed at the seed center.  The uncertainty of the
seed transport is *carried forward* through the product; it is not reset and
it is not replaced by an additive start-coordinate forcing term.

Default scope is deliberately one forward dense cell, center 67 -> 68, and
only the limiting long endpoints `oriented` and `two_line`.  If this closes,
the same mechanism can be chained forward from rigorously validated seeds.

Scope remains the frozen natural-cubic-spline linear cocycle.  The nonlinear
orbit cache is not yet validated against an exact trajectory.
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

import numpy as np

SCRIPT_VERSION = "2026-08-07-q550-forward-head-tail-cocycle-bridge-v2"
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_3.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_SEED = Path("support_q550_relative_defect_microbox_pilot_m192_raw.npz")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_forward_head_tail_bridge_pilot")


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


def parse_sweep(text: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    if not vals or vals[0] < 1:
        raise ValueError("subdivision sweep must contain positive integers")
    return vals


def spectral_upper(base: ModuleType, pilot: ModuleType, matrix: np.ndarray) -> float:
    return float(pilot.box_spectral_upper(base, base.float_box(matrix, ulps=6)))


def product_ball(
    base: ModuleType,
    pilot: ModuleType,
    a_center: np.ndarray,
    a_radius: float,
    b_center: np.ndarray,
    b_radius: float,
) -> tuple[np.ndarray, float]:
    """Enclose (A_c+dA)(B_c+dB) in spectral norm."""
    center = a_center @ b_center
    na = spectral_upper(base, pilot, a_center)
    nb = spectral_upper(base, pilot, b_center)
    radius = outward(na * b_radius + a_radius * nb + a_radius * b_radius)
    return center, radius


def inverse_ball(
    base: ModuleType,
    pilot: ModuleType,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Validated inverse ball around a numerical inverse.

    First validates the numerical inverse V of the exact center C using a
    Banach residual q = ||I - V C||.  Then perturbs from C to the full input
    ball ||K-C|| <= radius.
    """
    n = center.shape[0]
    v = np.linalg.inv(center)
    v_box = base.float_box(v, ulps=8)
    c_box = base.float_box(center, ulps=8)
    identity = base.constant_matrix_box(np.eye(n), exact_zero_one=True)
    residual_box = base.isub(identity, base.imatmul(v_box, c_box))
    q = float(pilot.box_spectral_upper(base, residual_box))
    if not q < 1.0:
        raise FloatingPointError(f"center inverse Banach residual >= 1: {q:.6e}")

    v_norm = spectral_upper(base, pilot, v)
    exact_center_inverse_norm = outward(v_norm / (1.0 - q))
    center_inverse_error = outward(v_norm * q / (1.0 - q))

    eta = outward(exact_center_inverse_norm * radius)
    if not eta < 1.0:
        raise FloatingPointError(
            f"inverse-ball perturbation failed: ||C^-1|| r = {eta:.6e} >= 1"
        )
    perturb_error = outward(
        exact_center_inverse_norm * exact_center_inverse_norm * radius / (1.0 - eta)
    )
    total_radius = outward(center_inverse_error + perturb_error)
    return v, total_radius, {
        "center_inverse_banach_residual": q,
        "validated_center_inverse_norm": exact_center_inverse_norm,
        "input_ball_eta": eta,
        "inverse_center_roundoff_radius": center_inverse_error,
        "inverse_input_perturbation_radius": perturb_error,
    }


def propagate_long_ball_forward(
    base: ModuleType,
    pilot: ModuleType,
    *,
    long_center: np.ndarray,
    long_radius: float,
    head_center: np.ndarray,
    head_radius: float,
    tail_center: np.ndarray,
    tail_radius: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """T_next = H T K^{-1}; old uncertainty survives inside the product."""
    tail_inv_center, tail_inv_radius, inv_diag = inverse_ball(
        base, pilot, tail_center, tail_radius
    )
    ht_center, ht_radius = product_ball(
        base, pilot, head_center, head_radius, long_center, long_radius
    )
    next_center, next_radius = product_ball(
        base, pilot, ht_center, ht_radius, tail_inv_center, tail_inv_radius
    )
    diag = {
        **inv_diag,
        "seed_radius": float(long_radius),
        "head_radius": float(head_radius),
        "tail_radius": float(tail_radius),
        "after_head_times_seed_radius": float(ht_radius),
        "propagated_radius": float(next_radius),
    }
    return next_center, next_radius, diag


def run_self_test() -> None:
    """Monte-Carlo algebra sanity check for product/inverse uncertainty formulas."""
    rng = np.random.default_rng(20260807)
    # Small near-identity 3x3 test, independent of project modules.
    def n2(a: np.ndarray) -> float:
        return float(np.linalg.norm(a, 2))

    for _ in range(100):
        h = np.eye(3) + 0.03*rng.normal(size=(3,3))
        t = np.eye(3) + 0.05*rng.normal(size=(3,3))
        k = np.eye(3) + 0.03*rng.normal(size=(3,3))
        rh, rt, rk = 2e-4, 3e-4, 2e-4
        # Non-rigorous mirror of the formulas, enough to detect algebra/sign bugs.
        ki = np.linalg.inv(k)
        eta = n2(ki)*rk
        ri = n2(ki)**2*rk/(1-eta)
        ht = h@t
        rht = n2(h)*rt + rh*n2(t) + rh*rt
        c = ht@ki
        rc = n2(ht)*ri + rht*n2(ki) + rht*ri
        for _j in range(10):
            def perturb(r):
                e=rng.normal(size=(3,3)); e*=r/max(n2(e),1e-300); return e
            exact=(h+perturb(rh))@(t+perturb(rt))@np.linalg.inv(k+perturb(rk))
            if n2(exact-c) > rc*(1+1e-10):
                raise AssertionError("head-tail ball algebra self-test failed")
    print("forward head-tail cocycle algebra self-test passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--pilot-script", type=Path, default=DEFAULT_PILOT)
    p.add_argument("--dense-raw", type=Path, default=DEFAULT_DENSE)
    p.add_argument("--seed-raw", type=Path, default=DEFAULT_SEED)
    p.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT)
    p.add_argument("--q", type=float, default=550.0)
    p.add_argument("--seed-index", type=int, default=67)
    p.add_argument("--neighbor-index", type=int, default=68)
    p.add_argument("--short-steps", type=int, default=440)
    p.add_argument("--subdivision-sweep", default="16,32,64")
    p.add_argument("--mp-dps", type=int, default=80)
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
    if args.neighbor_index != args.seed_index + 1:
        p.error("v2 is intentionally forward-only: --neighbor-index must equal seed-index + 1")
    if args.short_steps < 20:
        p.error("--short-steps must be at least 20")

    base = load_module(args.base_script, "headtail_base")
    base.MP_DPS = int(args.mp_dps)
    pilot = load_module(args.pilot_script, "headtail_pilot")
    audit = load_module(args.audit_script, "headtail_audit")
    sweep = parse_sweep(args.subdivision_sweep)

    with np.load(args.dense_raw, allow_pickle=True) as d, np.load(args.seed_raw, allow_pickle=True) as seed:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])
        archived = {
            "oriented": np.asarray(d["fine_oriented"], dtype=float),
            "two_line": np.asarray(d["fine_two_line"], dtype=float),
        }
        seed_index_stored = int(seed["center_index"])
        seed_centers = {
            name: np.asarray(seed[f"{name}_center"], dtype=float)
            for name in archived
        }
        seed_radii = {
            name: float(seed[f"{name}_radius"])
            for name in archived
        }

    if seed_index_stored != args.seed_index:
        raise ValueError(
            f"seed raw center={seed_index_stored}, requested seed={args.seed_index}"
        )
    if not (0 <= args.seed_index < args.neighbor_index < starts.size):
        raise ValueError("invalid seed/neighbor indexes")

    s0 = float(starts[args.seed_index])
    s1 = float(starts[args.neighbor_index])
    delta = s1-s0
    if not delta > 0.0:
        raise ValueError("forward dense gap must be positive")
    short_step = delta/args.short_steps

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
    kappa0 = (55.0/6.0)*(10.0/args.q)
    domain_left = s0
    domain_right = s1 + max(p_oriented, 2.0*p_line)
    envelope = pilot.LocalCubicGeneratorEnvelope(
        base, audit, spline, np.asarray(tail.s, dtype=float),
        domain_left=domain_left, domain_right=domain_right,
        kappa0=kappa0, params=params, basis=basis, basis_box=basis_box,
        left_matrix=inverse, left_box=inverse_box,
    )

    lengths = {"oriented": p_oriented, "two_line": 2.0*p_line}

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"seed -> neighbor     = {args.seed_index} -> {args.neighbor_index}")
    print(f"s0                   = {s0:.15f}")
    print(f"s1                   = {s1:.15f}")
    print(f"forward dense gap    = {delta:.15e}")
    print(f"short steps          = {args.short_steps}")
    print(f"short step           = {short_step:.15e}")
    print(f"subdivision sweep    = {sweep}")
    print("architecture         = H_head * T_seed * K_tail^{-1}")

    rows: list[dict[str, Any]] = []
    for subdivisions in sweep:
        print("="*78, flush=True)
        print(f"residual subdivisions={subdivisions}", flush=True)

        # The same forward tail advance is shared by both long transports.
        tail_short = pilot.validate_transport_relative_defect(
            base, envelope, start=s0, step=short_step, steps=args.short_steps,
            residual_subdivisions=subdivisions, label=f"tail-m{subdivisions}",
        )
        head_short = {}
        for name, length in lengths.items():
            head_short[name] = pilot.validate_transport_relative_defect(
                base, envelope, start=s0+length, step=short_step,
                steps=args.short_steps, residual_subdivisions=subdivisions,
                label=f"head-{name}-m{subdivisions}",
            )

        propagated = {}
        for name in ("oriented", "two_line"):
            c, r, diag = propagate_long_ball_forward(
                base, pilot,
                long_center=seed_centers[name], long_radius=seed_radii[name],
                head_center=head_short[name].center, head_radius=head_short[name].radius,
                tail_center=tail_short.center, tail_radius=tail_short.radius,
            )
            propagated[name] = (c, r, diag)

        oc, orad, odiag = propagated["oriented"]
        tc, trad, tdiag = propagated["two_line"]
        center_sep = float(np.linalg.norm(oc[:2,:2]-tc[:2,:2], ord=2))
        lower = center_sep-orad-trad
        archived_sep = float(np.linalg.norm(
            archived["oriented"][args.neighbor_index][:2,:2]
            - archived["two_line"][args.neighbor_index][:2,:2], ord=2
        ))
        discrepancy_o = pilot.relative_matrix_difference(
            oc, archived["oriented"][args.neighbor_index]
        )
        discrepancy_t = pilot.relative_matrix_difference(
            tc, archived["two_line"][args.neighbor_index]
        )

        row = {
            "residual_subdivisions": subdivisions,
            "tail_short_radius": tail_short.radius,
            "oriented_head_short_radius": head_short["oriented"].radius,
            "two_line_head_short_radius": head_short["two_line"].radius,
            "oriented_propagated_radius": orad,
            "two_line_propagated_radius": trad,
            "propagated_center_separation": center_sep,
            "archived_center_separation": archived_sep,
            "endpoint_separation_lower": lower,
            "endpoint_pass": bool(lower > 0.0),
            "oriented_archived_relative_discrepancy": discrepancy_o,
            "two_line_archived_relative_discrepancy": discrepancy_t,
            "tail_inverse_eta": odiag["input_ball_eta"],
            "tail_inverse_radius": odiag["inverse_center_roundoff_radius"] + odiag["inverse_input_perturbation_radius"],
            "seed_oriented_radius": seed_radii["oriented"],
            "seed_two_line_radius": seed_radii["two_line"],
        }
        rows.append(row)
        print(
            f"  m={subdivisions}: rTail={tail_short.radius:.6e} "
            f"rHeadO={head_short['oriented'].radius:.6e} "
            f"rHead2P={head_short['two_line'].radius:.6e}", flush=True
        )
        print(
            f"           rO={orad:.6e} r2P={trad:.6e} "
            f"sep_center={center_sep:.6e} lower={lower:+.6e} pass={lower>0.0}",
            flush=True,
        )
        print(
            f"           archived discrepancies: O={discrepancy_o:.3e} 2P={discrepancy_t:.3e} "
            f"tail inverse eta={odiag['input_ball_eta']:.3e}", flush=True
        )

    largest = rows[-1]
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "one-forward-cell frozen-spline head-tail cocycle bridge",
        "q": args.q,
        "seed_index": args.seed_index,
        "neighbor_index": args.neighbor_index,
        "forward_dense_gap": delta,
        "short_steps": args.short_steps,
        "subdivision_sweep": sweep,
        "rows": rows,
        "largest_subdivision_endpoint_passed": bool(largest["endpoint_pass"]),
        "classification": (
            "FORWARD HEAD-TAIL COCOYCLE BRIDGE PASSED AT LARGEST SUBDIVISION"
            if largest["endpoint_pass"] else
            "FORWARD HEAD-TAIL COCOYCLE BRIDGE STILL OPEN AT LARGEST SUBDIVISION"
        ),
    }

    prefix=args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path=prefix.with_suffix(".json")
    csv_path=prefix.with_name(prefix.name+"_rows.csv")
    report_path=prefix.with_name(prefix.name+"_report.txt")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    with csv_path.open("w", newline="") as fh:
        writer=csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    lines=[
        f"Script version: {SCRIPT_VERSION}", "",
        "Q=550 FORWARD HEAD-TAIL COCOYCLE BRIDGE PILOT",
        "===============================================", "",
        f"seed -> neighbor                    = {args.seed_index} -> {args.neighbor_index}",
        f"forward dense gap                   = {delta:.12e}",
        f"short steps                         = {args.short_steps}",
        f"subdivision sweep                   = {sweep}", "",
        "Identity used", "-------------",
        "T_L(s+delta) = H_head T_L(s) K_tail^{-1}",
        "Both H_head and K_tail are forward short transports.",
        "The seed radius is carried through the product as mechanical memory.", "",
        "Rows", "----",
    ]
    for row in rows:
        lines.append(
            f"m={row['residual_subdivisions']:>4} "
            f"rTail={row['tail_short_radius']:.6e} "
            f"rHeadO={row['oriented_head_short_radius']:.6e} "
            f"rHead2P={row['two_line_head_short_radius']:.6e} "
            f"rO={row['oriented_propagated_radius']:.6e} "
            f"r2P={row['two_line_propagated_radius']:.6e} "
            f"lower={row['endpoint_separation_lower']:+.6e} "
            f"pass={row['endpoint_pass']}"
        )
    lines += ["", f"classification = {payload['classification']}", "",
              "Scope: frozen-spline linear cocycle only; nonlinear orbit not yet validated."]
    report_path.write_text("\n".join(lines)+"\n")
    print("="*78)
    print("\n".join(lines))
    print(f"report = {report_path.resolve()}")
    print(f"JSON   = {json_path.resolve()}")
    print(f"CSV    = {csv_path.resolve()}")


if __name__ == "__main__":
    main()
