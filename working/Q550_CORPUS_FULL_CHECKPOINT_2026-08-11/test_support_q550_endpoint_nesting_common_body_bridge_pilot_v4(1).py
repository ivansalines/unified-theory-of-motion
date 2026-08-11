#!/usr/bin/env python3
"""Q=550 endpoint-nesting / common-body bridge pilot (v4).

The v3 bridge showed that the forward head-tail geometry is correct but still
paid two almost identical long-endpoint uncertainty budgets.  At Q=550 the two
long endpoints are nested:

    P_oriented < 2 P_line,
    DeltaP = 2 P_line - P_oriented = 1/720.

For a fixed start s,

    T_2P(s) = G_Delta(s) T_O(s),

where

    G_Delta(s) = Phi(s + 2 P_line, s + P_oriented)

is only the tiny extra head cap.  Therefore

    D(s) = T_O(s) - T_2P(s)
         = (I - G_Delta(s)) T_O(s).

Across one forward dense-start cell s0 -> s1,

    T_O(s1) = H_O T_O(s0) K^{-1},

so the endpoint difference is validated as

    D(s1) = (I-G1) H_O T_O(s0) K^{-1}.

Only ONE long body is carried.  Its relative uncertainty is multiplied on the
left by the small edge factor I-G1 before the active 2x2 separation is tested.
The tail is still shared and retained as a relative inverse factor.

Scope: one forward dense cell (default 67 -> 68) of the frozen natural-cubic-
spline linear cocycle.  The nonlinear orbit cache is not yet interval-validated
against an exact trajectory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mpmath as mp
import numpy as np

SCRIPT_VERSION = "2026-08-07-q550-endpoint-nesting-common-body-bridge-v4"
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_3.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_SEED = Path("support_q550_relative_defect_microbox_pilot_m192_raw.npz")
DEFAULT_SEED_REPORT = Path("support_q550_relative_defect_microbox_pilot_m192_report.txt")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_endpoint_nesting_common_body_bridge_pilot")


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


def upward_fraction(base: ModuleType, numerator: float, denominator: float) -> float:
    if not denominator > 0.0:
        raise FloatingPointError("non-positive denominator in rigorous fraction")
    with mp.workdps(base.MP_DPS):
        value = mp.mpf(numerator) / mp.mpf(denominator)
        _, hi = base.mp_bracket(value, ulps=4)
    return float(hi)


def spectral_upper(base: ModuleType, pilot: ModuleType, matrix: np.ndarray) -> float:
    return float(pilot.box_spectral_upper(base, base.float_box(matrix, ulps=6)))


def validated_inverse_center(
    base: ModuleType,
    pilot: ModuleType,
    center: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Validated inverse of the floating guide center only."""
    n = center.shape[0]
    v = np.linalg.inv(center)
    v_box = base.float_box(v, ulps=8)
    c_box = base.float_box(center, ulps=8)
    identity = base.constant_matrix_box(np.eye(n), exact_zero_one=True)
    residual = base.isub(identity, base.imatmul(v_box, c_box))
    q = float(pilot.box_spectral_upper(base, residual))
    if not q < 1.0:
        raise FloatingPointError(f"guide inverse Banach residual >= 1: {q:.6e}")
    v_norm = spectral_upper(base, pilot, v)
    exact_inv_norm = outward(v_norm / (1.0 - q))
    round_radius = outward(v_norm * q / (1.0 - q))
    return v, round_radius, exact_inv_norm, q


def parse_seed_accumulated_defects(path: Path) -> dict[str, float]:
    text = path.read_text()
    result: dict[str, float] = {}
    current: str | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped in {"line", "second", "oriented", "two_line"}:
            current = stripped
            continue
        if current is not None and "accumulated relative defect" in stripped:
            match = re.search(r"=\s*([+\-0-9.eE]+)", stripped)
            if match:
                result[current] = float(match.group(1))
    return result


def seed_relative_eps(
    base: ModuleType,
    pilot: ModuleType,
    *,
    center: np.ndarray,
    absolute_radius: float,
    parsed_alpha: dict[str, float],
) -> tuple[float, str, float | None]:
    alpha = parsed_alpha.get("oriented")
    if alpha is not None:
        return upward_expm1(base, alpha), "m192 accumulated relative defect", alpha
    _, _, inverse_norm, _ = validated_inverse_center(base, pilot, center)
    return outward(inverse_norm * absolute_radius), "absolute-ball fallback", None


def long_body_error(
    *,
    head_center: np.ndarray,
    seed_center: np.ndarray,
    head_eps: float,
    seed_eps: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Global error for A = H T_seed before the small edge factor.

    H = U_H (I+E_H), T = T_c (I+E_s), so

      A - U_H T_c
        = U_H E_H T_c (I+E_s) + U_H T_c E_s.
    """
    a0 = head_center @ seed_center
    nh = n2(head_center)
    nt = n2(seed_center)
    na = n2(a0)
    head_term = outward(nh * head_eps * nt * (1.0 + seed_eps))
    seed_term = outward(na * seed_eps)
    radius = outward(head_term + seed_term)
    return a0, radius, {
        "head_center_norm": nh,
        "seed_center_norm": nt,
        "body_center_norm": na,
        "head_body_error_term": head_term,
        "seed_body_error_term": seed_term,
    }


def edge_times_body_row_error(
    *,
    edge_transport_center: np.ndarray,
    edge_eps: float,
    body_center: np.ndarray,
    body_radius: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Bound first-two-row error of (I-G) A.

    G = U_G (I+E_G), ||E_G|| <= epsG.
    C = I-G = C0 + dC with C0=I-U_G and

        ||dC|| <= ||U_G|| epsG.

    A = A0 + dA, ||dA|| <= body_radius.
    Only first two rows of C A are needed downstream.
    """
    n = edge_transport_center.shape[0]
    c0 = np.eye(n) - edge_transport_center
    center = c0 @ body_center

    edge_center_norm = n2(edge_transport_center)
    c0_rows_norm = n2(c0[:2, :])
    body_center_norm = n2(body_center)
    edge_abs_radius = outward(edge_center_norm * edge_eps)

    # (C0+dC)(A0+dA)-C0 A0
    # = C0 dA + dC A0 + dC dA.
    edge_body_term = outward(c0_rows_norm * body_radius)
    edge_uncertainty_term = outward(edge_abs_radius * body_center_norm)
    cross_term = outward(edge_abs_radius * body_radius)
    row_radius = outward(edge_body_term + edge_uncertainty_term + cross_term)

    return center, row_radius, {
        "edge_transport_center_norm": edge_center_norm,
        "edge_factor_rows_norm": c0_rows_norm,
        "edge_absolute_radius": edge_abs_radius,
        "body_center_norm": body_center_norm,
        "edge_times_body_error_term": edge_body_term,
        "edge_uncertainty_error_term": edge_uncertainty_term,
        "edge_body_cross_term": cross_term,
    }


def shared_tail_active_bound(
    *,
    pre_tail_center: np.ndarray,
    pre_tail_row_radius: float,
    tail_inverse_center: np.ndarray,
    tail_inverse_center_radius: float,
    tail_relative_inverse_eps: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Apply the single uncertain tail inverse to an already-small difference."""
    v = tail_inverse_center
    v_norm = n2(v)
    v_active_cols_norm = n2(v[:, :2])
    pre_rows_norm = n2(pre_tail_center[:2, :])

    f = tail_relative_inverse_eps
    r_v = tail_inverse_center_radius
    common_tail_radius = outward(f * v_norm + r_v + f * r_v)

    center_full = pre_tail_center @ v
    center_active = center_full[:2, :2]

    active_radius = outward(
        pre_tail_row_radius * v_active_cols_norm
        + pre_rows_norm * common_tail_radius
        + pre_tail_row_radius * common_tail_radius
    )
    return center_active, active_radius, {
        "tail_inverse_center_norm": v_norm,
        "tail_inverse_active_columns_norm": v_active_cols_norm,
        "pre_tail_center_rows_norm": pre_rows_norm,
        "tail_inverse_common_radius": common_tail_radius,
    }


def run_self_test() -> None:
    """Monte-Carlo algebra check for the nested-endpoint common-body bound."""
    rng = np.random.default_rng(20260807)
    for _ in range(300):
        n = 4
        uh = np.eye(n) + 0.06 * rng.normal(size=(n, n))
        ts = np.eye(n) + 0.08 * rng.normal(size=(n, n))
        # Deliberately make the cap close to identity.
        ug = np.eye(n) + 0.004 * rng.normal(size=(n, n))
        uk = np.eye(n) + 0.05 * rng.normal(size=(n, n))
        v = np.linalg.inv(uk)

        eh, es, eg, ek = 4e-4, 5e-4, 2e-5, 4e-4
        fk = ek / (1.0 - ek)

        a0, ra, _ = long_body_error(
            head_center=uh,
            seed_center=ts,
            head_eps=eh,
            seed_eps=es,
        )
        d0, rd, _ = edge_times_body_row_error(
            edge_transport_center=ug,
            edge_eps=eg,
            body_center=a0,
            body_radius=ra,
        )
        c_active, radius, _ = shared_tail_active_bound(
            pre_tail_center=d0,
            pre_tail_row_radius=rd,
            tail_inverse_center=v,
            tail_inverse_center_radius=0.0,
            tail_relative_inverse_eps=fk,
        )

        def perturb(r: float) -> np.ndarray:
            e = rng.normal(size=(n, n))
            norm = n2(e)
            return e * (r / max(norm, 1e-300))

        for _j in range(20):
            e_h = perturb(eh)
            e_s = perturb(es)
            e_g = perturb(eg)
            e_k = perturb(ek)
            h = uh @ (np.eye(n) + e_h)
            t = ts @ (np.eye(n) + e_s)
            g = ug @ (np.eye(n) + e_g)
            k = uk @ (np.eye(n) + e_k)
            exact = (np.eye(n) - g) @ h @ t @ np.linalg.inv(k)
            actual = n2(exact[:2, :2] - c_active)
            if actual > radius * (1.0 + 1e-10):
                raise AssertionError(
                    f"endpoint-nesting active bound escaped: {actual:.3e} > {radius:.3e}"
                )
    print("endpoint-nesting common-body algebra self-test passed")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-script", type=Path, default=DEFAULT_BASE)
    p.add_argument("--pilot-script", type=Path, default=DEFAULT_PILOT)
    p.add_argument("--dense-raw", type=Path, default=DEFAULT_DENSE)
    p.add_argument("--seed-raw", type=Path, default=DEFAULT_SEED)
    p.add_argument("--seed-report", type=Path, default=DEFAULT_SEED_REPORT)
    p.add_argument("--audit-script", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--orbit", type=Path, default=DEFAULT_ORBIT)
    p.add_argument("--q", type=float, default=550.0)
    p.add_argument("--seed-index", type=int, default=67)
    p.add_argument("--neighbor-index", type=int, default=68)
    p.add_argument("--short-steps", type=int, default=440)
    p.add_argument("--cap-steps", type=int, default=16)
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
        p.error("v4 is intentionally forward-only: neighbor must be seed+1")
    if args.short_steps < 20:
        p.error("--short-steps must be at least 20")
    if args.cap_steps < 1:
        p.error("--cap-steps must be positive")

    base = load_module(args.base_script, "nest_v4_base")
    base.MP_DPS = int(args.mp_dps)
    pilot = load_module(args.pilot_script, "nest_v4_pilot")
    audit = load_module(args.audit_script, "nest_v4_audit")
    sweep = parse_sweep(args.subdivision_sweep)

    with np.load(args.dense_raw, allow_pickle=True) as d, np.load(args.seed_raw, allow_pickle=True) as seed:
        starts = np.asarray(d["starts"], dtype=float)
        basis = np.asarray(d["basis"], dtype=float)
        p_line = float(d["p_line"])
        p_oriented = float(d["p_oriented"])
        archived_oriented = np.asarray(d["fine_oriented"], dtype=float)
        archived_two_line = np.asarray(d["fine_two_line"], dtype=float)
        seed_index_stored = int(seed["center_index"])
        seed_center = np.asarray(seed["oriented_center"], dtype=float)
        seed_radius = float(seed["oriented_radius"])

    if seed_index_stored != args.seed_index:
        raise ValueError(
            f"seed raw center={seed_index_stored}, requested seed={args.seed_index}"
        )
    if not (0 <= args.seed_index < args.neighbor_index < starts.size):
        raise ValueError("invalid seed/neighbor indexes")

    p_two_line = 2.0 * p_line
    cap_length = p_two_line - p_oriented
    if not cap_length > 0.0:
        raise ValueError(
            "v4 endpoint nesting requires 2*P_line > P_oriented; "
            f"got delta={cap_length:+.16e}"
        )

    parsed_alpha: dict[str, float] = {}
    if args.seed_report.exists():
        parsed_alpha = parse_seed_accumulated_defects(args.seed_report)
    seed_eps, seed_eps_source, seed_alpha = seed_relative_eps(
        base,
        pilot,
        center=seed_center,
        absolute_radius=seed_radius,
        parsed_alpha=parsed_alpha,
    )

    s0 = float(starts[args.seed_index])
    s1 = float(starts[args.neighbor_index])
    delta = s1 - s0
    if not delta > 0.0:
        raise ValueError("forward dense gap must be positive")
    short_step = delta / args.short_steps
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
    domain_left = s0
    domain_right = s1 + p_two_line
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

    archived_diff = (
        archived_oriented[args.neighbor_index][:2, :2]
        - archived_two_line[args.neighbor_index][:2, :2]
    )
    archived_sep = n2(archived_diff)

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"seed -> neighbor     = {args.seed_index} -> {args.neighbor_index}")
    print(f"s0                   = {s0:.15f}")
    print(f"s1                   = {s1:.15f}")
    print(f"forward dense gap    = {delta:.15e}")
    print(f"P_oriented           = {p_oriented:.15e}")
    print(f"2 P_line             = {p_two_line:.15e}")
    print(f"edge cap DeltaP      = {cap_length:.15e}")
    print(f"short steps          = {args.short_steps}")
    print(f"cap steps            = {args.cap_steps}")
    print(f"subdivision sweep    = {sweep}")
    print("architecture         = (I-G_edge) * H_oriented * T_seed * K_tail^{-1}")
    print(f"seed oriented eps    = {seed_eps:.12e} ({seed_eps_source})")

    rows: list[dict[str, Any]] = []
    for subdivisions in sweep:
        print("=" * 78, flush=True)
        print(f"residual subdivisions={subdivisions}", flush=True)

        tail_short = pilot.validate_transport_relative_defect(
            base,
            envelope,
            start=s0,
            step=short_step,
            steps=args.short_steps,
            residual_subdivisions=subdivisions,
            label=f"tail-m{subdivisions}",
        )
        head_short = pilot.validate_transport_relative_defect(
            base,
            envelope,
            start=s0 + p_oriented,
            step=short_step,
            steps=args.short_steps,
            residual_subdivisions=subdivisions,
            label=f"head-oriented-m{subdivisions}",
        )
        # G1 is the tiny extra cap at the *new* head, from s1+P_oriented
        # to s1+2 P_line.
        edge_short = pilot.validate_transport_relative_defect(
            base,
            envelope,
            start=s1 + p_oriented,
            step=cap_step,
            steps=args.cap_steps,
            residual_subdivisions=subdivisions,
            label=f"edge-cap-m{subdivisions}",
        )

        tail_eps = upward_expm1(base, tail_short.accumulated_relative_defect)
        if not tail_eps < 1.0:
            raise FloatingPointError(f"tail relative epsilon >= 1: {tail_eps:.6e}")
        tail_inverse_relative_eps = upward_fraction(
            base, tail_eps, 1.0 - tail_eps
        )
        head_eps = upward_expm1(base, head_short.accumulated_relative_defect)
        edge_eps = upward_expm1(base, edge_short.accumulated_relative_defect)

        tail_inv_center, tail_inv_round_radius, _, tail_inv_q = validated_inverse_center(
            base, pilot, tail_short.center
        )

        body_center, body_radius, body_diag = long_body_error(
            head_center=head_short.center,
            seed_center=seed_center,
            head_eps=head_eps,
            seed_eps=seed_eps,
        )
        pre_tail_center, pre_tail_row_radius, edge_diag = edge_times_body_row_error(
            edge_transport_center=edge_short.center,
            edge_eps=edge_eps,
            body_center=body_center,
            body_radius=body_radius,
        )
        active_center, active_radius, tail_diag = shared_tail_active_bound(
            pre_tail_center=pre_tail_center,
            pre_tail_row_radius=pre_tail_row_radius,
            tail_inverse_center=tail_inv_center,
            tail_inverse_center_radius=tail_inv_round_radius,
            tail_relative_inverse_eps=tail_inverse_relative_eps,
        )

        sep_center = n2(active_center)
        lower = downward(sep_center - active_radius)
        center_discrepancy = float(
            np.linalg.norm(active_center - archived_diff, ord="fro")
            / max(np.linalg.norm(archived_diff, ord="fro"), 1e-300)
        )

        row = {
            "residual_subdivisions": subdivisions,
            "tail_absolute_radius": tail_short.radius,
            "tail_relative_eps": tail_eps,
            "tail_inverse_relative_eps": tail_inverse_relative_eps,
            "tail_inverse_center_banach_q": tail_inv_q,
            "head_absolute_radius": head_short.radius,
            "head_relative_eps": head_eps,
            "edge_cap_absolute_radius": edge_short.radius,
            "edge_cap_relative_eps": edge_eps,
            "edge_factor_rows_norm": edge_diag["edge_factor_rows_norm"],
            "edge_absolute_radius": edge_diag["edge_absolute_radius"],
            "seed_relative_eps": seed_eps,
            "body_absolute_radius": body_radius,
            "edge_times_body_row_radius": pre_tail_row_radius,
            "shared_tail_inverse_radius": tail_diag["tail_inverse_common_radius"],
            "active_difference_radius": active_radius,
            "propagated_center_separation": sep_center,
            "archived_center_separation": archived_sep,
            "center_relative_discrepancy": center_discrepancy,
            "endpoint_separation_lower": lower,
            "endpoint_pass": bool(lower > 0.0),
        }
        rows.append(row)

        print(
            f"  m={subdivisions}: epsTail={tail_eps:.6e} epsHead={head_eps:.6e} "
            f"epsEdge={edge_eps:.6e}", flush=True
        )
        print(
            f"           edgeRows={edge_diag['edge_factor_rows_norm']:.6e} "
            f"bodyR={body_radius:.6e} edgeBodyR={pre_tail_row_radius:.6e} "
            f"sharedTailR={tail_diag['tail_inverse_common_radius']:.6e}", flush=True
        )
        print(
            f"           sep_center={sep_center:.6e} activeR={active_radius:.6e} "
            f"lower={lower:+.6e} pass={lower > 0.0}", flush=True
        )
        print(
            f"           archived sep={archived_sep:.6e} center discrepancy={center_discrepancy:.3e} "
            f"tail inverse q={tail_inv_q:.3e}", flush=True
        )

    largest = rows[-1]
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "one-forward-cell frozen-spline endpoint-nesting common-body active bridge",
        "q": args.q,
        "seed_index": args.seed_index,
        "neighbor_index": args.neighbor_index,
        "forward_dense_gap": delta,
        "p_line": p_line,
        "p_oriented": p_oriented,
        "p_two_line": p_two_line,
        "edge_cap_length": cap_length,
        "short_steps": args.short_steps,
        "cap_steps": args.cap_steps,
        "subdivision_sweep": sweep,
        "seed_relative_eps": seed_eps,
        "seed_relative_eps_source": seed_eps_source,
        "seed_accumulated_relative_defect": seed_alpha,
        "rows": rows,
        "largest_subdivision_endpoint_passed": bool(largest["endpoint_pass"]),
        "classification": (
            "ENDPOINT-NESTING COMMON-BODY BRIDGE PASSED AT LARGEST SUBDIVISION"
            if largest["endpoint_pass"]
            else "ENDPOINT-NESTING COMMON-BODY BRIDGE STILL OPEN AT LARGEST SUBDIVISION"
        ),
    }

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_name(prefix.name + "_rows.csv")
    report_path = prefix.with_name(prefix.name + "_report.txt")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "Q=550 ENDPOINT-NESTING COMMON-BODY BRIDGE PILOT",
        "=================================================",
        "",
        f"seed -> neighbor                    = {args.seed_index} -> {args.neighbor_index}",
        f"forward dense gap                   = {delta:.12e}",
        f"P_oriented                          = {p_oriented:.12e}",
        f"2 P_line                            = {p_two_line:.12e}",
        f"edge cap DeltaP                     = {cap_length:.12e}",
        f"short steps                         = {args.short_steps}",
        f"cap steps                           = {args.cap_steps}",
        f"subdivision sweep                   = {sweep}",
        "",
        "Architecture",
        "------------",
        "T_2P = G_edge T_O, hence D = T_O - T_2P = (I-G_edge) T_O.",
        "Across the forward start cell: D1=(I-G1) H_O T_O,seed K_tail^{-1}.",
        "Only one long-body uncertainty is charged, and it is attenuated by I-G1.",
        "",
        f"seed oriented relative eps          = {seed_eps:.12e}",
        "",
        "Rows",
        "----",
    ]
    for row in rows:
        lines.append(
            f"m={row['residual_subdivisions']:>4} "
            f"epsTail={row['tail_relative_eps']:.6e} "
            f"epsHead={row['head_relative_eps']:.6e} "
            f"epsEdge={row['edge_cap_relative_eps']:.6e} "
            f"edgeRows={row['edge_factor_rows_norm']:.6e} "
            f"activeR={row['active_difference_radius']:.6e} "
            f"sep={row['propagated_center_separation']:.6e} "
            f"lower={row['endpoint_separation_lower']:+.6e} "
            f"pass={row['endpoint_pass']}"
        )
    lines += [
        "",
        f"classification = {payload['classification']}",
        "",
        "Scope: frozen-spline linear cocycle only; nonlinear orbit not yet validated.",
    ]
    report_path.write_text("\n".join(lines) + "\n")

    print("=" * 78)
    print("\n".join(lines))
    print(f"report = {report_path.resolve()}")
    print(f"JSON   = {json_path.resolve()}")
    print(f"CSV    = {csv_path.resolve()}")


if __name__ == "__main__":
    main()
