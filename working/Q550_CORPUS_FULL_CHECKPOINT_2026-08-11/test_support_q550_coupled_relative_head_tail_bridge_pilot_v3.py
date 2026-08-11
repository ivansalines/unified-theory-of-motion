#!/usr/bin/env python3
"""Q=550 coupled relative head-tail bridge pilot (v3).

This pilot keeps the successful forward head-tail cocycle identity from v2,

    T_L(s+delta) = H_L(s,delta) T_L(s) K(s,delta)^(-1),

but changes the uncertainty architecture in two ways suggested by the v2 logs.

1. Relative memory is preserved.
   The already validated seed transport is represented as

       T_seed_exact = T_seed_center (I + E_seed),

   and each short forward transport is represented as

       H_exact = U_H (I + E_H),
       K_exact = U_K (I + E_K).

   The relative error radii come from the accumulated relative-defect
   integrals; the tail is never converted into an absolute inverse ball.

2. The tail is shared by the two competing endpoints.
   For oriented O and two-line P,

       D_next = T_O,next - T_P,next
              = (A_O - A_P) K^(-1),

   so the same uncertain tail multiplies the *difference*.  We therefore
   validate the active 2x2 endpoint separation directly, instead of enclosing
   T_O and T_P independently and summing two global spectral radii.

This is still a pilot for one forward dense cell (67 -> 68) and still only for
the frozen natural-cubic-spline linear cocycle.  The nonlinear orbit cache is
not yet interval-validated against an exact trajectory.
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

SCRIPT_VERSION = "2026-08-07-q550-coupled-relative-head-tail-bridge-v3"
DEFAULT_BASE = Path("test_support_q550_interval_spline_cocycle_proof.py")
DEFAULT_PILOT = Path("test_support_q550_relative_defect_microbox_pilot_v1_3.py")
DEFAULT_DENSE = Path("support_q550_dense_start_span_closure_raw.npz")
DEFAULT_SEED = Path("support_q550_relative_defect_microbox_pilot_m192_raw.npz")
DEFAULT_SEED_REPORT = Path("support_q550_relative_defect_microbox_pilot_m192_report.txt")
DEFAULT_AUDIT = Path("test_support_mobius_monodromy_audit.py")
DEFAULT_ORBIT = Path("support_feedback_q_curtain_cache/q_550_orbit.npz")
DEFAULT_OUTPUT = Path("support_q550_coupled_relative_head_tail_bridge_pilot")


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
    """Validate the inverse of the chosen floating guide center only.

    This does *not* absorb tail uncertainty.  Tail uncertainty is retained as
    the relative factor (I+E_K)^(-1), shared by both endpoints.
    """
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
    """Read accumulated relative-defect integrals from the m=192 report."""
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


def fallback_seed_relative_eps(
    base: ModuleType,
    pilot: ModuleType,
    center: np.ndarray,
    absolute_radius: float,
) -> tuple[float, str]:
    """Convert an absolute seed ball to a right-relative correction if needed."""
    _, _, inverse_norm, _ = validated_inverse_center(base, pilot, center)
    return outward(inverse_norm * absolute_radius), "absolute-ball fallback"


def seed_relative_eps(
    base: ModuleType,
    pilot: ModuleType,
    *,
    name: str,
    center: np.ndarray,
    absolute_radius: float,
    parsed_alpha: dict[str, float],
) -> tuple[float, str, float | None]:
    alpha = parsed_alpha.get(name)
    if alpha is not None:
        return upward_expm1(base, alpha), "m192 accumulated relative defect", alpha
    eps, source = fallback_seed_relative_eps(
        base, pilot, center, absolute_radius
    )
    return eps, source, None


def head_seed_row_error(
    *,
    head_center: np.ndarray,
    seed_center: np.ndarray,
    head_eps: float,
    seed_eps: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Bound first-two-row error before the common tail.

    Exact pre-tail factor:

        A = U_H (I+E_H) T_seed (I+E_seed)

    around center A0 = U_H T_seed.
    """
    pre_center = head_center @ seed_center
    left_rows = head_center[:2, :]
    pre_rows = pre_center[:2, :]
    left_rows_norm = n2(left_rows)
    seed_norm = n2(seed_center)
    pre_rows_norm = n2(pre_rows)

    # U_H E_H T (I+E_s) + U_H T E_s
    head_term = outward(left_rows_norm * head_eps * seed_norm * (1.0 + seed_eps))
    seed_term = outward(pre_rows_norm * seed_eps)
    row_radius = outward(head_term + seed_term)
    return pre_center, row_radius, {
        "left_rows_norm": left_rows_norm,
        "seed_center_norm": seed_norm,
        "pre_rows_norm": pre_rows_norm,
        "head_row_error_term": head_term,
        "seed_row_error_term": seed_term,
    }


def shared_tail_difference_bound(
    *,
    oriented_pre_center: np.ndarray,
    oriented_pre_row_radius: float,
    two_line_pre_center: np.ndarray,
    two_line_pre_row_radius: float,
    tail_inverse_center: np.ndarray,
    tail_inverse_center_radius: float,
    tail_relative_inverse_eps: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Direct active-block bound for the endpoint difference with shared tail.

    Let Dpre_exact = Dpre_center + dD, with only the first-two-row error needed,
    and let the exact common tail inverse be J = V + dJ.  Then

        Dnext_exact = Dpre_exact J.

    The active 2x2 block uses only first two rows of Dpre and first two columns
    of J, so we avoid a global 9x9 radius.
    """
    dpre_center = oriented_pre_center - two_line_pre_center
    dpre_row_radius = outward(oriented_pre_row_radius + two_line_pre_row_radius)

    v = tail_inverse_center
    v_norm = n2(v)
    v_active_cols_norm = n2(v[:, :2])
    dpre_rows_norm = n2(dpre_center[:2, :])

    # Exact K^-1 = (I+F_K) U_K^-1, ||F_K|| <= f.
    # U_K^-1 itself is enclosed around V with radius rV.
    f = tail_relative_inverse_eps
    r_v = tail_inverse_center_radius
    common_tail_radius = outward(f * v_norm + r_v + f * r_v)

    center_full = dpre_center @ v
    center_active = center_full[:2, :2]

    # (D+dD)(V+dJ)-DV = dD V + D dJ + dD dJ.
    active_radius = outward(
        dpre_row_radius * v_active_cols_norm
        + dpre_rows_norm * common_tail_radius
        + dpre_row_radius * common_tail_radius
    )

    return center_active, active_radius, {
        "dpre_row_radius": dpre_row_radius,
        "tail_inverse_center_norm": v_norm,
        "tail_inverse_active_columns_norm": v_active_cols_norm,
        "dpre_center_rows_norm": dpre_rows_norm,
        "tail_inverse_common_radius": common_tail_radius,
    }


def run_self_test() -> None:
    """Monte-Carlo check of the shared-tail active-block algebra."""
    rng = np.random.default_rng(20260807)
    for _ in range(250):
        uh_o = np.eye(4) + 0.08 * rng.normal(size=(4, 4))
        uh_p = np.eye(4) + 0.08 * rng.normal(size=(4, 4))
        ts_o = np.eye(4) + 0.10 * rng.normal(size=(4, 4))
        ts_p = np.eye(4) + 0.10 * rng.normal(size=(4, 4))
        uk = np.eye(4) + 0.06 * rng.normal(size=(4, 4))
        v = np.linalg.inv(uk)
        eh_o, eh_p, es_o, es_p, ek = 3e-4, 3.2e-4, 4e-4, 4.1e-4, 3e-4
        fk = ek / (1.0 - ek)

        po, ro, _ = head_seed_row_error(
            head_center=uh_o, seed_center=ts_o, head_eps=eh_o, seed_eps=es_o
        )
        pp, rp, _ = head_seed_row_error(
            head_center=uh_p, seed_center=ts_p, head_eps=eh_p, seed_eps=es_p
        )
        center_active, radius, _ = shared_tail_difference_bound(
            oriented_pre_center=po,
            oriented_pre_row_radius=ro,
            two_line_pre_center=pp,
            two_line_pre_row_radius=rp,
            tail_inverse_center=v,
            tail_inverse_center_radius=0.0,
            tail_relative_inverse_eps=fk,
        )

        def perturb(r: float) -> np.ndarray:
            e = rng.normal(size=(4, 4))
            norm = n2(e)
            return e * (r / max(norm, 1e-300))

        for _j in range(20):
            eho = perturb(eh_o)
            ehp = perturb(eh_p)
            eso = perturb(es_o)
            esp = perturb(es_p)
            ekt = perturb(ek)
            exact_o = uh_o @ (np.eye(4) + eho) @ ts_o @ (np.eye(4) + eso)
            exact_p = uh_p @ (np.eye(4) + ehp) @ ts_p @ (np.eye(4) + esp)
            exact = (exact_o - exact_p) @ np.linalg.inv(np.eye(4) + ekt) @ v
            actual = n2(exact[:2, :2] - center_active)
            if actual > radius * (1.0 + 1e-10):
                raise AssertionError(
                    f"shared-tail active bound escaped: {actual:.3e} > {radius:.3e}"
                )
    print("coupled relative shared-tail algebra self-test passed")


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
        p.error("v3 is intentionally forward-only: neighbor must be seed+1")
    if args.short_steps < 20:
        p.error("--short-steps must be at least 20")

    base = load_module(args.base_script, "coupled_v3_base")
    base.MP_DPS = int(args.mp_dps)
    pilot = load_module(args.pilot_script, "coupled_v3_pilot")
    audit = load_module(args.audit_script, "coupled_v3_audit")
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

    parsed_alpha: dict[str, float] = {}
    if args.seed_report.exists():
        parsed_alpha = parse_seed_accumulated_defects(args.seed_report)

    seed_eps: dict[str, float] = {}
    seed_eps_source: dict[str, str] = {}
    seed_alpha_used: dict[str, float | None] = {}
    for name in ("oriented", "two_line"):
        eps, source, alpha = seed_relative_eps(
            base,
            pilot,
            name=name,
            center=seed_centers[name],
            absolute_radius=seed_radii[name],
            parsed_alpha=parsed_alpha,
        )
        seed_eps[name] = eps
        seed_eps_source[name] = source
        seed_alpha_used[name] = alpha

    s0 = float(starts[args.seed_index])
    s1 = float(starts[args.neighbor_index])
    delta = s1 - s0
    if not delta > 0.0:
        raise ValueError("forward dense gap must be positive")
    short_step = delta / args.short_steps

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
    domain_right = s1 + max(p_oriented, 2.0 * p_line)
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

    lengths = {"oriented": p_oriented, "two_line": 2.0 * p_line}

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"seed -> neighbor     = {args.seed_index} -> {args.neighbor_index}")
    print(f"s0                   = {s0:.15f}")
    print(f"s1                   = {s1:.15f}")
    print(f"forward dense gap    = {delta:.15e}")
    print(f"short steps          = {args.short_steps}")
    print(f"short step           = {short_step:.15e}")
    print(f"subdivision sweep    = {sweep}")
    print("architecture         = relative head/seed factors + shared tail on endpoint difference")
    for name in ("oriented", "two_line"):
        print(
            f"seed {name:>9} eps   = {seed_eps[name]:.12e} "
            f"({seed_eps_source[name]})"
        )

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
        head_short: dict[str, Any] = {}
        for name, length in lengths.items():
            head_short[name] = pilot.validate_transport_relative_defect(
                base,
                envelope,
                start=s0 + length,
                step=short_step,
                steps=args.short_steps,
                residual_subdivisions=subdivisions,
                label=f"head-{name}-m{subdivisions}",
            )

        tail_eps = upward_expm1(base, tail_short.accumulated_relative_defect)
        if not tail_eps < 1.0:
            raise FloatingPointError(f"tail relative epsilon >= 1: {tail_eps:.6e}")
        tail_inverse_relative_eps = upward_fraction(
            base, tail_eps, 1.0 - tail_eps
        )

        head_eps = {
            name: upward_expm1(base, head_short[name].accumulated_relative_defect)
            for name in head_short
        }

        tail_inv_center, tail_inv_round_radius, tail_inv_norm, tail_inv_q = validated_inverse_center(
            base, pilot, tail_short.center
        )

        pre: dict[str, tuple[np.ndarray, float, dict[str, float]]] = {}
        for name in ("oriented", "two_line"):
            pre[name] = head_seed_row_error(
                head_center=head_short[name].center,
                seed_center=seed_centers[name],
                head_eps=head_eps[name],
                seed_eps=seed_eps[name],
            )

        active_center, active_radius, shared_diag = shared_tail_difference_bound(
            oriented_pre_center=pre["oriented"][0],
            oriented_pre_row_radius=pre["oriented"][1],
            two_line_pre_center=pre["two_line"][0],
            two_line_pre_row_radius=pre["two_line"][1],
            tail_inverse_center=tail_inv_center,
            tail_inverse_center_radius=tail_inv_round_radius,
            tail_relative_inverse_eps=tail_inverse_relative_eps,
        )

        sep_center = n2(active_center)
        lower = float(np.nextafter(sep_center - active_radius, -np.inf))
        archived_diff = (
            archived["oriented"][args.neighbor_index][:2, :2]
            - archived["two_line"][args.neighbor_index][:2, :2]
        )
        archived_sep = n2(archived_diff)
        center_discrepancy = float(
            np.linalg.norm(active_center - archived_diff, ord="fro")
            / max(np.linalg.norm(archived_diff, ord="fro"), 1e-300)
        )

        row = {
            "residual_subdivisions": subdivisions,
            "tail_absolute_radius": tail_short.radius,
            "tail_relative_eps": tail_eps,
            "tail_inverse_relative_eps": tail_inverse_relative_eps,
            "tail_inverse_center_round_radius": tail_inv_round_radius,
            "tail_inverse_center_banach_q": tail_inv_q,
            "oriented_head_absolute_radius": head_short["oriented"].radius,
            "two_line_head_absolute_radius": head_short["two_line"].radius,
            "oriented_head_relative_eps": head_eps["oriented"],
            "two_line_head_relative_eps": head_eps["two_line"],
            "oriented_seed_relative_eps": seed_eps["oriented"],
            "two_line_seed_relative_eps": seed_eps["two_line"],
            "oriented_pre_tail_row_radius": pre["oriented"][1],
            "two_line_pre_tail_row_radius": pre["two_line"][1],
            "difference_pre_tail_row_radius": shared_diag["dpre_row_radius"],
            "shared_tail_inverse_radius": shared_diag["tail_inverse_common_radius"],
            "active_difference_radius": active_radius,
            "propagated_center_separation": sep_center,
            "archived_center_separation": archived_sep,
            "center_relative_discrepancy": center_discrepancy,
            "endpoint_separation_lower": lower,
            "endpoint_pass": bool(lower > 0.0),
        }
        rows.append(row)

        print(
            f"  m={subdivisions}: epsTail={tail_eps:.6e} fTail={tail_inverse_relative_eps:.6e} "
            f"epsHeadO={head_eps['oriented']:.6e} epsHead2P={head_eps['two_line']:.6e}",
            flush=True,
        )
        print(
            f"           preRowO={pre['oriented'][1]:.6e} preRow2P={pre['two_line'][1]:.6e} "
            f"sharedTailR={shared_diag['tail_inverse_common_radius']:.6e}",
            flush=True,
        )
        print(
            f"           sep_center={sep_center:.6e} activeR={active_radius:.6e} "
            f"lower={lower:+.6e} pass={lower > 0.0}",
            flush=True,
        )
        print(
            f"           archived sep={archived_sep:.6e} center discrepancy={center_discrepancy:.3e} "
            f"tail inverse q={tail_inv_q:.3e}",
            flush=True,
        )

    largest = rows[-1]
    payload = {
        "script_version": SCRIPT_VERSION,
        "scope": "one-forward-cell frozen-spline coupled-relative shared-tail active endpoint bridge",
        "q": args.q,
        "seed_index": args.seed_index,
        "neighbor_index": args.neighbor_index,
        "forward_dense_gap": delta,
        "short_steps": args.short_steps,
        "subdivision_sweep": sweep,
        "seed_relative_eps": seed_eps,
        "seed_relative_eps_source": seed_eps_source,
        "seed_accumulated_relative_defect": seed_alpha_used,
        "rows": rows,
        "largest_subdivision_endpoint_passed": bool(largest["endpoint_pass"]),
        "classification": (
            "COUPLED RELATIVE SHARED-TAIL BRIDGE PASSED AT LARGEST SUBDIVISION"
            if largest["endpoint_pass"]
            else "COUPLED RELATIVE SHARED-TAIL BRIDGE STILL OPEN AT LARGEST SUBDIVISION"
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
        "Q=550 COUPLED RELATIVE HEAD-TAIL BRIDGE PILOT",
        "===============================================",
        "",
        f"seed -> neighbor                    = {args.seed_index} -> {args.neighbor_index}",
        f"forward dense gap                   = {delta:.12e}",
        f"short steps                         = {args.short_steps}",
        f"subdivision sweep                   = {sweep}",
        "",
        "Architecture",
        "------------",
        "H = U_H (I+E_H), T_seed = T_c (I+E_seed), K = U_K (I+E_K)",
        "The same uncertain K^{-1} is applied once to the endpoint difference.",
        "Only the active 2x2 difference is bounded; irrelevant global modes are not charged.",
        "",
        f"seed oriented relative eps          = {seed_eps['oriented']:.12e}",
        f"seed two_line relative eps          = {seed_eps['two_line']:.12e}",
        "",
        "Rows",
        "----",
    ]
    for row in rows:
        lines.append(
            f"m={row['residual_subdivisions']:>4} "
            f"epsTail={row['tail_relative_eps']:.6e} "
            f"epsHeadO={row['oriented_head_relative_eps']:.6e} "
            f"epsHead2P={row['two_line_head_relative_eps']:.6e} "
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
