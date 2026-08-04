#!/usr/bin/env python3
"""Phase-resolved audit of the leading tangent mode.

The purpose is not to locate another threshold. It is to watch the already
validated unstable/near-neutral mode move through one reference cycle.

Recommended comparison
----------------------
Q = 522.25
    close to the moving stability curtain.

Q = 550
    safely beyond the curtain, with a robust positive exponent.

For each cached central orbit the script:

1. integrates the FULL analytic tangent equation through all available
   periods, with gauge projection and renormalization once per period;

2. uses the final --analysis-periods as an ensemble of aligned cycles,
   avoiding interpretation of one isolated trajectory phase;

3. resolves the instantaneous logarithmic growth exactly as

       g_FULL = g_OPEN + g_PHASE + g_KAPPA,

   where:
       OPEN   contains every tangent term except the two x->support returns;
       PHASE  contains only delta x -> delta Theta_dot;
       KAPPA  contains only delta x -> delta kappa_eff in the support rows;

4. measures where the tangent norm is carried:
       LOCAL      = (delta z, delta z_dot),
       RADIAL     = (delta y1, delta y1_dot, delta y2, delta y2_dot),
       ROTATING   = phase variables after gauge projection;

5. records physical tangent observables:
       delta x,
       delta Delta,
       delta rho norm,
       delta kappa_eff.

The plotted instantaneous growth is multiplied by T_ref. Therefore its
average over one normalized cycle equals the exponent per reference period.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import expit


ALPHA, BETA, GAMMA = 1.0, 0.8, 0.2
SCRIPT_VERSION = "2026-08-03-phase-resolved-tangent-cycle-audit-v1"


@dataclass(frozen=True)
class ModelParameters:
    phi0: float = 0.0
    e_reserve: float = 0.01
    barrier_B: float = 0.3
    mu_x: float = 0.05
    gamma_x: float = 0.3
    gamma_rho: float = 0.3
    m_x: float = 1.0


@dataclass
class OrbitData:
    q: float
    period_reference: float
    times: np.ndarray
    states: np.ndarray
    periods: int
    source: Path


@dataclass
class CaseResult:
    q: float
    period_reference: float
    orbit_periods: int
    alignment_periods: int
    analysis_periods: int
    phase_fraction: np.ndarray
    phase_angle: np.ndarray
    profile_mean: dict[str, np.ndarray]
    profile_std: dict[str, np.ndarray]
    period_lambda: np.ndarray
    period_branch_lambda: dict[str, np.ndarray]
    summary: dict[str, float]
    source: Path


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return expit(z)


def dU(rho: float, e_reserve: float) -> float:
    return (
        2.0 * ALPHA * rho
        - 3.0 * BETA * rho**2
        + 4.0 * GAMMA * rho**3
        - e_reserve**2 / rho**3
    )


def ddU(rho: float, e_reserve: float) -> float:
    return (
        2.0 * ALPHA
        - 6.0 * BETA * rho
        + 12.0 * GAMMA * rho**2
        + 3.0 * e_reserve**2 / rho**4
    )


def dV_barrier(x: float, barrier_B: float) -> float:
    return -barrier_B * (1.0 / x - 1.0 / (1.0 - x))


def ddV_barrier(x: float, barrier_B: float) -> float:
    return barrier_B * (1.0 / x**2 + 1.0 / (1.0 - x) ** 2)


def internal_to_physical(state: Sequence[float]) -> np.ndarray:
    internal = np.asarray(state, dtype=float)
    if internal.shape != (11,):
        raise ValueError("State must contain exactly 11 components.")

    rho1 = np.exp(internal[0])
    rho2 = np.exp(internal[4])
    x = sigmoid(internal[8])
    g = x * (1.0 - x)

    physical = internal.copy()
    physical[0] = rho1
    physical[1] = rho1 * internal[1]
    physical[4] = rho2
    physical[5] = rho2 * internal[5]
    physical[8] = x
    physical[9] = g * internal[9]
    return physical


def physical_to_internal(state: Sequence[float]) -> np.ndarray:
    physical = np.asarray(state, dtype=float)
    if physical.shape != (11,):
        raise ValueError("State must contain exactly 11 components.")

    rho1, rho1_dot = physical[0], physical[1]
    rho2, rho2_dot = physical[4], physical[5]
    x, x_dot = physical[8], physical[9]

    if rho1 <= 0.0 or rho2 <= 0.0:
        raise ValueError("rho1 and rho2 must be positive.")
    if not 0.0 < x < 1.0:
        raise ValueError("x must remain strictly inside (0, 1).")

    g = x * (1.0 - x)
    internal = physical.copy()
    internal[0] = np.log(rho1)
    internal[1] = rho1_dot / rho1
    internal[4] = np.log(rho2)
    internal[5] = rho2_dot / rho2
    internal[8] = np.log(x / (1.0 - x))
    internal[9] = x_dot / g
    return internal


def parse_case(raw: str) -> tuple[float, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "--case must use the form Q=path/to/orbit.npz"
        )
    q_text, path_text = raw.split("=", 1)
    try:
        q = float(q_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid Q in --case: {q_text!r}"
        ) from exc

    path = Path(path_text.strip())
    if not path_text.strip():
        raise argparse.ArgumentTypeError("The orbit path is empty.")
    return q, path


def load_orbit(case_q: float, path: Path) -> OrbitData:
    if not path.exists():
        raise FileNotFoundError(f"Orbit cache not found: {path}")

    archive = np.load(path)
    required = {"period_reference", "times", "states"}
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(
            f"{path} is missing required arrays: {', '.join(missing)}"
        )

    period_reference = float(
        np.asarray(archive["period_reference"]).item()
    )
    times = np.asarray(archive["times"], dtype=float)
    states = np.asarray(archive["states"], dtype=float)

    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"{path}: times must be one-dimensional.")
    if states.ndim != 2 or states.shape != (11, times.size):
        raise ValueError(
            f"{path}: states must have shape (11, number_of_times)."
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(states)):
        raise FloatingPointError(f"{path}: orbit contains non-finite data.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path}: times must be strictly increasing.")

    duration = float(times[-1] - times[0])
    period_count_float = duration / period_reference
    periods = int(round(period_count_float))
    if not np.isclose(
        period_count_float,
        periods,
        rtol=2e-10,
        atol=2e-8,
    ):
        raise ValueError(
            f"{path}: duration is not an integer number of reference periods: "
            f"{period_count_float:.12f}"
        )

    if "q" in archive.files:
        stored_q = float(np.asarray(archive["q"]).item())
        if not np.isclose(stored_q, case_q, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"{path}: command Q={case_q:g}, stored Q={stored_q:g}."
            )

    times = times - times[0]
    return OrbitData(
        q=case_q,
        period_reference=period_reference,
        times=times,
        states=states,
        periods=periods,
        source=path,
    )


def gauge_project(vector: np.ndarray) -> np.ndarray:
    projected = np.asarray(vector, dtype=float).copy()
    delta_phase = projected[2] + projected[6] - projected[10]
    projected[2] = delta_phase / 3.0
    projected[6] = delta_phase / 3.0
    projected[10] = -delta_phase / 3.0
    return projected


def weighted_vector(vector: np.ndarray, omega_scale: float) -> np.ndarray:
    weighted = gauge_project(vector)
    weighted = weighted.copy()
    weighted[[1, 3, 5, 7, 9]] /= omega_scale
    return weighted


def weighted_norm(vector: np.ndarray, omega_scale: float) -> float:
    return float(np.linalg.norm(weighted_vector(vector, omega_scale)))


def normalized_initial_x_direction(
    central_state: np.ndarray,
    physical_perturbation: float,
    omega_scale: float,
) -> np.ndarray:
    physical = internal_to_physical(central_state)
    x0 = float(physical[8])

    if not 0.0 < x0 - physical_perturbation < x0 + physical_perturbation < 1.0:
        raise ValueError("Initial x perturbation leaves (0, 1).")

    plus = physical.copy()
    minus = physical.copy()
    plus[8] = x0 + physical_perturbation
    minus[8] = x0 - physical_perturbation

    deviation = 0.5 * (
        physical_to_internal(plus) - physical_to_internal(minus)
    )
    deviation = gauge_project(deviation)
    norm = weighted_norm(deviation, omega_scale)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Invalid initial tangent norm.")
    return deviation / norm


def jacobian_components(
    state: np.ndarray,
    *,
    kappa0: float,
    params: ModelParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return OPEN, PHASE-return, and KAPPA-return Jacobian components."""
    (
        y1,
        v1,
        theta1,
        omega1,
        y2,
        v2,
        theta2,
        omega2,
        z,
        vz,
        Theta,
    ) = state

    rho1 = np.exp(y1)
    rho2 = np.exp(y2)
    x = sigmoid(z)
    g = x * (1.0 - x)

    delta = theta1 + theta2 - params.phi0 - Theta
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)

    kappa_eff = kappa0 * x
    ratio21 = rho2 / rho1
    ratio12 = rho1 / rho2

    open_j = np.zeros((11, 11), dtype=float)
    phase_j = np.zeros((11, 11), dtype=float)
    kappa_j = np.zeros((11, 11), dtype=float)

    # y1_dot = v1
    open_j[0, 1] = 1.0

    log_derivative_u1 = (
        ddU(rho1, params.e_reserve)
        - dU(rho1, params.e_reserve) / rho1
    )
    radial_1 = kappa_eff * ratio21 * cos_delta
    angular_1 = kappa_eff * ratio21 * sin_delta

    open_j[1, 0] = -log_derivative_u1 + radial_1
    open_j[1, 1] = -params.gamma_rho - 2.0 * v1
    open_j[1, 2] = angular_1
    open_j[1, 3] = 2.0 * omega1
    open_j[1, 4] = -radial_1
    open_j[1, 6] = angular_1
    open_j[1, 10] = -angular_1
    kappa_j[1, 8] = -kappa0 * g * ratio21 * cos_delta

    # theta1_dot = omega1
    open_j[2, 3] = 1.0

    open_j[3, 0] = -angular_1
    open_j[3, 1] = -2.0 * omega1
    open_j[3, 2] = radial_1
    open_j[3, 3] = -2.0 * v1
    open_j[3, 4] = angular_1
    open_j[3, 6] = radial_1
    open_j[3, 10] = -radial_1
    kappa_j[3, 8] = kappa0 * g * ratio21 * sin_delta

    # y2_dot = v2
    open_j[4, 5] = 1.0

    log_derivative_u2 = (
        ddU(rho2, params.e_reserve)
        - dU(rho2, params.e_reserve) / rho2
    )
    radial_2 = kappa_eff * ratio12 * cos_delta
    angular_2 = kappa_eff * ratio12 * sin_delta

    open_j[5, 0] = -radial_2
    open_j[5, 2] = angular_2
    open_j[5, 4] = -log_derivative_u2 + radial_2
    open_j[5, 5] = -params.gamma_rho - 2.0 * v2
    open_j[5, 6] = angular_2
    open_j[5, 7] = 2.0 * omega2
    open_j[5, 10] = -angular_2
    kappa_j[5, 8] = -kappa0 * g * ratio12 * cos_delta

    # theta2_dot = omega2
    open_j[6, 7] = 1.0

    open_j[7, 0] = angular_2
    open_j[7, 2] = radial_2
    open_j[7, 4] = -angular_2
    open_j[7, 5] = -2.0 * omega2
    open_j[7, 6] = radial_2
    open_j[7, 7] = -2.0 * v2
    open_j[7, 10] = -radial_2
    kappa_j[7, 8] = kappa0 * g * ratio12 * sin_delta

    # z_dot = vz
    open_j[8, 9] = 1.0

    activity = rho1 * rho2 * sin_delta
    activity_cos = rho1 * rho2 * cos_delta
    activity_factor = (
        params.mu_x * activity
        / (params.m_x * g)
    )
    phase_factor = (
        params.mu_x * activity_cos
        / (params.m_x * g)
    )

    numerator = (
        -dV_barrier(x, params.barrier_B)
        + params.mu_x * activity
        - params.gamma_x * g * vz
    ) / params.m_x

    open_j[9, 0] = activity_factor
    open_j[9, 2] = phase_factor
    open_j[9, 4] = activity_factor
    open_j[9, 6] = phase_factor
    open_j[9, 8] = (
        (
            -ddV_barrier(x, params.barrier_B)
            - params.gamma_x * (1.0 - 2.0 * x) * vz
        ) / params.m_x
        - numerator * (1.0 - 2.0 * x) / g
        + 2.0 * g * vz**2
    )
    open_j[9, 9] = (
        -params.gamma_x / params.m_x
        - 2.0 * (1.0 - 2.0 * x) * vz
    )
    open_j[9, 10] = -phase_factor

    # The only PHASE-return term: delta z -> delta Theta_dot.
    phase_j[10, 8] = g

    if not (
        np.all(np.isfinite(open_j))
        and np.all(np.isfinite(phase_j))
        and np.all(np.isfinite(kappa_j))
    ):
        raise FloatingPointError("Non-finite Jacobian component.")

    return open_j, phase_j, kappa_j


def prepare_orbit_lattice(
    orbit: OrbitData,
    steps_per_period: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    total_steps = orbit.periods * steps_per_period
    recorded_start = float(orbit.times[0])
    recorded_end = float(orbit.times[-1])
    recorded_duration = recorded_end - recorded_start
    time_step = recorded_duration / total_steps

    spline = CubicSpline(
        orbit.times,
        orbit.states,
        axis=1,
        bc_type="natural",
        extrapolate=False,
    )

    node_times = (
        recorded_start
        + np.arange(total_steps + 1, dtype=float) * time_step
    )
    node_times[0] = recorded_start
    node_times[-1] = recorded_end
    midpoint_times = 0.5 * (node_times[:-1] + node_times[1:])

    node_times = np.clip(node_times, recorded_start, recorded_end)
    midpoint_times = np.clip(
        midpoint_times,
        recorded_start,
        recorded_end,
    )

    nodes = np.asarray(spline(node_times), dtype=float).T
    midpoints = np.asarray(spline(midpoint_times), dtype=float).T

    if nodes.shape != (total_steps + 1, 11):
        raise RuntimeError("Unexpected node lattice shape.")
    if midpoints.shape != (total_steps, 11):
        raise RuntimeError("Unexpected midpoint lattice shape.")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(midpoints)):
        raise FloatingPointError("Orbit lattice contains non-finite data.")

    return nodes, midpoints, time_step


def apply_jacobian(
    vector: np.ndarray,
    state: np.ndarray,
    kappa0: float,
    params: ModelParameters,
) -> np.ndarray:
    open_j, phase_j, kappa_j = jacobian_components(
        state,
        kappa0=kappa0,
        params=params,
    )
    return (open_j + phase_j + kappa_j) @ vector


def growth_contribution(
    vector: np.ndarray,
    derivative: np.ndarray,
    omega_scale: float,
) -> float:
    weighted = weighted_vector(vector, omega_scale)
    weighted_derivative = weighted_vector(derivative, omega_scale)
    denominator = float(np.dot(weighted, weighted))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise FloatingPointError("Invalid tangent norm in growth contribution.")
    return float(np.dot(weighted, weighted_derivative) / denominator)


def state_group_shares(
    vector: np.ndarray,
    omega_scale: float,
) -> tuple[float, float, float]:
    weighted = weighted_vector(vector, omega_scale)
    squares = weighted**2
    total = float(np.sum(squares))
    if total <= 0.0:
        raise FloatingPointError("Invalid weighted tangent norm.")

    local = float(np.sum(squares[[8, 9]]) / total)
    radial = float(np.sum(squares[[0, 1, 4, 5]]) / total)
    rotating = float(np.sum(squares[[2, 3, 6, 7, 10]]) / total)
    return local, radial, rotating


def physical_tangent_observables(
    vector: np.ndarray,
    central_state: np.ndarray,
    kappa0: float,
    omega_scale: float,
) -> dict[str, float]:
    projected = gauge_project(vector)
    norm = weighted_norm(projected, omega_scale)
    if norm <= 0.0:
        raise FloatingPointError("Invalid tangent norm in observables.")

    y1, _, theta1, _, y2, _, theta2, _, z, vz, Theta = central_state
    rho1 = np.exp(y1)
    rho2 = np.exp(y2)
    x = sigmoid(z)
    g = x * (1.0 - x)

    delta_x = g * projected[8]
    delta_xdot = (
        g * projected[9]
        + g * (1.0 - 2.0 * x) * vz * projected[8]
    )
    delta_delta = projected[2] + projected[6] - projected[10]
    delta_rho = math.hypot(
        rho1 * projected[0],
        rho2 * projected[4],
    )
    delta_kappa = kappa0 * delta_x

    base_delta = theta1 + theta2 - Theta
    base_delta = (base_delta + np.pi) % (2.0 * np.pi) - np.pi
    activity = rho1 * rho2 * np.sin(base_delta)

    return {
        "abs_delta_x": abs(delta_x) / norm,
        "abs_delta_xdot": abs(delta_xdot) / norm,
        "abs_delta_Delta": abs(delta_delta) / norm,
        "delta_rho_norm": delta_rho / norm,
        "abs_delta_kappa": abs(delta_kappa) / norm,
        "base_Delta": base_delta,
        "base_activity": activity,
    }


def zero_crossings(values: np.ndarray) -> int:
    signs = np.sign(values)
    nonzero = signs != 0.0
    if np.count_nonzero(nonzero) < 2:
        return 0
    compact = signs[nonzero]
    return int(np.count_nonzero(compact[1:] != compact[:-1]))


def circular_phase_at_extreme(
    phase_fraction: np.ndarray,
    values: np.ndarray,
    maximum: bool,
) -> float:
    index = int(np.argmax(values) if maximum else np.argmin(values))
    return float(phase_fraction[index])


def analyze_case(
    orbit: OrbitData,
    *,
    analysis_periods: int,
    steps_per_period: int,
    initial_x_perturbation: float,
    params: ModelParameters,
    progress_every: int,
) -> CaseResult:
    if analysis_periods < 5:
        raise ValueError("--analysis-periods must be at least 5.")
    if analysis_periods >= orbit.periods:
        raise ValueError(
            f"Q={orbit.q:g}: analysis periods ({analysis_periods}) must be "
            f"smaller than available periods ({orbit.periods})."
        )

    alignment_periods = orbit.periods - analysis_periods
    kappa0 = (55.0 / 6.0) * (10.0 / orbit.q)

    initial_physical = internal_to_physical(orbit.states[:, 0])
    initial_x = float(initial_physical[8])
    omega_scale = float(
        np.sqrt(
            ddV_barrier(initial_x, params.barrier_B)
            / params.m_x
        )
    )

    nodes, midpoints, time_step = prepare_orbit_lattice(
        orbit,
        steps_per_period,
    )
    vector = normalized_initial_x_direction(
        nodes[0],
        initial_x_perturbation,
        omega_scale,
    )

    metric_names = [
        "g_total",
        "g_open",
        "g_phase",
        "g_kappa",
        "share_local",
        "share_radial",
        "share_rotating",
        "abs_delta_x",
        "abs_delta_xdot",
        "abs_delta_Delta",
        "delta_rho_norm",
        "abs_delta_kappa",
        "base_Delta",
        "base_activity",
    ]
    samples = {
        name: np.empty((analysis_periods, steps_per_period), dtype=float)
        for name in metric_names
    }

    period_lambda = np.empty(analysis_periods, dtype=float)
    period_branch_lambda = {
        "open": np.empty(analysis_periods, dtype=float),
        "phase": np.empty(analysis_periods, dtype=float),
        "kappa": np.empty(analysis_periods, dtype=float),
    }

    print()
    print(
        f"Q={orbit.q:g}: orbit periods={orbit.periods}, "
        f"alignment={alignment_periods}, analysis={analysis_periods}"
    )

    analysis_index = 0
    for period in range(orbit.periods):
        first_step = period * steps_per_period
        last_step = first_step + steps_per_period
        start_vector = vector.copy()

        branch_integrals = {
            "open": 0.0,
            "phase": 0.0,
            "kappa": 0.0,
        }

        record = period >= alignment_periods

        for local_step, step in enumerate(range(first_step, last_step)):
            central_start = nodes[step]
            central_mid = midpoints[step]
            central_end = nodes[step + 1]

            k1 = apply_jacobian(
                vector,
                central_start,
                kappa0,
                params,
            )
            midpoint_vector_1 = vector + 0.5 * time_step * k1

            k2 = apply_jacobian(
                midpoint_vector_1,
                central_mid,
                kappa0,
                params,
            )
            midpoint_vector_2 = vector + 0.5 * time_step * k2

            k3 = apply_jacobian(
                midpoint_vector_2,
                central_mid,
                kappa0,
                params,
            )
            end_vector_trial = vector + time_step * k3

            k4 = apply_jacobian(
                end_vector_trial,
                central_end,
                kappa0,
                params,
            )

            # Use the second midpoint estimate for phase-resolved diagnostics.
            diagnostic_vector = midpoint_vector_2
            open_j, phase_j, kappa_j = jacobian_components(
                central_mid,
                kappa0=kappa0,
                params=params,
            )
            d_open = open_j @ diagnostic_vector
            d_phase = phase_j @ diagnostic_vector
            d_kappa = kappa_j @ diagnostic_vector
            d_total = d_open + d_phase + d_kappa

            g_open = growth_contribution(
                diagnostic_vector,
                d_open,
                omega_scale,
            )
            g_phase = growth_contribution(
                diagnostic_vector,
                d_phase,
                omega_scale,
            )
            g_kappa = growth_contribution(
                diagnostic_vector,
                d_kappa,
                omega_scale,
            )
            g_total = growth_contribution(
                diagnostic_vector,
                d_total,
                omega_scale,
            )

            branch_integrals["open"] += g_open * time_step
            branch_integrals["phase"] += g_phase * time_step
            branch_integrals["kappa"] += g_kappa * time_step

            if record:
                local_share, radial_share, rotating_share = (
                    state_group_shares(
                        diagnostic_vector,
                        omega_scale,
                    )
                )
                observables = physical_tangent_observables(
                    diagnostic_vector,
                    central_mid,
                    kappa0,
                    omega_scale,
                )

                # Multiplication by T_ref makes the cycle average equal to
                # lambda per reference period.
                samples["g_total"][analysis_index, local_step] = (
                    orbit.period_reference * g_total
                )
                samples["g_open"][analysis_index, local_step] = (
                    orbit.period_reference * g_open
                )
                samples["g_phase"][analysis_index, local_step] = (
                    orbit.period_reference * g_phase
                )
                samples["g_kappa"][analysis_index, local_step] = (
                    orbit.period_reference * g_kappa
                )
                samples["share_local"][analysis_index, local_step] = local_share
                samples["share_radial"][analysis_index, local_step] = radial_share
                samples["share_rotating"][analysis_index, local_step] = (
                    rotating_share
                )
                for name, value in observables.items():
                    samples[name][analysis_index, local_step] = value

            vector = vector + (
                time_step / 6.0
            ) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        vector = gauge_project(vector)
        end_norm = weighted_norm(vector, omega_scale)
        if not np.isfinite(end_norm) or end_norm <= 0.0:
            raise FloatingPointError(
                f"Q={orbit.q:g}: invalid period-end tangent norm."
            )
        period_log = math.log(end_norm)

        vector /= end_norm

        # Keep one continuous orientation at period boundaries.
        if np.dot(
            weighted_vector(vector, omega_scale),
            weighted_vector(start_vector, omega_scale),
        ) < 0.0:
            vector *= -1.0

        if record:
            period_lambda[analysis_index] = period_log
            for name in period_branch_lambda:
                period_branch_lambda[name][analysis_index] = (
                    branch_integrals[name]
                )
            analysis_index += 1

        if (
            period == 0
            or period + 1 == orbit.periods
            or (period + 1) % progress_every == 0
        ):
            print(
                f"  Q={orbit.q:g}: [{period + 1:>4}/{orbit.periods}] "
                f"period log={period_log:+.6e}"
            )

    if analysis_index != analysis_periods:
        raise RuntimeError("Internal analysis-period accounting failed.")

    phase_fraction = (
        np.arange(steps_per_period, dtype=float) + 0.5
    ) / steps_per_period
    phase_angle = 2.0 * np.pi * phase_fraction

    profile_mean = {
        name: np.mean(values, axis=0)
        for name, values in samples.items()
    }
    profile_std = {
        name: np.std(values, axis=0, ddof=1)
        for name, values in samples.items()
    }

    summary: dict[str, float] = {}
    summary["lambda_mean"] = float(np.mean(period_lambda))
    summary["lambda_standard_error"] = float(
        np.std(period_lambda, ddof=1) / np.sqrt(period_lambda.size)
    )

    for branch in ["open", "phase", "kappa"]:
        values = period_branch_lambda[branch]
        summary[f"lambda_{branch}_mean"] = float(np.mean(values))
        summary[f"lambda_{branch}_standard_error"] = float(
            np.std(values, ddof=1) / np.sqrt(values.size)
        )

        profile = profile_mean[f"g_{branch}"]
        summary[f"{branch}_positive_area"] = float(
            np.mean(np.maximum(profile, 0.0))
        )
        summary[f"{branch}_negative_area"] = float(
            np.mean(np.minimum(profile, 0.0))
        )
        summary[f"{branch}_zero_crossings"] = float(
            zero_crossings(profile)
        )
        summary[f"{branch}_max_feed_phase"] = circular_phase_at_extreme(
            phase_fraction,
            profile,
            maximum=True,
        )
        summary[f"{branch}_max_brake_phase"] = circular_phase_at_extreme(
            phase_fraction,
            profile,
            maximum=False,
        )

    branch_sum = (
        summary["lambda_open_mean"]
        + summary["lambda_phase_mean"]
        + summary["lambda_kappa_mean"]
    )
    summary["branch_closure_residual"] = (
        summary["lambda_mean"] - branch_sum
    )

    for group in ["local", "radial", "rotating"]:
        profile = profile_mean[f"share_{group}"]
        summary[f"{group}_mean_share"] = float(np.mean(profile))
        summary[f"{group}_max_share"] = float(np.max(profile))
        summary[f"{group}_max_phase"] = circular_phase_at_extreme(
            phase_fraction,
            profile,
            maximum=True,
        )

    summary["total_positive_phase_fraction"] = float(
        np.mean(profile_mean["g_total"] > 0.0)
    )
    summary["total_zero_crossings"] = float(
        zero_crossings(profile_mean["g_total"])
    )

    return CaseResult(
        q=orbit.q,
        period_reference=orbit.period_reference,
        orbit_periods=orbit.periods,
        alignment_periods=alignment_periods,
        analysis_periods=analysis_periods,
        phase_fraction=phase_fraction,
        phase_angle=phase_angle,
        profile_mean=profile_mean,
        profile_std=profile_std,
        period_lambda=period_lambda,
        period_branch_lambda=period_branch_lambda,
        summary=summary,
        source=orbit.source,
    )


def q_token(q: float) -> str:
    text = f"{q:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def save_case_dashboard(
    output_path: Path,
    result: CaseResult,
) -> None:
    phase = result.phase_fraction
    mean = result.profile_mean

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
        constrained_layout=True,
    )
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(phase, mean["g_total"], linewidth=2.0, label="FULL")
    ax1.plot(phase, mean["g_open"], linewidth=1.5, label="OPEN contribution")
    ax1.plot(phase, mean["g_phase"], linewidth=1.5, label="PHASE return")
    ax1.plot(phase, mean["g_kappa"], linewidth=1.5, label="KAPPA return")
    ax1.axhline(0.0, linewidth=1.0)
    ax1.set_title("Instantaneous logarithmic growth")
    ax1.set_xlabel("reference-cycle phase")
    ax1.set_ylabel("T_ref · d(log ||δu||)/dt")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for name, label in [
        ("g_total", "FULL"),
        ("g_open", "OPEN"),
        ("g_phase", "PHASE"),
        ("g_kappa", "KAPPA"),
    ]:
        cumulative = np.cumsum(mean[name]) / phase.size
        ax2.plot(phase, cumulative, linewidth=1.7, label=label)
    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_title("Cumulative log-growth through the cycle")
    ax2.set_xlabel("reference-cycle phase")
    ax2.set_ylabel("accumulated log factor")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(phase, mean["share_local"], linewidth=1.8, label="LOCAL x")
    ax3.plot(phase, mean["share_radial"], linewidth=1.8, label="RADIAL support")
    ax3.plot(
        phase,
        mean["share_rotating"],
        linewidth=1.8,
        label="ROTATING phase",
    )
    ax3.set_title("Where the tangent norm is carried")
    ax3.set_xlabel("reference-cycle phase")
    ax3.set_ylabel("weighted norm share")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    observable_names = [
        ("abs_delta_x", "|δx|"),
        ("abs_delta_Delta", "|δΔ|"),
        ("delta_rho_norm", "||δρ||"),
        ("abs_delta_kappa", "|δκ_eff|"),
    ]
    for name, label in observable_names:
        values = mean[name]
        scale = float(np.max(np.abs(values)))
        normalized = values / scale if scale > 0.0 else values
        ax4.plot(phase, normalized, linewidth=1.7, label=label)

    ax4.set_title("Physical tangent observables (each normalized to its maximum)")
    ax4.set_xlabel("reference-cycle phase")
    ax4.set_ylabel("normalized magnitude")
    ax4.set_ylim(0.0, 1.05)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    figure.suptitle(
        f"Phase-resolved leading tangent mode at Q={result.q:g}\n"
        f"λ={result.summary['lambda_mean']:+.6e} per reference period",
        fontsize=17,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_comparison_plot(
    output_path: Path,
    results: list[CaseResult],
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        constrained_layout=True,
    )
    ax1, ax2 = axes

    for result in results:
        ax1.plot(
            result.phase_fraction,
            result.profile_mean["g_total"],
            linewidth=2.0,
            label=f"FULL Q={result.q:g}",
        )
        ax1.plot(
            result.phase_fraction,
            result.profile_mean["g_phase"],
            linestyle="--",
            linewidth=1.5,
            label=f"PHASE Q={result.q:g}",
        )
        ax1.plot(
            result.phase_fraction,
            result.profile_mean["g_kappa"],
            linestyle=":",
            linewidth=1.7,
            label=f"KAPPA Q={result.q:g}",
        )

    ax1.axhline(0.0, linewidth=1.0)
    ax1.set_title("Growth choreography across the reference cycle")
    ax1.set_xlabel("reference-cycle phase")
    ax1.set_ylabel("T_ref · instantaneous growth")
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=2)

    for result in results:
        ax2.plot(
            result.phase_fraction,
            result.profile_mean["share_local"],
            linewidth=1.8,
            label=f"LOCAL Q={result.q:g}",
        )
        ax2.plot(
            result.phase_fraction,
            result.profile_mean["share_rotating"],
            linestyle="--",
            linewidth=1.8,
            label=f"ROTATING Q={result.q:g}",
        )
        ax2.plot(
            result.phase_fraction,
            result.profile_mean["share_radial"],
            linestyle=":",
            linewidth=1.8,
            label=f"RADIAL Q={result.q:g}",
        )

    ax2.set_title("Handoff of the tangent norm")
    ax2.set_xlabel("reference-cycle phase")
    ax2.set_ylabel("weighted norm share")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(ncol=2)

    figure.suptitle(
        "Near-curtain versus beyond-curtain tangent motion",
        fontsize=17,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_profiles_csv(
    output_path: Path,
    results: list[CaseResult],
) -> None:
    profile_names = list(results[0].profile_mean.keys())

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["Q", "phase_fraction", "phase_angle_rad"]
        for name in profile_names:
            header.extend([f"{name}_mean", f"{name}_std"])
        writer.writerow(header)

        for result in results:
            for index, phase in enumerate(result.phase_fraction):
                row: list[float] = [
                    result.q,
                    phase,
                    result.phase_angle[index],
                ]
                for name in profile_names:
                    row.extend(
                        [
                            result.profile_mean[name][index],
                            result.profile_std[name][index],
                        ]
                    )
                writer.writerow(row)


def save_periods_csv(
    output_path: Path,
    results: list[CaseResult],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "analysis_period_index",
                "lambda_full",
                "lambda_open_contribution",
                "lambda_phase_contribution",
                "lambda_kappa_contribution",
                "closure_residual",
            ]
        )
        for result in results:
            for index in range(result.analysis_periods):
                open_value = result.period_branch_lambda["open"][index]
                phase_value = result.period_branch_lambda["phase"][index]
                kappa_value = result.period_branch_lambda["kappa"][index]
                full_value = result.period_lambda[index]
                writer.writerow(
                    [
                        result.q,
                        index + 1,
                        full_value,
                        open_value,
                        phase_value,
                        kappa_value,
                        full_value - open_value - phase_value - kappa_value,
                    ]
                )


def save_summary_csv(
    output_path: Path,
    results: list[CaseResult],
) -> None:
    keys = sorted(results[0].summary.keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "source",
                "orbit_periods",
                "alignment_periods",
                "analysis_periods",
                *keys,
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.q,
                    str(result.source.resolve()),
                    result.orbit_periods,
                    result.alignment_periods,
                    result.analysis_periods,
                    *[result.summary[key] for key in keys],
                ]
            )


def save_raw_npz(
    output_path: Path,
    results: list[CaseResult],
) -> None:
    payload: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
    }
    for result in results:
        prefix = f"q_{q_token(result.q)}"
        payload[f"{prefix}_q"] = result.q
        payload[f"{prefix}_period_reference"] = result.period_reference
        payload[f"{prefix}_phase_fraction"] = result.phase_fraction
        payload[f"{prefix}_period_lambda"] = result.period_lambda
        for name, values in result.profile_mean.items():
            payload[f"{prefix}_{name}_mean"] = values
            payload[f"{prefix}_{name}_std"] = result.profile_std[name]
        for name, values in result.period_branch_lambda.items():
            payload[f"{prefix}_period_lambda_{name}"] = values
        for name, value in result.summary.items():
            payload[f"{prefix}_summary_{name}"] = value

    np.savez_compressed(output_path, **payload)


def build_report(
    results: list[CaseResult],
    output_paths: dict[str, Path],
) -> str:
    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "PURPOSE",
        "-------",
        "Resolve the leading FULL tangent mode through one reference cycle,",
        "using an ensemble average over many late aligned cycles.",
        "",
        "EXACT INSTANTANEOUS DECOMPOSITION",
        "---------------------------------",
        "g_FULL = g_OPEN + g_PHASE + g_KAPPA",
        "",
    ]

    for result in results:
        s = result.summary
        lines.extend(
            [
                f"Q={result.q:g}",
                f"  source orbit                    = {result.source.resolve()}",
                f"  orbit periods                   = {result.orbit_periods}",
                f"  alignment periods               = {result.alignment_periods}",
                f"  phase-averaged periods          = {result.analysis_periods}",
                f"  lambda FULL                     = {s['lambda_mean']:+.12e}",
                f"  lambda FULL standard error      = {s['lambda_standard_error']:.12e}",
                f"  lambda OPEN contribution        = {s['lambda_open_mean']:+.12e}",
                f"  lambda PHASE contribution       = {s['lambda_phase_mean']:+.12e}",
                f"  lambda KAPPA contribution       = {s['lambda_kappa_mean']:+.12e}",
                f"  branch closure residual         = {s['branch_closure_residual']:+.12e}",
                f"  FULL positive phase fraction    = {s['total_positive_phase_fraction']:.9f}",
                f"  FULL zero crossings             = {int(round(s['total_zero_crossings']))}",
                "",
                "  PHASE alternation",
                f"    positive area                 = {s['phase_positive_area']:+.12e}",
                f"    negative area                 = {s['phase_negative_area']:+.12e}",
                f"    zero crossings                = {int(round(s['phase_zero_crossings']))}",
                f"    strongest feed phase          = {s['phase_max_feed_phase']:.9f}",
                f"    strongest brake phase         = {s['phase_max_brake_phase']:.9f}",
                "",
                "  KAPPA alternation",
                f"    positive area                 = {s['kappa_positive_area']:+.12e}",
                f"    negative area                 = {s['kappa_negative_area']:+.12e}",
                f"    zero crossings                = {int(round(s['kappa_zero_crossings']))}",
                f"    strongest feed phase          = {s['kappa_max_feed_phase']:.9f}",
                f"    strongest brake phase         = {s['kappa_max_brake_phase']:.9f}",
                "",
                "  Mean tangent-norm shares",
                f"    LOCAL                         = {s['local_mean_share']:.9f}",
                f"    RADIAL                        = {s['radial_mean_share']:.9f}",
                f"    ROTATING                      = {s['rotating_mean_share']:.9f}",
                "",
                "  Phases of maximum occupancy",
                f"    LOCAL                         = {s['local_max_phase']:.9f}",
                f"    RADIAL                        = {s['radial_max_phase']:.9f}",
                f"    ROTATING                      = {s['rotating_max_phase']:.9f}",
                "",
            ]
        )

    lines.extend(
        [
            "READING GUIDE",
            "-------------",
            "The branch curves may contain large positive and negative arcs even",
            "when their cycle integral is small. That alternation is the local",
            "choreography of feeding and braking. The norm-share curves show",
            "where the perturbation is carried while those exchanges occur.",
            "",
            "The near-curtain case is not interpreted from one period: every",
            "profile is the phase-conditioned average of the requested number",
            "of late aligned periods.",
            "",
            f"comparison PNG = {output_paths['comparison'].resolve()}",
            f"profiles CSV   = {output_paths['profiles'].resolve()}",
            f"periods CSV    = {output_paths['periods'].resolve()}",
            f"summary CSV    = {output_paths['summary'].resolve()}",
            f"raw NPZ        = {output_paths['raw'].resolve()}",
        ]
    )

    for name, path in output_paths.items():
        if name.startswith("case_"):
            lines.append(f"{name} PNG = {path.resolve()}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch the leading tangent mode move through one reference cycle "
            "and decompose its instantaneous growth into OPEN, PHASE, KAPPA."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help=(
            "repeatable case in the form Q=orbit_cache.npz; "
            "defaults to Q=522.25 and Q=550 cache paths"
        ),
    )
    parser.add_argument("--analysis-periods", type=int, default=80)
    parser.add_argument("--steps-per-period", type=int, default=160)
    parser.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output-prefix",
        default="support_tangent_cycle_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cases = args.case
    if not cases:
        cases = [
            (
                522.25,
                Path(
                    "support_feedback_q_curtain_focus_cache/"
                    "q_522p25_orbit.npz"
                ),
            ),
            (
                550.0,
                Path(
                    "support_feedback_q_curtain_cache/"
                    "q_550_orbit.npz"
                ),
            ),
        ]

    if len(cases) < 1:
        raise ValueError("At least one --case is required.")
    if args.analysis_periods < 5:
        raise ValueError("--analysis-periods must be at least 5.")
    if args.steps_per_period < 40:
        raise ValueError("--steps-per-period must be at least 40.")
    if args.initial_x_perturbation <= 0.0:
        raise ValueError("--initial-x-perturbation must be positive.")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive.")

    params = ModelParameters(gamma_rho=args.gamma_rho)

    print(f"Script version       = {SCRIPT_VERSION}")
    print(f"analysis periods     = {args.analysis_periods}")
    print(f"steps per period     = {args.steps_per_period}")
    print("cases:")
    for q, path in cases:
        print(f"  Q={q:g}: {path}")

    results: list[CaseResult] = []
    for q, path in sorted(cases, key=lambda item: item[0]):
        orbit = load_orbit(q, path)
        result = analyze_case(
            orbit,
            analysis_periods=args.analysis_periods,
            steps_per_period=args.steps_per_period,
            initial_x_perturbation=args.initial_x_perturbation,
            params=params,
            progress_every=args.progress_every,
        )
        results.append(result)

    prefix = Path(args.output_prefix)
    output_paths: dict[str, Path] = {
        "comparison": prefix.with_name(prefix.name + "_comparison.png"),
        "profiles": prefix.with_name(prefix.name + "_phase_profiles.csv"),
        "periods": prefix.with_name(prefix.name + "_periods.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.csv"),
        "raw": prefix.with_name(prefix.name + "_raw.npz"),
        "report": prefix.with_name(prefix.name + "_report.txt"),
    }

    for result in results:
        key = f"case_Q{q_token(result.q)}"
        output_paths[key] = prefix.with_name(
            prefix.name + f"_Q{q_token(result.q)}.png"
        )
        save_case_dashboard(output_paths[key], result)

    save_comparison_plot(output_paths["comparison"], results)
    save_profiles_csv(output_paths["profiles"], results)
    save_periods_csv(output_paths["periods"], results)
    save_summary_csv(output_paths["summary"], results)
    save_raw_npz(output_paths["raw"], results)

    report = build_report(results, output_paths)
    output_paths["report"].write_text(report, encoding="utf-8")

    print()
    print(report)


if __name__ == "__main__":
    main()
