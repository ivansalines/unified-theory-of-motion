#!/usr/bin/env python3
"""Möbius / antiperiodic monodromy audit for the recurrent tangent line.

This script does not open a new parameter scan and does not generate new
central orbits. It reuses the existing Q=522.25 and Q=550 orbit caches.

The numerical question is deliberately split into three layers:

1. Tangent-line recurrence

       w(s + P_line) approximately equals -w(s)
       w(s + 2 P_line) approximately equals  w(s)

   where s = Theta/(2*pi) is the unwrapped support-rotation coordinate.

2. Closure of the base tangent equation

   A negative return of the tangent line is a Möbius/Floquet statement only
   when the reduced tangent coefficients close over the same interval.  The
   script therefore compares the reduced s-time Jacobian B(s) at s, s+P_line,
   and s+2P_line.

3. Transport / monodromy

   The reduced fundamental matrix is integrated over P_line and 2 P_line.
   The observed leading tangent direction is propagated and its signed gain,
   line overlap, and matched transport eigenvalue are measured.

Interpretation rule
-------------------

- coefficient loop closed at P_line, line overlap ~1, signed return ~-1:
      Z2=-1 Möbius-line-bundle candidate supported;

- coefficient loop not closed at P_line but closed at 2 P_line:
      antiperiodic half-cycle / negative transport supported, but Möbius over
      the P_line base cycle is not established;

- neither coefficient loop closes:
      recurrent tangent line over a non-closed base segment; topological claim
      remains unsupported.

The reduced tangent coordinates are

    [dy1, dv1, domega1, dy2, dv2, domega2, dz, dvz, dDelta]

with

    dDelta = dtheta1 + dtheta2 - dTheta.

Four different full-space embeddings of dDelta are compared.  The reduced
Jacobian must be invariant to this gauge choice before any interpretation is
allowed.
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
from scipy.signal import find_peaks
from scipy.special import expit


ALPHA, BETA, GAMMA = 1.0, 0.8, 0.2
SCRIPT_VERSION = "2026-08-04-mobius-monodromy-audit-v1"
REDUCED_DIMENSION = 9


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
class UniformTail:
    s: np.ndarray
    states: np.ndarray
    reduced_direction: np.ndarray
    metric: np.ndarray
    omega_scale: float


@dataclass
class AnchorRecord:
    q: float
    anchor_index: int
    anchor_s: float
    anchor_phase: float
    line_period: float
    coefficient_closure_p: float
    coefficient_closure_2p: float
    base_closure_p: float
    base_closure_2p: float
    observed_line_overlap_p: float
    observed_signed_overlap_p: float
    observed_line_overlap_2p: float
    observed_signed_overlap_2p: float
    euclidean_signed_overlap_p: float
    euclidean_signed_overlap_2p: float
    transport_observed_line_overlap_p: float
    transport_observed_signed_overlap_p: float
    transport_observed_line_overlap_2p: float
    transport_observed_signed_overlap_2p: float
    signed_gain_p: float
    signed_gain_2p: float
    log_abs_gain_p: float
    log_abs_gain_2p: float
    matched_eigenvalue_p_real: float
    matched_eigenvalue_p_imag: float
    matched_eigenvalue_p_abs: float
    matched_eigenvector_alignment_p: float
    matched_eigenvalue_2p_real: float
    matched_eigenvalue_2p_imag: float
    matched_eigenvalue_2p_abs: float
    matched_eigenvector_alignment_2p: float
    transport_matrix_condition_p: float
    transport_matrix_condition_2p: float


@dataclass
class CaseResult:
    q: float
    source: Path
    line_period: float
    oriented_period: float
    recurrence_lag: np.ndarray
    recurrence_signed: np.ndarray
    recurrence_line: np.ndarray
    anchors: list[AnchorRecord]
    median_spectrum_p: np.ndarray
    median_spectrum_2p: np.ndarray
    metric: np.ndarray
    gauge_max_absolute_error: float
    gauge_max_relative_error: float
    resolution_relative_error_p: float
    resolution_relative_error_2p: float
    summary: dict[str, float | str | bool]


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
        raise argparse.ArgumentTypeError("--case must use Q=orbit_cache.npz")
    q_text, path_text = raw.split("=", 1)
    try:
        q = float(q_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid Q value: {q_text!r}"
        ) from exc

    path_text = path_text.strip()
    if not path_text:
        raise argparse.ArgumentTypeError("The orbit path is empty.")
    return q, Path(path_text)


def load_orbit(case_q: float, path: Path) -> OrbitData:
    if not path.exists():
        raise FileNotFoundError(f"Orbit cache not found: {path}")

    archive = np.load(path)
    required = {"period_reference", "times", "states"}
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"{path} is missing: {', '.join(missing)}")

    period_reference = float(
        np.asarray(archive["period_reference"]).item()
    )
    times = np.asarray(archive["times"], dtype=float)
    states = np.asarray(archive["states"], dtype=float)

    if times.ndim != 1 or times.size < 2:
        raise ValueError(f"{path}: invalid time array.")
    if states.shape != (11, times.size):
        raise ValueError(
            f"{path}: states must have shape (11, number_of_times)."
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(states)):
        raise FloatingPointError(f"{path}: non-finite orbit data.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path}: times are not strictly increasing.")

    if "q" in archive.files:
        stored_q = float(np.asarray(archive["q"]).item())
        if not np.isclose(stored_q, case_q, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"{path}: requested Q={case_q:g}, stored Q={stored_q:g}."
            )

    times = times - times[0]
    duration = float(times[-1])
    period_float = duration / period_reference
    periods = int(round(period_float))
    if not np.isclose(period_float, periods, rtol=2e-10, atol=2e-8):
        raise ValueError(
            f"{path}: duration is not an integer number of T_ref."
        )

    return OrbitData(
        q=case_q,
        period_reference=period_reference,
        times=times,
        states=states,
        periods=periods,
        source=path,
    )


def full_jacobian(
    state: np.ndarray,
    *,
    kappa0: float,
    params: ModelParameters,
) -> np.ndarray:
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

    jacobian = np.zeros((11, 11), dtype=float)

    jacobian[0, 1] = 1.0

    log_u1 = (
        ddU(rho1, params.e_reserve)
        - dU(rho1, params.e_reserve) / rho1
    )
    radial_1 = kappa_eff * ratio21 * cos_delta
    angular_1 = kappa_eff * ratio21 * sin_delta

    jacobian[1, 0] = -log_u1 + radial_1
    jacobian[1, 1] = -params.gamma_rho - 2.0 * v1
    jacobian[1, 2] = angular_1
    jacobian[1, 3] = 2.0 * omega1
    jacobian[1, 4] = -radial_1
    jacobian[1, 6] = angular_1
    jacobian[1, 8] = -kappa0 * g * ratio21 * cos_delta
    jacobian[1, 10] = -angular_1

    jacobian[2, 3] = 1.0

    jacobian[3, 0] = -angular_1
    jacobian[3, 1] = -2.0 * omega1
    jacobian[3, 2] = radial_1
    jacobian[3, 3] = -2.0 * v1
    jacobian[3, 4] = angular_1
    jacobian[3, 6] = radial_1
    jacobian[3, 8] = kappa0 * g * ratio21 * sin_delta
    jacobian[3, 10] = -radial_1

    jacobian[4, 5] = 1.0

    log_u2 = (
        ddU(rho2, params.e_reserve)
        - dU(rho2, params.e_reserve) / rho2
    )
    radial_2 = kappa_eff * ratio12 * cos_delta
    angular_2 = kappa_eff * ratio12 * sin_delta

    jacobian[5, 0] = -radial_2
    jacobian[5, 2] = angular_2
    jacobian[5, 4] = -log_u2 + radial_2
    jacobian[5, 5] = -params.gamma_rho - 2.0 * v2
    jacobian[5, 6] = angular_2
    jacobian[5, 7] = 2.0 * omega2
    jacobian[5, 8] = -kappa0 * g * ratio12 * cos_delta
    jacobian[5, 10] = -angular_2

    jacobian[6, 7] = 1.0

    jacobian[7, 0] = angular_2
    jacobian[7, 2] = radial_2
    jacobian[7, 4] = -angular_2
    jacobian[7, 5] = -2.0 * omega2
    jacobian[7, 6] = radial_2
    jacobian[7, 7] = -2.0 * v2
    jacobian[7, 8] = kappa0 * g * ratio12 * sin_delta
    jacobian[7, 10] = -radial_2

    jacobian[8, 9] = 1.0

    activity = rho1 * rho2 * sin_delta
    activity_cos = rho1 * rho2 * cos_delta
    activity_factor = params.mu_x * activity / (params.m_x * g)
    phase_factor = params.mu_x * activity_cos / (params.m_x * g)

    numerator = (
        -dV_barrier(x, params.barrier_B)
        + params.mu_x * activity
        - params.gamma_x * g * vz
    ) / params.m_x

    jacobian[9, 0] = activity_factor
    jacobian[9, 2] = phase_factor
    jacobian[9, 4] = activity_factor
    jacobian[9, 6] = phase_factor
    jacobian[9, 8] = (
        (
            -ddV_barrier(x, params.barrier_B)
            - params.gamma_x * (1.0 - 2.0 * x) * vz
        ) / params.m_x
        - numerator * (1.0 - 2.0 * x) / g
        + 2.0 * g * vz**2
    )
    jacobian[9, 9] = (
        -params.gamma_x / params.m_x
        - 2.0 * (1.0 - 2.0 * x) * vz
    )
    jacobian[9, 10] = -phase_factor

    jacobian[10, 8] = g

    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("Non-finite full Jacobian.")
    return jacobian


def reduced_extraction() -> np.ndarray:
    extraction = np.zeros((REDUCED_DIMENSION, 11), dtype=float)
    extraction[0, 0] = 1.0
    extraction[1, 1] = 1.0
    extraction[2, 3] = 1.0
    extraction[3, 4] = 1.0
    extraction[4, 5] = 1.0
    extraction[5, 7] = 1.0
    extraction[6, 8] = 1.0
    extraction[7, 9] = 1.0
    extraction[8, 2] = 1.0
    extraction[8, 6] = 1.0
    extraction[8, 10] = -1.0
    return extraction


def reduced_embedding(name: str) -> np.ndarray:
    embedding = np.zeros((11, REDUCED_DIMENSION), dtype=float)
    embedding[0, 0] = 1.0
    embedding[1, 1] = 1.0
    embedding[3, 2] = 1.0
    embedding[4, 3] = 1.0
    embedding[5, 4] = 1.0
    embedding[7, 5] = 1.0
    embedding[8, 6] = 1.0
    embedding[9, 7] = 1.0

    if name == "symmetric":
        embedding[2, 8] = 1.0 / 3.0
        embedding[6, 8] = 1.0 / 3.0
        embedding[10, 8] = -1.0 / 3.0
    elif name == "theta1":
        embedding[2, 8] = 1.0
    elif name == "theta2":
        embedding[6, 8] = 1.0
    elif name == "support":
        embedding[10, 8] = -1.0
    else:
        raise ValueError(f"Unknown reduced embedding: {name}")

    extraction = reduced_extraction()
    identity_error = np.max(
        np.abs(extraction @ embedding - np.eye(REDUCED_DIMENSION))
    )
    if identity_error > 1e-14:
        raise RuntimeError(
            f"Invalid reduced embedding {name}: R E != I."
        )
    return embedding


def full_weight_matrix(omega_scale: float) -> np.ndarray:
    weights = np.ones(11, dtype=float)
    weights[[1, 3, 5, 7, 9]] = 1.0 / omega_scale
    return np.diag(weights**2)


def reduced_metric(omega_scale: float) -> np.ndarray:
    embedding = reduced_embedding("symmetric")
    return embedding.T @ full_weight_matrix(omega_scale) @ embedding


def metric_dot(a: np.ndarray, b: np.ndarray, metric: np.ndarray) -> float:
    return float(np.real(np.conjugate(a) @ metric @ b))


def metric_norm(a: np.ndarray, metric: np.ndarray) -> float:
    value = metric_dot(a, a, metric)
    if value <= 0.0 or not np.isfinite(value):
        raise FloatingPointError("Invalid reduced metric norm.")
    return math.sqrt(value)


def normalize_metric(a: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return np.asarray(a) / metric_norm(a, metric)


def reduced_jacobian_time(
    state: np.ndarray,
    *,
    kappa0: float,
    params: ModelParameters,
    embedding_name: str = "symmetric",
) -> np.ndarray:
    extraction = reduced_extraction()
    embedding = reduced_embedding(embedding_name)
    return extraction @ full_jacobian(
        state,
        kappa0=kappa0,
        params=params,
    ) @ embedding


def reduced_jacobian_rotation(
    state: np.ndarray,
    *,
    kappa0: float,
    params: ModelParameters,
    embedding_name: str = "symmetric",
) -> np.ndarray:
    x = float(sigmoid(state[8]))
    if x <= 0.0:
        raise FloatingPointError("Theta_dot=x must remain positive.")
    return (2.0 * np.pi / x) * reduced_jacobian_time(
        state,
        kappa0=kappa0,
        params=params,
        embedding_name=embedding_name,
    )


def gauge_project_full(vector: np.ndarray) -> np.ndarray:
    projected = np.asarray(vector, dtype=float).copy()
    delta_phase = projected[2] + projected[6] - projected[10]
    projected[2] = delta_phase / 3.0
    projected[6] = delta_phase / 3.0
    projected[10] = -delta_phase / 3.0
    return projected


def full_weighted_norm(vector: np.ndarray, omega_scale: float) -> float:
    projected = gauge_project_full(vector)
    weighted = projected.copy()
    weighted[[1, 3, 5, 7, 9]] /= omega_scale
    return float(np.linalg.norm(weighted))


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
    deviation = gauge_project_full(deviation)
    norm = full_weighted_norm(deviation, omega_scale)
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("Invalid initial tangent direction.")
    return deviation / norm


def prepare_orbit_lattice(
    orbit: OrbitData,
    steps_per_period: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    total_steps = orbit.periods * steps_per_period
    start = float(orbit.times[0])
    end = float(orbit.times[-1])
    duration = end - start
    dt = duration / total_steps

    spline = CubicSpline(
        orbit.times,
        orbit.states,
        axis=1,
        bc_type="natural",
        extrapolate=False,
    )

    node_times = start + np.arange(total_steps + 1, dtype=float) * dt
    node_times[0] = start
    node_times[-1] = end
    midpoint_times = 0.5 * (node_times[:-1] + node_times[1:])

    node_times = np.clip(node_times, start, end)
    midpoint_times = np.clip(midpoint_times, start, end)

    nodes = np.asarray(spline(node_times), dtype=float).T
    midpoints = np.asarray(spline(midpoint_times), dtype=float).T

    if nodes.shape != (total_steps + 1, 11):
        raise RuntimeError("Unexpected orbit node shape.")
    if midpoints.shape != (total_steps, 11):
        raise RuntimeError("Unexpected orbit midpoint shape.")
    if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(midpoints)):
        raise FloatingPointError("Orbit interpolation produced non-finite data.")

    return nodes, midpoints, dt


def collect_uniform_tail(
    orbit: OrbitData,
    *,
    tail_periods: int,
    steps_per_period: int,
    samples_per_rotation: int,
    initial_x_perturbation: float,
    params: ModelParameters,
    progress_every: int,
) -> UniformTail:
    if tail_periods < 20:
        raise ValueError("--tail-periods must be at least 20.")

    nodes, midpoints, dt = prepare_orbit_lattice(
        orbit,
        steps_per_period,
    )
    kappa0 = (55.0 / 6.0) * (10.0 / orbit.q)

    initial_x = float(internal_to_physical(nodes[0])[8])
    omega_scale = float(
        np.sqrt(
            ddV_barrier(initial_x, params.barrier_B)
            / params.m_x
        )
    )
    metric = reduced_metric(omega_scale)
    extraction = reduced_extraction()

    vector = normalized_initial_x_direction(
        nodes[0],
        initial_x_perturbation,
        omega_scale,
    )

    store_start_period = max(0, orbit.periods - tail_periods)
    s_values: list[float] = []
    state_values: list[np.ndarray] = []
    reduced_directions: list[np.ndarray] = []
    previous_reduced: np.ndarray | None = None

    print()
    print(
        f"Q={orbit.q:g}: orbit={orbit.periods} periods, "
        f"tail={tail_periods} periods"
    )

    for period in range(orbit.periods):
        first_step = period * steps_per_period
        last_step = first_step + steps_per_period

        for step in range(first_step, last_step):
            central_start = nodes[step]
            central_mid = midpoints[step]
            central_end = nodes[step + 1]

            j1 = full_jacobian(
                central_start,
                kappa0=kappa0,
                params=params,
            )
            k1 = j1 @ vector

            midpoint_1 = vector + 0.5 * dt * k1
            j2 = full_jacobian(
                central_mid,
                kappa0=kappa0,
                params=params,
            )
            k2 = j2 @ midpoint_1

            diagnostic_vector = vector + 0.5 * dt * k2
            k3 = j2 @ diagnostic_vector

            end_trial = vector + dt * k3
            j4 = full_jacobian(
                central_end,
                kappa0=kappa0,
                params=params,
            )
            k4 = j4 @ end_trial

            if period >= store_start_period:
                projected = gauge_project_full(diagnostic_vector)
                norm = full_weighted_norm(projected, omega_scale)
                if norm <= 0.0 or not np.isfinite(norm):
                    raise FloatingPointError("Invalid diagnostic tangent norm.")
                projected /= norm

                reduced = extraction @ projected
                reduced = normalize_metric(reduced, metric)

                if (
                    previous_reduced is not None
                    and metric_dot(reduced, previous_reduced, metric) < 0.0
                ):
                    reduced *= -1.0

                previous_reduced = reduced.copy()
                s_values.append(float(central_mid[10] / (2.0 * np.pi)))
                state_values.append(central_mid.copy())
                reduced_directions.append(reduced)

            vector = vector + (
                dt / 6.0
            ) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        vector = gauge_project_full(vector)
        end_norm = full_weighted_norm(vector, omega_scale)
        if end_norm <= 0.0 or not np.isfinite(end_norm):
            raise FloatingPointError("Invalid period-end tangent norm.")
        vector /= end_norm

        if (
            period == 0
            or period + 1 == orbit.periods
            or (period + 1) % progress_every == 0
        ):
            print(f"  Q={orbit.q:g}: [{period + 1:>4}/{orbit.periods}]")

    s_raw = np.asarray(s_values, dtype=float)
    states_raw = np.asarray(state_values, dtype=float)
    directions_raw = np.asarray(reduced_directions, dtype=float)

    if s_raw.size < 100:
        raise RuntimeError("Insufficient tail samples.")
    if states_raw.shape != (s_raw.size, 11):
        raise RuntimeError("Unexpected stored state shape.")
    if directions_raw.shape != (s_raw.size, REDUCED_DIMENSION):
        raise RuntimeError("Unexpected stored reduced-direction shape.")
    if np.any(np.diff(s_raw) <= 0.0):
        raise ValueError("Support-rotation coordinate is not increasing.")

    start = float(s_raw[0])
    end = float(s_raw[-1])
    count = int(math.floor((end - start) * samples_per_rotation)) + 1
    s_uniform = start + np.arange(count, dtype=float) / samples_per_rotation

    states_uniform = np.empty((count, 11), dtype=float)
    for component in range(11):
        states_uniform[:, component] = np.interp(
            s_uniform,
            s_raw,
            states_raw[:, component],
        )

    directions_uniform = np.empty((count, REDUCED_DIMENSION), dtype=float)
    for component in range(REDUCED_DIMENSION):
        directions_uniform[:, component] = np.interp(
            s_uniform,
            s_raw,
            directions_raw[:, component],
        )

    for index in range(count):
        directions_uniform[index] = normalize_metric(
            directions_uniform[index],
            metric,
        )
        if (
            index > 0
            and metric_dot(
                directions_uniform[index],
                directions_uniform[index - 1],
                metric,
            ) < 0.0
        ):
            directions_uniform[index] *= -1.0

    return UniformTail(
        s=s_uniform,
        states=states_uniform,
        reduced_direction=directions_uniform,
        metric=metric,
        omega_scale=omega_scale,
    )


def recurrence_curves(
    directions: np.ndarray,
    metric: np.ndarray,
    *,
    samples_per_rotation: int,
    max_lag_rotations: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_lag_bins = min(
        int(round(max_lag_rotations * samples_per_rotation)),
        directions.shape[0] - 2,
    )
    lag_bins = np.arange(max_lag_bins + 1, dtype=int)
    signed = np.empty(lag_bins.size, dtype=float)
    line = np.empty(lag_bins.size, dtype=float)

    for output_index, lag in enumerate(lag_bins):
        if lag == 0:
            dots = np.ones(directions.shape[0], dtype=float)
        else:
            left = directions[:-lag]
            right = directions[lag:]
            dots = np.einsum("ni,ij,nj->n", left, metric, right)
        signed[output_index] = float(np.mean(dots))
        line[output_index] = float(np.mean(np.abs(dots)))

    return lag_bins / samples_per_rotation, signed, line


def estimate_line_period(
    lag: np.ndarray,
    line: np.ndarray,
    *,
    min_period: float,
    max_period: float,
) -> tuple[float, int]:
    mask = (lag >= min_period) & (lag <= max_period)
    indices = np.flatnonzero(mask)
    if indices.size < 3:
        raise ValueError("Invalid line-period search interval.")

    local = line[indices]
    peaks, _ = find_peaks(
        local,
        height=0.65,
        prominence=0.15,
    )
    if peaks.size:
        chosen = int(indices[peaks[0]])
    else:
        chosen = int(indices[np.argmax(local)])
    return float(lag[chosen]), chosen


def estimate_oriented_period(
    lag: np.ndarray,
    signed: np.ndarray,
    line_period: float,
) -> float:
    mask = (
        (lag >= 1.65 * line_period)
        & (lag <= 2.35 * line_period)
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 2.0 * line_period
    return float(lag[int(indices[np.argmax(signed[indices])])])


def build_state_spline(tail: UniformTail) -> CubicSpline:
    return CubicSpline(
        tail.s,
        tail.states,
        axis=0,
        bc_type="natural",
        extrapolate=False,
    )


def interpolate_direction(
    tail: UniformTail,
    s_value: float,
) -> np.ndarray:
    vector = np.empty(REDUCED_DIMENSION, dtype=float)
    for component in range(REDUCED_DIMENSION):
        vector[component] = np.interp(
            s_value,
            tail.s,
            tail.reduced_direction[:, component],
        )
    return normalize_metric(vector, tail.metric)


def base_feature_matrix(states: np.ndarray, params: ModelParameters) -> np.ndarray:
    y1 = states[:, 0]
    v1 = states[:, 1]
    omega1 = states[:, 3]
    y2 = states[:, 4]
    v2 = states[:, 5]
    omega2 = states[:, 7]
    z = states[:, 8]
    vz = states[:, 9]
    delta = states[:, 2] + states[:, 6] - params.phi0 - states[:, 10]

    return np.column_stack(
        (
            y1,
            v1,
            omega1,
            y2,
            v2,
            omega2,
            z,
            vz,
            np.sin(delta),
            np.cos(delta),
        )
    )


def feature_scales(tail: UniformTail, params: ModelParameters) -> np.ndarray:
    features = base_feature_matrix(tail.states, params)
    std = np.std(features, axis=0, ddof=1)
    magnitude = np.maximum(np.mean(np.abs(features), axis=0), 1.0)
    return np.maximum(std, 1e-9 * magnitude)


def base_closure(
    state_a: np.ndarray,
    state_b: np.ndarray,
    scales: np.ndarray,
    params: ModelParameters,
) -> float:
    features = base_feature_matrix(
        np.vstack((state_a, state_b)),
        params,
    )
    normalized = (features[1] - features[0]) / scales
    return float(np.sqrt(np.mean(normalized**2)))


def relative_matrix_difference(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(a, ord="fro")),
        float(np.linalg.norm(b, ord="fro")),
        1e-15,
    )
    return float(np.linalg.norm(a - b, ord="fro") / denominator)


def integrate_transport(
    state_spline: CubicSpline,
    *,
    start_s: float,
    length: float,
    steps: int,
    kappa0: float,
    params: ModelParameters,
) -> np.ndarray:
    if steps < 20:
        raise ValueError("Transport integration requires at least 20 steps.")

    ds = length / steps
    matrix = np.eye(REDUCED_DIMENSION, dtype=float)

    for step in range(steps):
        s0 = start_s + step * ds
        sm = s0 + 0.5 * ds
        s1 = s0 + ds

        state0 = np.asarray(state_spline(s0), dtype=float)
        statem = np.asarray(state_spline(sm), dtype=float)
        state1 = np.asarray(state_spline(s1), dtype=float)

        b0 = reduced_jacobian_rotation(
            state0,
            kappa0=kappa0,
            params=params,
        )
        bm = reduced_jacobian_rotation(
            statem,
            kappa0=kappa0,
            params=params,
        )
        b1 = reduced_jacobian_rotation(
            state1,
            kappa0=kappa0,
            params=params,
        )

        k1 = b0 @ matrix
        k2 = bm @ (matrix + 0.5 * ds * k1)
        k3 = bm @ (matrix + 0.5 * ds * k2)
        k4 = b1 @ (matrix + ds * k3)
        matrix = matrix + (ds / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )

    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("Transport matrix became non-finite.")
    return matrix


def complex_metric_alignment(
    vector: np.ndarray,
    eigenvector: np.ndarray,
    metric: np.ndarray,
) -> float:
    norm_vector = metric_norm(vector, metric)
    norm_eigen = math.sqrt(
        float(
            np.real(
                np.conjugate(eigenvector) @ metric @ eigenvector
            )
        )
    )
    if norm_eigen <= 0.0 or not np.isfinite(norm_eigen):
        return 0.0
    numerator = abs(
        np.conjugate(eigenvector) @ metric @ vector
    )
    return float(numerator / (norm_vector * norm_eigen))


def matched_eigenpair(
    matrix: np.ndarray,
    direction: np.ndarray,
    metric: np.ndarray,
) -> tuple[complex, float, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    alignments = np.array(
        [
            complex_metric_alignment(
                direction,
                eigenvectors[:, index],
                metric,
            )
            for index in range(eigenvalues.size)
        ],
        dtype=float,
    )
    best = int(np.argmax(alignments))
    return complex(eigenvalues[best]), float(alignments[best]), eigenvalues


def gauge_invariance_check(
    tail: UniformTail,
    *,
    sample_count: int,
    kappa0: float,
    params: ModelParameters,
) -> tuple[float, float]:
    indices = np.linspace(
        0,
        tail.states.shape[0] - 1,
        min(sample_count, tail.states.shape[0]),
        dtype=int,
    )
    reference_name = "symmetric"
    other_names = ["theta1", "theta2", "support"]
    max_absolute = 0.0
    max_relative = 0.0

    for index in indices:
        state = tail.states[index]
        reference = reduced_jacobian_rotation(
            state,
            kappa0=kappa0,
            params=params,
            embedding_name=reference_name,
        )
        for name in other_names:
            candidate = reduced_jacobian_rotation(
                state,
                kappa0=kappa0,
                params=params,
                embedding_name=name,
            )
            difference = candidate - reference
            max_absolute = max(
                max_absolute,
                float(np.max(np.abs(difference))),
            )
            max_relative = max(
                max_relative,
                relative_matrix_difference(candidate, reference),
            )

    return max_absolute, max_relative


def analyze_case(
    orbit: OrbitData,
    tail: UniformTail,
    *,
    samples_per_rotation: int,
    max_recurrence_lag: float,
    min_line_period: float,
    max_line_period: float,
    anchor_count: int,
    anchor_span_rotations: float,
    transport_steps: int,
    gauge_check_samples: int,
    resolution_check_anchors: int,
    params: ModelParameters,
    closure_tolerance: float,
    line_overlap_tolerance: float,
    signed_overlap_tolerance: float,
) -> CaseResult:
    kappa0 = (55.0 / 6.0) * (10.0 / orbit.q)

    recurrence_lag, recurrence_signed, recurrence_line = recurrence_curves(
        tail.reduced_direction,
        tail.metric,
        samples_per_rotation=samples_per_rotation,
        max_lag_rotations=max_recurrence_lag,
    )
    line_period, _ = estimate_line_period(
        recurrence_lag,
        recurrence_line,
        min_period=min_line_period,
        max_period=max_line_period,
    )
    oriented_period = estimate_oriented_period(
        recurrence_lag,
        recurrence_signed,
        line_period,
    )

    gauge_abs, gauge_rel = gauge_invariance_check(
        tail,
        sample_count=gauge_check_samples,
        kappa0=kappa0,
        params=params,
    )

    state_spline = build_state_spline(tail)
    scales = feature_scales(tail, params)

    available_end = float(tail.s[-1] - 2.0 * line_period)
    available_start = max(
        float(tail.s[0]),
        available_end - anchor_span_rotations,
    )
    if available_end <= available_start:
        raise RuntimeError("Tail is too short for anchor analysis.")

    anchors_s = np.linspace(
        available_start,
        available_end,
        anchor_count,
    )

    records: list[AnchorRecord] = []
    spectra_p: list[np.ndarray] = []
    spectra_2p: list[np.ndarray] = []

    for anchor_index, anchor_s in enumerate(anchors_s):
        state0 = np.asarray(state_spline(anchor_s), dtype=float)
        state_p = np.asarray(
            state_spline(anchor_s + line_period),
            dtype=float,
        )
        state_2p = np.asarray(
            state_spline(anchor_s + 2.0 * line_period),
            dtype=float,
        )

        b0 = reduced_jacobian_rotation(
            state0,
            kappa0=kappa0,
            params=params,
        )
        bp = reduced_jacobian_rotation(
            state_p,
            kappa0=kappa0,
            params=params,
        )
        b2p = reduced_jacobian_rotation(
            state_2p,
            kappa0=kappa0,
            params=params,
        )

        coefficient_closure_p = relative_matrix_difference(b0, bp)
        coefficient_closure_2p = relative_matrix_difference(b0, b2p)
        base_closure_p = base_closure(
            state0,
            state_p,
            scales,
            params,
        )
        base_closure_2p = base_closure(
            state0,
            state_2p,
            scales,
            params,
        )

        u0 = interpolate_direction(tail, anchor_s)
        up = interpolate_direction(tail, anchor_s + line_period)
        u2p = interpolate_direction(tail, anchor_s + 2.0 * line_period)

        observed_signed_p = metric_dot(u0, up, tail.metric)
        observed_signed_2p = metric_dot(u0, u2p, tail.metric)
        observed_line_p = abs(observed_signed_p)
        observed_line_2p = abs(observed_signed_2p)

        u0_e = u0 / np.linalg.norm(u0)
        up_e = up / np.linalg.norm(up)
        u2p_e = u2p / np.linalg.norm(u2p)
        euclidean_signed_p = float(np.dot(u0_e, up_e))
        euclidean_signed_2p = float(np.dot(u0_e, u2p_e))

        matrix_p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=line_period,
            steps=transport_steps,
            kappa0=kappa0,
            params=params,
        )
        matrix_2p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=2.0 * line_period,
            steps=2 * transport_steps,
            kappa0=kappa0,
            params=params,
        )

        predicted_p = matrix_p @ u0
        predicted_2p = matrix_2p @ u0
        norm_p = metric_norm(predicted_p, tail.metric)
        norm_2p = metric_norm(predicted_2p, tail.metric)
        normalized_predicted_p = predicted_p / norm_p
        normalized_predicted_2p = predicted_2p / norm_2p

        predicted_observed_signed_p = metric_dot(
            normalized_predicted_p,
            up,
            tail.metric,
        )
        predicted_observed_signed_2p = metric_dot(
            normalized_predicted_2p,
            u2p,
            tail.metric,
        )

        transport_start_signed_p = metric_dot(
            normalized_predicted_p,
            u0,
            tail.metric,
        )
        transport_start_signed_2p = metric_dot(
            normalized_predicted_2p,
            u0,
            tail.metric,
        )
        signed_gain_p = math.copysign(
            norm_p,
            transport_start_signed_p,
        )
        signed_gain_2p = math.copysign(
            norm_2p,
            transport_start_signed_2p,
        )

        matched_p, alignment_p, spectrum_p = matched_eigenpair(
            matrix_p,
            u0,
            tail.metric,
        )
        matched_2p, alignment_2p, spectrum_2p = matched_eigenpair(
            matrix_2p,
            u0,
            tail.metric,
        )
        spectra_p.append(spectrum_p)
        spectra_2p.append(spectrum_2p)

        records.append(
            AnchorRecord(
                q=orbit.q,
                anchor_index=anchor_index,
                anchor_s=float(anchor_s),
                anchor_phase=float(anchor_s % 1.0),
                line_period=line_period,
                coefficient_closure_p=coefficient_closure_p,
                coefficient_closure_2p=coefficient_closure_2p,
                base_closure_p=base_closure_p,
                base_closure_2p=base_closure_2p,
                observed_line_overlap_p=observed_line_p,
                observed_signed_overlap_p=observed_signed_p,
                observed_line_overlap_2p=observed_line_2p,
                observed_signed_overlap_2p=observed_signed_2p,
                euclidean_signed_overlap_p=euclidean_signed_p,
                euclidean_signed_overlap_2p=euclidean_signed_2p,
                transport_observed_line_overlap_p=abs(
                    predicted_observed_signed_p
                ),
                transport_observed_signed_overlap_p=(
                    predicted_observed_signed_p
                ),
                transport_observed_line_overlap_2p=abs(
                    predicted_observed_signed_2p
                ),
                transport_observed_signed_overlap_2p=(
                    predicted_observed_signed_2p
                ),
                signed_gain_p=signed_gain_p,
                signed_gain_2p=signed_gain_2p,
                log_abs_gain_p=math.log(abs(signed_gain_p)),
                log_abs_gain_2p=math.log(abs(signed_gain_2p)),
                matched_eigenvalue_p_real=float(np.real(matched_p)),
                matched_eigenvalue_p_imag=float(np.imag(matched_p)),
                matched_eigenvalue_p_abs=float(abs(matched_p)),
                matched_eigenvector_alignment_p=alignment_p,
                matched_eigenvalue_2p_real=float(np.real(matched_2p)),
                matched_eigenvalue_2p_imag=float(np.imag(matched_2p)),
                matched_eigenvalue_2p_abs=float(abs(matched_2p)),
                matched_eigenvector_alignment_2p=alignment_2p,
                transport_matrix_condition_p=float(np.linalg.cond(matrix_p)),
                transport_matrix_condition_2p=float(np.linalg.cond(matrix_2p)),
            )
        )

    median_index = len(records) // 2
    median_spectrum_p = spectra_p[median_index]
    median_spectrum_2p = spectra_2p[median_index]

    resolution_indices = np.linspace(
        0,
        len(records) - 1,
        min(resolution_check_anchors, len(records)),
        dtype=int,
    )
    resolution_errors_p: list[float] = []
    resolution_errors_2p: list[float] = []
    for index in resolution_indices:
        anchor_s = records[index].anchor_s
        coarse_p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=line_period,
            steps=transport_steps,
            kappa0=kappa0,
            params=params,
        )
        fine_p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=line_period,
            steps=2 * transport_steps,
            kappa0=kappa0,
            params=params,
        )
        coarse_2p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=2.0 * line_period,
            steps=2 * transport_steps,
            kappa0=kappa0,
            params=params,
        )
        fine_2p = integrate_transport(
            state_spline,
            start_s=anchor_s,
            length=2.0 * line_period,
            steps=4 * transport_steps,
            kappa0=kappa0,
            params=params,
        )
        resolution_errors_p.append(
            relative_matrix_difference(coarse_p, fine_p)
        )
        resolution_errors_2p.append(
            relative_matrix_difference(coarse_2p, fine_2p)
        )

    resolution_error_p = float(np.max(resolution_errors_p))
    resolution_error_2p = float(np.max(resolution_errors_2p))

    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(record, name) for record in records])

    median_coefficient_p = float(np.median(values("coefficient_closure_p")))
    median_coefficient_2p = float(np.median(values("coefficient_closure_2p")))
    max_coefficient_p = float(np.max(values("coefficient_closure_p")))
    max_coefficient_2p = float(np.max(values("coefficient_closure_2p")))
    median_base_p = float(np.median(values("base_closure_p")))
    median_base_2p = float(np.median(values("base_closure_2p")))
    median_line_p = float(np.median(values("observed_line_overlap_p")))
    median_line_2p = float(np.median(values("observed_line_overlap_2p")))
    median_signed_p = float(np.median(values("observed_signed_overlap_p")))
    median_signed_2p = float(np.median(values("observed_signed_overlap_2p")))
    min_line_p = float(np.min(values("observed_line_overlap_p")))
    min_line_2p = float(np.min(values("observed_line_overlap_2p")))
    median_transport_match_p = float(
        np.median(values("transport_observed_line_overlap_p"))
    )
    median_transport_match_2p = float(
        np.median(values("transport_observed_line_overlap_2p"))
    )
    median_signed_gain_p = float(np.median(values("signed_gain_p")))
    median_signed_gain_2p = float(np.median(values("signed_gain_2p")))
    sign_consistency_p = float(
        np.mean(np.sign(values("observed_signed_overlap_p")) == -1.0)
    )
    sign_consistency_2p = float(
        np.mean(np.sign(values("observed_signed_overlap_2p")) == 1.0)
    )
    metric_sign_disagreement_p = float(
        np.mean(
            np.sign(values("observed_signed_overlap_p"))
            != np.sign(values("euclidean_signed_overlap_p"))
        )
    )
    metric_sign_disagreement_2p = float(
        np.mean(
            np.sign(values("observed_signed_overlap_2p"))
            != np.sign(values("euclidean_signed_overlap_2p"))
        )
    )

    coefficient_closed_p = bool(
        median_coefficient_p <= closure_tolerance
        and max_coefficient_p <= 5.0 * closure_tolerance
    )
    coefficient_closed_2p = bool(
        median_coefficient_2p <= closure_tolerance
        and max_coefficient_2p <= 5.0 * closure_tolerance
    )
    antiperiodic_line = bool(
        median_line_p >= line_overlap_tolerance
        and min_line_p >= line_overlap_tolerance - 0.01
        and median_signed_p <= -signed_overlap_tolerance
        and sign_consistency_p >= 0.95
    )
    oriented_return = bool(
        median_line_2p >= line_overlap_tolerance
        and min_line_2p >= line_overlap_tolerance - 0.01
        and median_signed_2p >= signed_overlap_tolerance
        and sign_consistency_2p >= 0.95
    )
    gauge_passed = bool(gauge_rel <= 1e-10)
    metric_sign_passed = bool(
        metric_sign_disagreement_p == 0.0
        and metric_sign_disagreement_2p == 0.0
    )
    resolution_passed = bool(
        resolution_error_p <= 5e-5
        and resolution_error_2p <= 1e-4
    )

    if (
        coefficient_closed_p
        and antiperiodic_line
        and oriented_return
        and gauge_passed
        and metric_sign_passed
    ):
        classification = (
            "Z2=-1 Möbius-line-bundle candidate supported on the P_line "
            "coefficient loop."
        )
        z2_value = -1.0
    elif (
        not coefficient_closed_p
        and coefficient_closed_2p
        and antiperiodic_line
        and oriented_return
    ):
        classification = (
            "Antiperiodic half-cycle supported; the base tangent equation "
            "closes only after 2 P_line, so Möbius over P_line is not "
            "established."
        )
        z2_value = float("nan")
    elif antiperiodic_line and oriented_return:
        classification = (
            "Recurrent antiperiodic tangent line supported, but the reduced "
            "coefficient loop does not close at the tested interval; this is "
            "transport recurrence, not yet a Möbius/Floquet proof."
        )
        z2_value = float("nan")
    elif coefficient_closed_p and median_signed_p > 0.0:
        classification = (
            "The P_line coefficient loop closes with an orientable positive "
            "line return; Z2=+1."
        )
        z2_value = 1.0
    else:
        classification = (
            "Inconclusive: recurrence, closure, or robustness conditions did "
            "not pass simultaneously."
        )
        z2_value = float("nan")

    summary: dict[str, float | str | bool] = {
        "line_period_rotations": line_period,
        "oriented_period_rotations": oriented_period,
        "median_coefficient_closure_p": median_coefficient_p,
        "median_coefficient_closure_2p": median_coefficient_2p,
        "maximum_coefficient_closure_p": max_coefficient_p,
        "maximum_coefficient_closure_2p": max_coefficient_2p,
        "median_base_closure_p": median_base_p,
        "median_base_closure_2p": median_base_2p,
        "median_line_overlap_p": median_line_p,
        "median_signed_overlap_p": median_signed_p,
        "minimum_line_overlap_p": min_line_p,
        "median_line_overlap_2p": median_line_2p,
        "median_signed_overlap_2p": median_signed_2p,
        "minimum_line_overlap_2p": min_line_2p,
        "median_transport_observed_line_overlap_p": median_transport_match_p,
        "median_transport_observed_line_overlap_2p": median_transport_match_2p,
        "median_signed_gain_p": median_signed_gain_p,
        "median_signed_gain_2p": median_signed_gain_2p,
        "negative_return_fraction_p": sign_consistency_p,
        "positive_return_fraction_2p": sign_consistency_2p,
        "metric_sign_disagreement_fraction_p": metric_sign_disagreement_p,
        "metric_sign_disagreement_fraction_2p": metric_sign_disagreement_2p,
        "gauge_max_absolute_error": gauge_abs,
        "gauge_max_relative_error": gauge_rel,
        "resolution_relative_error_p": resolution_error_p,
        "resolution_relative_error_2p": resolution_error_2p,
        "coefficient_closed_p": coefficient_closed_p,
        "coefficient_closed_2p": coefficient_closed_2p,
        "antiperiodic_line_passed": antiperiodic_line,
        "oriented_return_passed": oriented_return,
        "gauge_invariance_passed": gauge_passed,
        "metric_sign_invariance_passed": metric_sign_passed,
        "resolution_check_passed": resolution_passed,
        "z2_value": z2_value,
        "classification": classification,
    }

    return CaseResult(
        q=orbit.q,
        source=orbit.source,
        line_period=line_period,
        oriented_period=oriented_period,
        recurrence_lag=recurrence_lag,
        recurrence_signed=recurrence_signed,
        recurrence_line=recurrence_line,
        anchors=records,
        median_spectrum_p=median_spectrum_p,
        median_spectrum_2p=median_spectrum_2p,
        metric=tail.metric,
        gauge_max_absolute_error=gauge_abs,
        gauge_max_relative_error=gauge_rel,
        resolution_relative_error_p=resolution_error_p,
        resolution_relative_error_2p=resolution_error_2p,
        summary=summary,
    )


def q_token(q: float) -> str:
    text = f"{q:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def save_anchor_plot(path: Path, result: CaseResult) -> None:
    phase = np.asarray([record.anchor_phase for record in result.anchors])
    coefficient_p = np.asarray(
        [record.coefficient_closure_p for record in result.anchors]
    )
    coefficient_2p = np.asarray(
        [record.coefficient_closure_2p for record in result.anchors]
    )
    signed_p = np.asarray(
        [record.observed_signed_overlap_p for record in result.anchors]
    )
    signed_2p = np.asarray(
        [record.observed_signed_overlap_2p for record in result.anchors]
    )
    gain_p = np.asarray([record.signed_gain_p for record in result.anchors])
    gain_2p = np.asarray([record.signed_gain_2p for record in result.anchors])
    transport_match_p = np.asarray(
        [
            record.transport_observed_line_overlap_p
            for record in result.anchors
        ]
    )
    transport_match_2p = np.asarray(
        [
            record.transport_observed_line_overlap_2p
            for record in result.anchors
        ]
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(phase, coefficient_p, marker="o", label="P_line")
    ax1.plot(phase, coefficient_2p, marker="s", label="2 P_line")
    ax1.set_yscale("log")
    ax1.set_title("Reduced coefficient-loop closure")
    ax1.set_xlabel("anchor support phase")
    ax1.set_ylabel("relative Frobenius mismatch")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(phase, signed_p, marker="o", label="observed P_line")
    ax2.plot(phase, signed_2p, marker="s", label="observed 2 P_line")
    ax2.axhline(0.0, linewidth=1.0)
    ax2.axhline(-1.0, linewidth=0.8, linestyle=":")
    ax2.axhline(+1.0, linewidth=0.8, linestyle=":")
    ax2.set_title("Signed tangent return")
    ax2.set_xlabel("anchor support phase")
    ax2.set_ylabel("metric overlap")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(phase, gain_p, marker="o", label="signed gain P_line")
    ax3.plot(phase, gain_2p, marker="s", label="signed gain 2 P_line")
    ax3.axhline(0.0, linewidth=1.0)
    ax3.set_title("Transport gain along the recurrent line")
    ax3.set_xlabel("anchor support phase")
    ax3.set_ylabel("signed metric gain")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(
        phase,
        transport_match_p,
        marker="o",
        label="transport vs observed P_line",
    )
    ax4.plot(
        phase,
        transport_match_2p,
        marker="s",
        label="transport vs observed 2 P_line",
    )
    ax4.set_title("Fundamental-matrix prediction check")
    ax4.set_xlabel("anchor support phase")
    ax4.set_ylabel("tangent-line overlap")
    ax4.set_ylim(0.0, 1.02)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    figure.suptitle(
        f"Möbius / antiperiodic anchor audit at Q={result.q:g}",
        fontsize=16,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_spectrum_plot(path: Path, result: CaseResult) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
        constrained_layout=True,
    )

    for axis, spectrum, label in [
        (axes[0], result.median_spectrum_p, "P_line transport"),
        (axes[1], result.median_spectrum_2p, "2 P_line transport"),
    ]:
        axis.scatter(np.real(spectrum), np.imag(spectrum), s=55)
        angles = np.linspace(0.0, 2.0 * np.pi, 400)
        axis.plot(np.cos(angles), np.sin(angles), linestyle=":", linewidth=1.0)
        axis.axhline(0.0, linewidth=0.8)
        axis.axvline(0.0, linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.set_xlabel("real part")
        axis.set_ylabel("imaginary part")
        axis.grid(True, alpha=0.3)

    figure.suptitle(
        f"Reduced transport spectrum at the median anchor, Q={result.q:g}",
        fontsize=16,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_recurrence_plot(path: Path, results: list[CaseResult]) -> None:
    figure, axis = plt.subplots(figsize=(11, 7))

    for result in results:
        stroke_lag = result.recurrence_lag / result.line_period
        axis.plot(
            stroke_lag,
            result.recurrence_line,
            linewidth=2.0,
            label=f"line overlap Q={result.q:g}",
        )
        axis.plot(
            stroke_lag,
            result.recurrence_signed,
            linestyle="--",
            linewidth=1.5,
            label=f"signed overlap Q={result.q:g}",
        )

    axis.axvline(1.0, linewidth=1.0, linestyle=":")
    axis.axvline(2.0, linewidth=1.0, linestyle=":")
    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Tangent recurrence on the inferred line-stroke clock")
    axis.set_xlabel("lag [P_line strokes]")
    axis.set_ylabel("mean overlap")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_decision_plot(path: Path, results: list[CaseResult]) -> None:
    labels = [f"Q={result.q:g}" for result in results]
    x = np.arange(len(results), dtype=float)
    width = 0.18

    coefficient_p = [
        float(result.summary["median_coefficient_closure_p"])
        for result in results
    ]
    coefficient_2p = [
        float(result.summary["median_coefficient_closure_2p"])
        for result in results
    ]
    signed_p = [
        float(result.summary["median_signed_overlap_p"])
        for result in results
    ]
    signed_2p = [
        float(result.summary["median_signed_overlap_2p"])
        for result in results
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        constrained_layout=True,
    )

    axes[0].bar(x - width / 2.0, coefficient_p, width, label="closure P_line")
    axes[0].bar(x + width / 2.0, coefficient_2p, width, label="closure 2 P_line")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("median coefficient mismatch")
    axes[0].set_title("Does the reduced tangent equation close?")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2.0, signed_p, width, label="signed return P_line")
    axes[1].bar(x + width / 2.0, signed_2p, width, label="signed return 2 P_line")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("median tangent overlap")
    axes[1].set_title("Does the line return reversed and then oriented?")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    figure.suptitle("Möbius decision diagnostics", fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_anchors_csv(path: Path, results: list[CaseResult]) -> None:
    field_names = list(AnchorRecord.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(field_names)
        for result in results:
            for record in result.anchors:
                writer.writerow([getattr(record, name) for name in field_names])


def save_spectra_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "interval",
                "eigenvalue_index",
                "real",
                "imaginary",
                "absolute_value",
            ]
        )
        for result in results:
            for interval, spectrum in [
                ("P_line", result.median_spectrum_p),
                ("2P_line", result.median_spectrum_2p),
            ]:
                for index, eigenvalue in enumerate(spectrum):
                    writer.writerow(
                        [
                            result.q,
                            interval,
                            index,
                            np.real(eigenvalue),
                            np.imag(eigenvalue),
                            abs(eigenvalue),
                        ]
                    )


def save_summary_csv(path: Path, results: list[CaseResult]) -> None:
    keys = list(results[0].summary.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Q", "source", *keys])
        for result in results:
            writer.writerow(
                [
                    result.q,
                    str(result.source.resolve()),
                    *[result.summary[key] for key in keys],
                ]
            )


def save_raw_npz(path: Path, results: list[CaseResult]) -> None:
    payload: dict[str, object] = {"script_version": SCRIPT_VERSION}
    for result in results:
        prefix = f"q_{q_token(result.q)}"
        payload[f"{prefix}_q"] = result.q
        payload[f"{prefix}_line_period"] = result.line_period
        payload[f"{prefix}_oriented_period"] = result.oriented_period
        payload[f"{prefix}_recurrence_lag"] = result.recurrence_lag
        payload[f"{prefix}_recurrence_signed"] = result.recurrence_signed
        payload[f"{prefix}_recurrence_line"] = result.recurrence_line
        payload[f"{prefix}_median_spectrum_p"] = result.median_spectrum_p
        payload[f"{prefix}_median_spectrum_2p"] = result.median_spectrum_2p
        payload[f"{prefix}_metric"] = result.metric
        for key, value in result.summary.items():
            if isinstance(value, (float, int, bool, np.number)):
                payload[f"{prefix}_summary_{key}"] = value
    np.savez_compressed(path, **payload)


def build_report(
    results: list[CaseResult],
    output_paths: dict[str, Path],
    args: argparse.Namespace,
) -> str:
    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "PURPOSE",
        "-------",
        "Distinguish a genuine Z2/Möbius tangent-line return from a negative",
        "transport multiplier, half-cycle antiperiodicity, or gauge artifact.",
        "",
        "DECISION REQUIREMENTS",
        "---------------------",
        f"coefficient closure tolerance       = {args.closure_tolerance:.12e}",
        f"minimum tangent-line overlap        = {args.line_overlap_tolerance:.12f}",
        f"minimum signed return magnitude     = {args.signed_overlap_tolerance:.12f}",
        "",
    ]

    for result in results:
        s = result.summary
        lines.extend(
            [
                f"Q={result.q:g}",
                f"  source orbit                         = {result.source.resolve()}",
                "",
                "  Recurrent line",
                f"    P_line                              = {float(s['line_period_rotations']):.12f} rotations",
                f"    P_oriented                          = {float(s['oriented_period_rotations']):.12f} rotations",
                f"    median line overlap at P            = {float(s['median_line_overlap_p']):.12e}",
                f"    median signed overlap at P          = {float(s['median_signed_overlap_p']):+.12e}",
                f"    median line overlap at 2P           = {float(s['median_line_overlap_2p']):.12e}",
                f"    median signed overlap at 2P         = {float(s['median_signed_overlap_2p']):+.12e}",
                f"    negative return fraction at P       = {float(s['negative_return_fraction_p']):.9f}",
                f"    positive return fraction at 2P      = {float(s['positive_return_fraction_2p']):.9f}",
                "",
                "  Reduced coefficient-loop closure",
                f"    median mismatch at P                = {float(s['median_coefficient_closure_p']):.12e}",
                f"    maximum mismatch at P               = {float(s['maximum_coefficient_closure_p']):.12e}",
                f"    median mismatch at 2P               = {float(s['median_coefficient_closure_2p']):.12e}",
                f"    maximum mismatch at 2P              = {float(s['maximum_coefficient_closure_2p']):.12e}",
                f"    coefficient loop closed at P        = {s['coefficient_closed_p']}",
                f"    coefficient loop closed at 2P       = {s['coefficient_closed_2p']}",
                "",
                "  Central reduced-base closure",
                f"    median normalized mismatch at P     = {float(s['median_base_closure_p']):.12e}",
                f"    median normalized mismatch at 2P    = {float(s['median_base_closure_2p']):.12e}",
                "",
                "  Fundamental transport",
                f"    median transport-vs-observed line P = {float(s['median_transport_observed_line_overlap_p']):.12e}",
                f"    median transport-vs-observed 2P     = {float(s['median_transport_observed_line_overlap_2p']):.12e}",
                f"    median signed gain at P             = {float(s['median_signed_gain_p']):+.12e}",
                f"    median signed gain at 2P            = {float(s['median_signed_gain_2p']):+.12e}",
                "",
                "  Robustness",
                f"    gauge max absolute error            = {float(s['gauge_max_absolute_error']):.12e}",
                f"    gauge max relative error            = {float(s['gauge_max_relative_error']):.12e}",
                f"    metric sign disagreements at P      = {float(s['metric_sign_disagreement_fraction_p']):.9f}",
                f"    metric sign disagreements at 2P     = {float(s['metric_sign_disagreement_fraction_2p']):.9f}",
                f"    transport resolution error at P     = {float(s['resolution_relative_error_p']):.12e}",
                f"    transport resolution error at 2P    = {float(s['resolution_relative_error_2p']):.12e}",
                f"    gauge invariance passed             = {s['gauge_invariance_passed']}",
                f"    metric-sign invariance passed       = {s['metric_sign_invariance_passed']}",
                f"    resolution check passed             = {s['resolution_check_passed']}",
                "",
                "  Topological classification",
                f"    Z2 value                             = {s['z2_value']}",
                f"    classification                       = {s['classification']}",
                "",
            ]
        )

    lines.extend(
        [
            "CAUTION",
            "-------",
            "A negative tangent return alone is not a Möbius proof.  The same",
            "interval must close the reduced tangent coefficient loop.  If the",
            "loop closes only after 2 P_line, the correct statement is a",
            "negative half-cycle transport / antiperiodic mode, not a Möbius",
            "bundle over the shorter interval.",
            "",
            f"decision PNG   = {output_paths['decision'].resolve()}",
            f"recurrence PNG = {output_paths['recurrence'].resolve()}",
            f"anchors CSV    = {output_paths['anchors'].resolve()}",
            f"spectra CSV    = {output_paths['spectra'].resolve()}",
            f"summary CSV    = {output_paths['summary'].resolve()}",
            f"raw NPZ        = {output_paths['raw'].resolve()}",
        ]
    )

    for key, path in output_paths.items():
        if key.startswith("anchor_Q"):
            lines.append(f"{key} PNG = {path.resolve()}")
        elif key.startswith("spectrum_Q"):
            lines.append(f"{key} PNG = {path.resolve()}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the recurrent antiperiodic tangent line defines a "
            "closed-loop Z2/Möbius structure or only a negative transport "
            "half-cycle."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help=(
            "repeatable Q=orbit_cache.npz; defaults to Q=522.25 and Q=550"
        ),
    )
    parser.add_argument("--tail-periods", type=int, default=70)
    parser.add_argument("--steps-per-period", type=int, default=160)
    parser.add_argument("--samples-per-rotation", type=int, default=720)
    parser.add_argument("--max-recurrence-lag", type=float, default=0.65)
    parser.add_argument("--min-line-period", type=float, default=0.12)
    parser.add_argument("--max-line-period", type=float, default=0.32)
    parser.add_argument("--anchor-count", type=int, default=24)
    parser.add_argument("--anchor-span-rotations", type=float, default=6.0)
    parser.add_argument("--transport-steps", type=int, default=320)
    parser.add_argument("--gauge-check-samples", type=int, default=24)
    parser.add_argument("--resolution-check-anchors", type=int, default=4)
    parser.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--closure-tolerance", type=float, default=1e-2)
    parser.add_argument("--line-overlap-tolerance", type=float, default=0.99)
    parser.add_argument("--signed-overlap-tolerance", type=float, default=0.99)
    parser.add_argument(
        "--output-prefix",
        default="support_mobius_monodromy_audit",
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

    if args.tail_periods < 20:
        raise ValueError("--tail-periods must be at least 20.")
    if args.steps_per_period < 40:
        raise ValueError("--steps-per-period must be at least 40.")
    if args.samples_per_rotation < 180:
        raise ValueError("--samples-per-rotation must be at least 180.")
    if args.max_recurrence_lag <= args.max_line_period:
        raise ValueError(
            "--max-recurrence-lag must exceed --max-line-period."
        )
    if not 0.0 < args.min_line_period < args.max_line_period:
        raise ValueError("Invalid line-period search interval.")
    if args.anchor_count < 4:
        raise ValueError("--anchor-count must be at least 4.")
    if args.anchor_span_rotations <= 0.5:
        raise ValueError("--anchor-span-rotations must exceed 0.5.")
    if args.transport_steps < 40:
        raise ValueError("--transport-steps must be at least 40.")
    if args.gauge_check_samples < 4:
        raise ValueError("--gauge-check-samples must be at least 4.")
    if args.resolution_check_anchors < 1:
        raise ValueError("--resolution-check-anchors must be positive.")
    if args.initial_x_perturbation <= 0.0:
        raise ValueError("--initial-x-perturbation must be positive.")
    if args.closure_tolerance <= 0.0:
        raise ValueError("--closure-tolerance must be positive.")
    if not 0.0 < args.line_overlap_tolerance <= 1.0:
        raise ValueError("Invalid --line-overlap-tolerance.")
    if not 0.0 < args.signed_overlap_tolerance <= 1.0:
        raise ValueError("Invalid --signed-overlap-tolerance.")

    params = ModelParameters(gamma_rho=args.gamma_rho)

    print(f"Script version             = {SCRIPT_VERSION}")
    print(f"tail periods               = {args.tail_periods}")
    print(f"steps per period           = {args.steps_per_period}")
    print(f"samples per rotation       = {args.samples_per_rotation}")
    print(
        f"line-period search         = "
        f"[{args.min_line_period}, {args.max_line_period}]"
    )
    print(f"anchor count               = {args.anchor_count}")
    print(f"anchor span rotations      = {args.anchor_span_rotations}")
    print(f"transport steps per P_line = {args.transport_steps}")
    print(f"closure tolerance          = {args.closure_tolerance:.3e}")
    print("cases:")
    for q, path in cases:
        print(f"  Q={q:g}: {path}")

    results: list[CaseResult] = []
    for q, path in sorted(cases, key=lambda item: item[0]):
        orbit = load_orbit(q, path)
        tail = collect_uniform_tail(
            orbit,
            tail_periods=args.tail_periods,
            steps_per_period=args.steps_per_period,
            samples_per_rotation=args.samples_per_rotation,
            initial_x_perturbation=args.initial_x_perturbation,
            params=params,
            progress_every=args.progress_every,
        )
        result = analyze_case(
            orbit,
            tail,
            samples_per_rotation=args.samples_per_rotation,
            max_recurrence_lag=args.max_recurrence_lag,
            min_line_period=args.min_line_period,
            max_line_period=args.max_line_period,
            anchor_count=args.anchor_count,
            anchor_span_rotations=args.anchor_span_rotations,
            transport_steps=args.transport_steps,
            gauge_check_samples=args.gauge_check_samples,
            resolution_check_anchors=args.resolution_check_anchors,
            params=params,
            closure_tolerance=args.closure_tolerance,
            line_overlap_tolerance=args.line_overlap_tolerance,
            signed_overlap_tolerance=args.signed_overlap_tolerance,
        )
        results.append(result)

        print()
        print(f"Q={q:g} classification:")
        print(f"  P_line              = {result.line_period:.12f}")
        print(
            "  coefficient closure P / 2P = "
            f"{float(result.summary['median_coefficient_closure_p']):.3e} / "
            f"{float(result.summary['median_coefficient_closure_2p']):.3e}"
        )
        print(
            "  signed return P / 2P       = "
            f"{float(result.summary['median_signed_overlap_p']):+.9f} / "
            f"{float(result.summary['median_signed_overlap_2p']):+.9f}"
        )
        print(f"  {result.summary['classification']}")

    prefix = Path(args.output_prefix)
    output_paths: dict[str, Path] = {
        "decision": prefix.with_name(prefix.name + "_decision.png"),
        "recurrence": prefix.with_name(prefix.name + "_recurrence.png"),
        "anchors": prefix.with_name(prefix.name + "_anchors.csv"),
        "spectra": prefix.with_name(prefix.name + "_spectra.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.csv"),
        "raw": prefix.with_name(prefix.name + "_raw.npz"),
        "report": prefix.with_name(prefix.name + "_report.txt"),
    }

    for result in results:
        token = q_token(result.q)
        output_paths[f"anchor_Q{token}"] = prefix.with_name(
            prefix.name + f"_Q{token}_anchors.png"
        )
        output_paths[f"spectrum_Q{token}"] = prefix.with_name(
            prefix.name + f"_Q{token}_spectrum.png"
        )
        save_anchor_plot(output_paths[f"anchor_Q{token}"], result)
        save_spectrum_plot(output_paths[f"spectrum_Q{token}"], result)

    save_decision_plot(output_paths["decision"], results)
    save_recurrence_plot(output_paths["recurrence"], results)
    save_anchors_csv(output_paths["anchors"], results)
    save_spectra_csv(output_paths["spectra"], results)
    save_summary_csv(output_paths["summary"], results)
    save_raw_npz(output_paths["raw"], results)

    report = build_report(results, output_paths, args)
    output_paths["report"].write_text(report, encoding="utf-8")

    print()
    print(report)


if __name__ == "__main__":
    main()
