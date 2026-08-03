#!/usr/bin/env python3
"""Track the feedback stability curtain as Q changes.

For every requested Q this script:

1. prepares an independent LOW-history central state:
       warm-up at Q_start,
       linear period-by-period ramp to Q,
       fixed-Q recorded hold;

2. integrates the analytic tangent system on that recorded orbit for
   continuous diagnostic feedback gains:

       eta_phase in [0, 1]
       eta_kappa in [0, 1];

3. finds, for every eta_kappa,

       eta_phase_critical(Q, eta_kappa)

   such that

       lambda(Q, eta_phase_critical, eta_kappa) = 0;

4. finds the top-edge touch curve

       Q_touch(eta_kappa)

   from

       lambda(Q_touch, eta_phase=1, eta_kappa) = 0.

The physical closed loop is the corner

       eta_phase = 1, eta_kappa = 1.

Therefore Q_touch(1) is the point where the physical system first pierces
the local stability curtain.

The central orbit is never reused across different Q values. Each Q is
prepared independently from the same warmed Q_start state, avoiding
continuation-history contamination.

The script uses:
- LSODA/DOP853 for the nonlinear preparation and orbit recording;
- vectorized analytic-tangent RK4 for the gain scan;
- period-wise tangent renormalization;
- mandatory OPEN-loop damping validation at every Q;
- optional comparison with q_growth_threshold_refined.csv;
- resumable per-Q cache files.

Outputs are separate figures rather than a multi-panel dashboard:
- critical phase map over (Q, eta_kappa);
- critical phase versus Q;
- top-edge lambda versus Q;
- Q_touch versus eta_kappa.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import expit


ALPHA, BETA, GAMMA = 1.0, 0.8, 0.2
SCRIPT_VERSION = "2026-08-01-feedback-curtain-vs-q-v1.1-endpoint-fix"
ORBIT_CACHE_COMPAT_VERSION = "2026-08-01-feedback-curtain-vs-q-v1"


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
class GrowthReference:
    available: bool
    slope: float
    q_critical: float
    r_squared: float
    q: np.ndarray
    growth_rate: np.ndarray
    source: str


@dataclass
class BatchResult:
    eta_phase: np.ndarray
    eta_kappa: np.ndarray
    lambda_retained: np.ndarray
    block_standard_error: np.ndarray
    first_half_lambda: np.ndarray
    second_half_lambda: np.ndarray
    drift: np.ndarray


@dataclass
class QScanResult:
    q: float
    periods: int
    discard_periods: int
    phase_values: np.ndarray
    kappa_values: np.ndarray
    lambda_map: np.ndarray
    stderr_map: np.ndarray
    drift_map: np.ndarray
    lambda_phase_zero: np.ndarray
    lambda_phase_one: np.ndarray
    eta_phase_critical: np.ndarray
    lambda_at_critical: np.ndarray
    phase_bracket_width: np.ndarray
    local_phase_slope: np.ndarray
    boundary_valid: np.ndarray
    open_lambda: float
    open_expected: float
    open_mismatch: float
    open_validation_pass: bool
    full_lambda: float
    growth_reference_lambda: float
    q550_anchor_mismatch: float


@dataclass
class TouchResult:
    eta_kappa: np.ndarray
    q_touch: np.ndarray
    fit_slope: np.ndarray
    fit_intercept: np.ndarray
    fit_r_squared: np.ndarray
    bracket_q_low: np.ndarray
    bracket_q_high: np.ndarray
    valid: np.ndarray


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return expit(z)


def parse_float_list(raw: str) -> np.ndarray:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one comma-separated numeric value.")
    array = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(array)):
        raise ValueError("The numeric list contains a non-finite value.")
    return np.unique(array)


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


def build_initial_physical_state(n: float) -> np.ndarray:
    omega0, rho0, delta0, x0 = 0.05, 1.2, 0.3, 0.5
    return np.array(
        [
            rho0,
            0.0,
            0.0,
            +n * omega0,
            rho0,
            0.0,
            delta0,
            -n * omega0,
            x0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def physical_to_internal(state: Sequence[float]) -> np.ndarray:
    physical = np.asarray(state, dtype=float)
    if physical.shape != (11,):
        raise ValueError("State must contain exactly 11 components.")

    rho1, rho1_dot = physical[0], physical[1]
    rho2, rho2_dot = physical[4], physical[5]
    x, x_dot = physical[8], physical[9]

    if rho1 <= 0.0 or rho2 <= 0.0:
        raise ValueError("rho1 and rho2 must remain strictly positive.")
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


def characteristic_periods(
    initial_physical_state: Sequence[float],
    params: ModelParameters,
) -> tuple[float, float, float]:
    state = np.asarray(initial_physical_state, dtype=float)
    x0 = state[8]

    stiffness_x = params.barrier_B * (
        1.0 / x0**2 + 1.0 / (1.0 - x0) ** 2
    )
    omega_n_sq = stiffness_x / params.m_x
    damping_rate = params.gamma_x / (2.0 * params.m_x)
    omega_d_sq = omega_n_sq - damping_rate**2
    if omega_d_sq <= 0.0:
        raise ValueError("No real damped x period exists.")

    period_x = 2.0 * np.pi / np.sqrt(omega_d_sq)
    delta_rate0 = state[3] + state[7] - x0
    period_delta = (
        period_x
        if abs(delta_rate0) < 1e-10
        else 2.0 * np.pi / abs(delta_rate0)
    )
    return period_x, period_delta, max(period_x, period_delta)


def kappa_from_q(q: float, n: float) -> float:
    if q <= 0.0:
        raise ValueError("Q must be positive.")
    return (55.0 / 6.0) * (n / q)


def derivatives_internal(
    _t: float,
    state: np.ndarray,
    kappa0: float,
    params: ModelParameters,
) -> np.ndarray:
    (
        y1,
        y1_dot,
        theta1,
        theta1_dot,
        y2,
        y2_dot,
        theta2,
        theta2_dot,
        z,
        z_dot,
        Theta,
    ) = state

    rho1 = np.exp(y1)
    rho2 = np.exp(y2)
    x = sigmoid(z)
    g = x * (1.0 - x)
    if g < 1e-14:
        raise FloatingPointError(
            "x moved too close to a boundary; internal dynamics became singular."
        )

    rho1_dot = rho1 * y1_dot
    rho2_dot = rho2 * y2_dot
    delta = theta1 + theta2 - params.phi0 - Theta
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    kappa_eff = kappa0 * x

    rho1_ddot = (
        rho1 * theta1_dot**2
        - dU(rho1, params.e_reserve)
        - kappa_eff * rho2 * cos_delta
        - params.gamma_rho * rho1_dot
    )
    rho2_ddot = (
        rho2 * theta2_dot**2
        - dU(rho2, params.e_reserve)
        - kappa_eff * rho1 * cos_delta
        - params.gamma_rho * rho2_dot
    )

    y1_ddot = rho1_ddot / rho1 - y1_dot**2
    y2_ddot = rho2_ddot / rho2 - y2_dot**2

    theta1_ddot = (
        kappa_eff * rho2 * sin_delta / rho1
        - 2.0 * y1_dot * theta1_dot
    )
    theta2_ddot = (
        kappa_eff * rho1 * sin_delta / rho2
        - 2.0 * y2_dot * theta2_dot
    )

    activity = rho1 * rho2 * sin_delta
    x_dot = g * z_dot
    x_ddot = (
        -dV_barrier(x, params.barrier_B)
        + params.mu_x * activity
        - params.gamma_x * x_dot
    ) / params.m_x

    z_ddot = x_ddot / g - (1.0 - 2.0 * x) * z_dot**2

    derivative = np.array(
        [
            y1_dot,
            y1_ddot,
            theta1_dot,
            theta1_ddot,
            y2_dot,
            y2_ddot,
            theta2_dot,
            theta2_ddot,
            z_dot,
            z_ddot,
            x,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError("Non-finite derivative encountered.")
    return derivative


def integrate_final_state(
    state: np.ndarray,
    *,
    duration: float,
    kappa0: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    solution = solve_ivp(
        derivatives_internal,
        (0.0, duration),
        state,
        args=(kappa0, params),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    final_state = solution.y[:, -1]
    if not np.all(np.isfinite(final_state)):
        raise FloatingPointError("Integration produced a non-finite final state.")
    return final_state


def warm_state_at_q_start(
    initial_state: np.ndarray,
    *,
    q_start: float,
    n: float,
    warmup_periods: int,
    period_reference: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    print(
        f"Warming one common state at Q={q_start:g} "
        f"for {warmup_periods} reference periods..."
    )
    return integrate_final_state(
        initial_state,
        duration=warmup_periods * period_reference,
        kappa0=kappa_from_q(q_start, n),
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )


def ramp_to_q(
    warmed_state: np.ndarray,
    *,
    q_start: float,
    q_target: float,
    n: float,
    ramp_steps: int,
    ramp_periods_per_step: int,
    period_reference: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    state = warmed_state.copy()

    if ramp_steps <= 0:
        return state

    for step in range(ramp_steps):
        fraction = (step + 1) / ramp_steps
        q_now = q_start + fraction * (q_target - q_start)
        state = integrate_final_state(
            state,
            duration=ramp_periods_per_step * period_reference,
            kappa0=kappa_from_q(q_now, n),
            params=params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if (
            step == 0
            or step + 1 == ramp_steps
            or (step + 1) % max(1, ramp_steps // 4) == 0
        ):
            physical = internal_to_physical(state)
            print(
                f"  ramp Q={q_target:g}: [{step + 1:>4}/{ramp_steps}] "
                f"Q_now={q_now:.6f}  x={physical[8]:.9f}"
            )

    return state


def record_fixed_q_orbit(
    initial_state: np.ndarray,
    *,
    q_target: float,
    n: float,
    periods: int,
    samples_per_period: int,
    chunk_periods: int,
    period_reference: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    state = initial_state.copy()
    kappa0 = kappa_from_q(q_target, n)

    times: list[np.ndarray] = []
    states: list[np.ndarray] = []
    completed = 0
    elapsed = 0.0

    while completed < periods:
        current_periods = min(chunk_periods, periods - completed)
        duration = current_periods * period_reference
        sample_count = current_periods * samples_per_period + 1
        local_times = np.linspace(0.0, duration, sample_count)

        solution = solve_ivp(
            derivatives_internal,
            (0.0, duration),
            state,
            args=(kappa0, params),
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=local_times,
        )
        if not solution.success:
            raise RuntimeError(
                f"Q={q_target:g} orbit recording failed after "
                f"{completed} periods: {solution.message}"
            )
        if not np.all(np.isfinite(solution.y)):
            raise FloatingPointError(
                f"Q={q_target:g} orbit recording became non-finite."
            )

        if completed == 0:
            times.append(elapsed + local_times)
            states.append(solution.y)
        else:
            times.append(elapsed + local_times[1:])
            states.append(solution.y[:, 1:])

        state = solution.y[:, -1]
        completed += current_periods
        elapsed += duration

        physical = internal_to_physical(state)
        print(
            f"  record Q={q_target:g}: [{completed:>4}/{periods}] "
            f"x={physical[8]:.9f}  "
            f"rho=({physical[0]:.6f}, {physical[4]:.6f})"
        )

    return np.concatenate(times), np.concatenate(states, axis=1)


def gauge_project_batch(vectors: np.ndarray) -> np.ndarray:
    projected = np.asarray(vectors, dtype=float).copy()
    delta_phase = projected[:, 2] + projected[:, 6] - projected[:, 10]
    projected[:, 2] = delta_phase / 3.0
    projected[:, 6] = delta_phase / 3.0
    projected[:, 10] = -delta_phase / 3.0
    return projected


def weighted_norm_batch(
    vectors: np.ndarray,
    omega_scale: float,
) -> np.ndarray:
    projected = gauge_project_batch(vectors)
    weighted = projected.copy()
    weighted[:, [1, 3, 5, 7, 9]] /= omega_scale
    return np.sqrt(np.sum(weighted**2, axis=1))


def normalized_initial_x_direction(
    central_state: np.ndarray,
    physical_perturbation: float,
    omega_scale: float,
) -> np.ndarray:
    physical = internal_to_physical(central_state)
    x0 = physical[8]

    if not 0.0 < x0 - physical_perturbation < x0 + physical_perturbation < 1.0:
        raise ValueError("Initial x perturbation leaves the interval (0, 1).")

    plus = physical.copy()
    minus = physical.copy()
    plus[8] = x0 + physical_perturbation
    minus[8] = x0 - physical_perturbation

    deviation = 0.5 * (
        physical_to_internal(plus) - physical_to_internal(minus)
    )
    deviation = gauge_project_batch(deviation[np.newaxis, :])[0]
    norm = weighted_norm_batch(
        deviation[np.newaxis, :],
        omega_scale,
    )[0]
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Invalid initial tangent norm.")
    return deviation / norm


def tangent_rhs_batch(
    vectors: np.ndarray,
    central_state: np.ndarray,
    eta_phase: np.ndarray,
    eta_kappa: np.ndarray,
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
    ) = central_state

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

    radial_1 = kappa_eff * ratio21 * cos_delta
    angular_1 = kappa_eff * ratio21 * sin_delta
    radial_2 = kappa_eff * ratio12 * cos_delta
    angular_2 = kappa_eff * ratio12 * sin_delta

    log_derivative_u1 = (
        ddU(rho1, params.e_reserve)
        - dU(rho1, params.e_reserve) / rho1
    )
    log_derivative_u2 = (
        ddU(rho2, params.e_reserve)
        - dU(rho2, params.e_reserve) / rho2
    )

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

    j98 = (
        (
            -ddV_barrier(x, params.barrier_B)
            - params.gamma_x * (1.0 - 2.0 * x) * vz
        ) / params.m_x
        - numerator * (1.0 - 2.0 * x) / g
        + 2.0 * g * vz**2
    )
    j99 = (
        -params.gamma_x / params.m_x
        - 2.0 * (1.0 - 2.0 * x) * vz
    )

    d = vectors
    out = np.empty_like(d)

    out[:, 0] = d[:, 1]

    out[:, 1] = (
        (-log_derivative_u1 + radial_1) * d[:, 0]
        + (-params.gamma_rho - 2.0 * v1) * d[:, 1]
        + angular_1 * d[:, 2]
        + 2.0 * omega1 * d[:, 3]
        - radial_1 * d[:, 4]
        + angular_1 * d[:, 6]
        - angular_1 * d[:, 10]
        + eta_kappa
        * (-kappa0 * g * ratio21 * cos_delta)
        * d[:, 8]
    )

    out[:, 2] = d[:, 3]

    out[:, 3] = (
        -angular_1 * d[:, 0]
        - 2.0 * omega1 * d[:, 1]
        + radial_1 * d[:, 2]
        - 2.0 * v1 * d[:, 3]
        + angular_1 * d[:, 4]
        + radial_1 * d[:, 6]
        - radial_1 * d[:, 10]
        + eta_kappa
        * (kappa0 * g * ratio21 * sin_delta)
        * d[:, 8]
    )

    out[:, 4] = d[:, 5]

    out[:, 5] = (
        -radial_2 * d[:, 0]
        + angular_2 * d[:, 2]
        + (-log_derivative_u2 + radial_2) * d[:, 4]
        + (-params.gamma_rho - 2.0 * v2) * d[:, 5]
        + angular_2 * d[:, 6]
        + 2.0 * omega2 * d[:, 7]
        - angular_2 * d[:, 10]
        + eta_kappa
        * (-kappa0 * g * ratio12 * cos_delta)
        * d[:, 8]
    )

    out[:, 6] = d[:, 7]

    out[:, 7] = (
        angular_2 * d[:, 0]
        + radial_2 * d[:, 2]
        - angular_2 * d[:, 4]
        - 2.0 * omega2 * d[:, 5]
        + radial_2 * d[:, 6]
        - 2.0 * v2 * d[:, 7]
        - radial_2 * d[:, 10]
        + eta_kappa
        * (kappa0 * g * ratio12 * sin_delta)
        * d[:, 8]
    )

    out[:, 8] = d[:, 9]

    out[:, 9] = (
        activity_factor * d[:, 0]
        + phase_factor * d[:, 2]
        + activity_factor * d[:, 4]
        + phase_factor * d[:, 6]
        + j98 * d[:, 8]
        + j99 * d[:, 9]
        - phase_factor * d[:, 10]
    )

    out[:, 10] = eta_phase * g * d[:, 8]

    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite tangent derivative encountered.")
    return out


def block_standard_error_batch(
    retained: np.ndarray,
    block_periods: int,
) -> np.ndarray:
    pair_count, retained_periods = retained.shape
    block_count = retained_periods // block_periods

    if block_count >= 2:
        trimmed = retained[:, : block_count * block_periods]
        blocks = trimmed.reshape(
            pair_count,
            block_count,
            block_periods,
        ).mean(axis=2)
        return np.std(blocks, axis=1, ddof=1) / np.sqrt(block_count)

    if retained_periods >= 2:
        return (
            np.std(retained, axis=1, ddof=1)
            / np.sqrt(retained_periods)
        )
    return np.full(pair_count, np.nan)


def prepare_rk4_orbit_samples(
    times: np.ndarray,
    states: np.ndarray,
    periods: int,
    steps_per_period: int,
    period_reference: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate the orbit on an RK4 lattice without crossing its endpoint.

    Chunked nonlinear integration can leave the recorded final timestamp a few
    floating-point ulps below or above periods * period_reference. Building the
    RK4 lattice from the theoretical duration can therefore request the spline
    just outside its domain. With extrapolate=False that single endpoint
    becomes NaN.

    The recorded interval is the source of truth here. Its duration is first
    checked against the theoretical duration, then divided exactly into the
    requested number of RK4 steps. The last node is explicitly pinned to the
    final recorded timestamp.
    """
    times = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)

    if times.ndim != 1 or times.size < 2:
        raise ValueError("Recorded orbit times must be a one-dimensional array.")
    if states.ndim != 2 or states.shape != (11, times.size):
        raise ValueError(
            "Recorded orbit states must have shape (11, number_of_times)."
        )
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(states)):
        raise FloatingPointError("Recorded orbit already contains non-finite data.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("Recorded orbit times must be strictly increasing.")

    total_steps = periods * steps_per_period
    expected_duration = periods * period_reference
    recorded_start = float(times[0])
    recorded_end = float(times[-1])
    recorded_duration = recorded_end - recorded_start

    if not np.isclose(
        recorded_duration,
        expected_duration,
        rtol=2e-11,
        atol=2e-9,
    ):
        raise ValueError(
            "Recorded orbit duration does not match the requested number of "
            f"periods: recorded={recorded_duration:.17e}, "
            f"expected={expected_duration:.17e}, "
            f"difference={recorded_duration - expected_duration:+.3e}."
        )

    time_step = recorded_duration / total_steps

    spline = CubicSpline(
        times,
        states,
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

    # Final ulp-level guard against an interpolation request outside the
    # recorded interval.
    node_times = np.clip(node_times, recorded_start, recorded_end)
    midpoint_times = np.clip(
        midpoint_times,
        recorded_start,
        recorded_end,
    )

    node_states = np.asarray(
        spline(node_times),
        dtype=float,
    ).T
    midpoint_states = np.asarray(
        spline(midpoint_times),
        dtype=float,
    ).T

    if (
        node_states.shape != (total_steps + 1, 11)
        or midpoint_states.shape != (total_steps, 11)
    ):
        raise RuntimeError("Unexpected interpolated orbit shape.")

    node_finite = np.all(np.isfinite(node_states), axis=1)
    midpoint_finite = np.all(np.isfinite(midpoint_states), axis=1)
    if not np.all(node_finite) or not np.all(midpoint_finite):
        bad_nodes = np.flatnonzero(~node_finite)
        bad_midpoints = np.flatnonzero(~midpoint_finite)
        raise FloatingPointError(
            "Orbit interpolation produced non-finite data after endpoint "
            f"clamping. Bad node indices={bad_nodes[:10].tolist()}, "
            f"bad midpoint indices={bad_midpoints[:10].tolist()}, "
            f"recorded interval=[{recorded_start:.17e}, "
            f"{recorded_end:.17e}]."
        )

    return node_states, midpoint_states, time_step


def integrate_gain_pairs(
    eta_phase: np.ndarray,
    eta_kappa: np.ndarray,
    *,
    node_states: np.ndarray,
    midpoint_states: np.ndarray,
    steps_per_period: int,
    periods: int,
    discard_periods: int,
    block_periods: int,
    initial_direction: np.ndarray,
    omega_scale: float,
    time_step: float,
    kappa0: float,
    params: ModelParameters,
    label: str,
    progress_every: int,
) -> BatchResult:
    eta_phase = np.asarray(eta_phase, dtype=float)
    eta_kappa = np.asarray(eta_kappa, dtype=float)

    if eta_phase.ndim != 1 or eta_kappa.ndim != 1:
        raise ValueError("Gain arrays must be one-dimensional.")
    if eta_phase.size != eta_kappa.size:
        raise ValueError("Gain arrays must have equal length.")
    if eta_phase.size == 0:
        raise ValueError("At least one gain pair is required.")

    pair_count = eta_phase.size
    vectors = np.repeat(
        initial_direction[np.newaxis, :],
        pair_count,
        axis=0,
    )
    local_logs = np.empty((pair_count, periods), dtype=float)

    print(f"  {label}: integrating {pair_count} gain pairs")

    for period in range(periods):
        first_step = period * steps_per_period
        last_step = first_step + steps_per_period

        for step in range(first_step, last_step):
            central_start = node_states[step]
            central_mid = midpoint_states[step]
            central_end = node_states[step + 1]

            k1 = tangent_rhs_batch(
                vectors,
                central_start,
                eta_phase,
                eta_kappa,
                kappa0,
                params,
            )
            k2 = tangent_rhs_batch(
                vectors + 0.5 * time_step * k1,
                central_mid,
                eta_phase,
                eta_kappa,
                kappa0,
                params,
            )
            k3 = tangent_rhs_batch(
                vectors + 0.5 * time_step * k2,
                central_mid,
                eta_phase,
                eta_kappa,
                kappa0,
                params,
            )
            k4 = tangent_rhs_batch(
                vectors + time_step * k3,
                central_end,
                eta_phase,
                eta_kappa,
                kappa0,
                params,
            )

            vectors = vectors + (
                time_step / 6.0
            ) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        vectors = gauge_project_batch(vectors)
        norms = weighted_norm_batch(vectors, omega_scale)

        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            bad = np.flatnonzero((~np.isfinite(norms)) | (norms <= 0.0))
            raise FloatingPointError(
                f"{label}: invalid tangent norm at indices {bad[:10]}."
            )

        local_logs[:, period] = np.log(norms)
        vectors /= norms[:, np.newaxis]

        if (
            period == 0
            or period + 1 == periods
            or (period + 1) % progress_every == 0
        ):
            running = np.mean(local_logs[:, : period + 1], axis=1)
            print(
                f"    [{period + 1:>4}/{periods}] "
                f"lambda range [{np.min(running):+.6e}, "
                f"{np.max(running):+.6e}]"
            )

    retained = local_logs[:, discard_periods:]
    retained_lambda = np.mean(retained, axis=1)
    stderr = block_standard_error_batch(retained, block_periods)

    split = retained.shape[1] // 2
    first_half = np.mean(retained[:, :split], axis=1)
    second_half = np.mean(retained[:, split:], axis=1)

    return BatchResult(
        eta_phase=eta_phase,
        eta_kappa=eta_kappa,
        lambda_retained=retained_lambda,
        block_standard_error=stderr,
        first_half_lambda=first_half,
        second_half_lambda=second_half,
        drift=second_half - first_half,
    )


def read_growth_reference(
    path: Path,
    fit_q_min: float,
    fit_q_max: float,
    fallback_slope: float,
    fallback_qc: float,
) -> GrowthReference:
    if not path.exists():
        return GrowthReference(
            available=False,
            slope=fallback_slope,
            q_critical=fallback_qc,
            r_squared=float("nan"),
            q=np.array([], dtype=float),
            growth_rate=np.array([], dtype=float),
            source="fallback values from prior threshold closure",
        )

    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    q_values = []
    lambda_values = []
    for row in rows:
        try:
            q_values.append(float(row["Q"]))
            lambda_values.append(
                float(row["lambda_per_reference_period"])
            )
        except (KeyError, TypeError, ValueError):
            continue

    q = np.asarray(q_values, dtype=float)
    growth = np.asarray(lambda_values, dtype=float)
    mask = (
        np.isfinite(q)
        & np.isfinite(growth)
        & (q >= fit_q_min)
        & (q <= fit_q_max)
    )

    if np.count_nonzero(mask) < 3:
        return GrowthReference(
            available=False,
            slope=fallback_slope,
            q_critical=fallback_qc,
            r_squared=float("nan"),
            q=q,
            growth_rate=growth,
            source=(
                f"{path.resolve()} was readable but did not contain at least "
                "three points inside the requested local fit interval"
            ),
        )

    slope, intercept = np.polyfit(q[mask], growth[mask], 1)
    prediction = slope * q[mask] + intercept
    residual = float(np.sum((growth[mask] - prediction) ** 2))
    total = float(np.sum((growth[mask] - np.mean(growth[mask])) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")

    return GrowthReference(
        available=True,
        slope=float(slope),
        q_critical=float(-intercept / slope),
        r_squared=float(r_squared),
        q=q,
        growth_rate=growth,
        source=str(path.resolve()),
    )


def growth_reference_value(
    reference: GrowthReference,
    q: float,
) -> float:
    return reference.slope * (q - reference.q_critical)


def adaptive_period_count(
    q: float,
    reference: GrowthReference,
    *,
    base_periods: int,
    max_periods: int,
    target_log_window: float,
    minimum_rate_scale: float,
) -> int:
    estimated_rate = abs(growth_reference_value(reference, q))
    effective_rate = max(estimated_rate, minimum_rate_scale)
    requested = int(math.ceil(target_log_window / effective_rate))
    return int(np.clip(requested, base_periods, max_periods))


def config_digest(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def q_token(q: float) -> str:
    text = f"{q:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def save_orbit_cache(
    path: Path,
    *,
    signature: str,
    q: float,
    periods: int,
    discard_periods: int,
    period_reference: float,
    times: np.ndarray,
    states: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        script_version=SCRIPT_VERSION,
        signature=signature,
        q=q,
        periods=periods,
        discard_periods=discard_periods,
        period_reference=period_reference,
        times=times,
        states=states,
    )


def load_orbit_cache(
    path: Path,
    signature: str,
) -> tuple[int, int, float, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    archive = np.load(path)
    stored_signature = str(np.asarray(archive["signature"]).item())
    if stored_signature != signature:
        return None
    return (
        int(np.asarray(archive["periods"]).item()),
        int(np.asarray(archive["discard_periods"]).item()),
        float(np.asarray(archive["period_reference"]).item()),
        np.asarray(archive["times"], dtype=float),
        np.asarray(archive["states"], dtype=float),
    )


def save_q_result_cache(
    path: Path,
    signature: str,
    result: QScanResult,
) -> None:
    np.savez_compressed(
        path,
        script_version=SCRIPT_VERSION,
        signature=signature,
        q=result.q,
        periods=result.periods,
        discard_periods=result.discard_periods,
        phase_values=result.phase_values,
        kappa_values=result.kappa_values,
        lambda_map=result.lambda_map,
        stderr_map=result.stderr_map,
        drift_map=result.drift_map,
        lambda_phase_zero=result.lambda_phase_zero,
        lambda_phase_one=result.lambda_phase_one,
        eta_phase_critical=result.eta_phase_critical,
        lambda_at_critical=result.lambda_at_critical,
        phase_bracket_width=result.phase_bracket_width,
        local_phase_slope=result.local_phase_slope,
        boundary_valid=result.boundary_valid,
        open_lambda=result.open_lambda,
        open_expected=result.open_expected,
        open_mismatch=result.open_mismatch,
        open_validation_pass=int(result.open_validation_pass),
        full_lambda=result.full_lambda,
        growth_reference_lambda=result.growth_reference_lambda,
        q550_anchor_mismatch=result.q550_anchor_mismatch,
    )


def load_q_result_cache(
    path: Path,
    signature: str,
) -> QScanResult | None:
    if not path.exists():
        return None
    archive = np.load(path)
    stored_signature = str(np.asarray(archive["signature"]).item())
    if stored_signature != signature:
        return None

    return QScanResult(
        q=float(np.asarray(archive["q"]).item()),
        periods=int(np.asarray(archive["periods"]).item()),
        discard_periods=int(
            np.asarray(archive["discard_periods"]).item()
        ),
        phase_values=np.asarray(archive["phase_values"], dtype=float),
        kappa_values=np.asarray(archive["kappa_values"], dtype=float),
        lambda_map=np.asarray(archive["lambda_map"], dtype=float),
        stderr_map=np.asarray(archive["stderr_map"], dtype=float),
        drift_map=np.asarray(archive["drift_map"], dtype=float),
        lambda_phase_zero=np.asarray(
            archive["lambda_phase_zero"], dtype=float
        ),
        lambda_phase_one=np.asarray(
            archive["lambda_phase_one"], dtype=float
        ),
        eta_phase_critical=np.asarray(
            archive["eta_phase_critical"], dtype=float
        ),
        lambda_at_critical=np.asarray(
            archive["lambda_at_critical"], dtype=float
        ),
        phase_bracket_width=np.asarray(
            archive["phase_bracket_width"], dtype=float
        ),
        local_phase_slope=np.asarray(
            archive["local_phase_slope"], dtype=float
        ),
        boundary_valid=np.asarray(
            archive["boundary_valid"], dtype=bool
        ),
        open_lambda=float(np.asarray(archive["open_lambda"]).item()),
        open_expected=float(
            np.asarray(archive["open_expected"]).item()
        ),
        open_mismatch=float(
            np.asarray(archive["open_mismatch"]).item()
        ),
        open_validation_pass=bool(
            int(np.asarray(archive["open_validation_pass"]).item())
        ),
        full_lambda=float(np.asarray(archive["full_lambda"]).item()),
        growth_reference_lambda=float(
            np.asarray(archive["growth_reference_lambda"]).item()
        ),
        q550_anchor_mismatch=float(
            np.asarray(archive["q550_anchor_mismatch"]).item()
        ),
    )


def locate_phase_brackets(
    phase_values: np.ndarray,
    lambda_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kappa_count = lambda_map.shape[1]
    low_phase = np.full(kappa_count, np.nan)
    high_phase = np.full(kappa_count, np.nan)
    low_lambda = np.full(kappa_count, np.nan)
    high_lambda = np.full(kappa_count, np.nan)
    valid = np.zeros(kappa_count, dtype=bool)

    for column in range(kappa_count):
        values = lambda_map[:, column]
        for index in range(phase_values.size - 1):
            left = values[index]
            right = values[index + 1]
            if left <= 0.0 <= right:
                low_phase[column] = phase_values[index]
                high_phase[column] = phase_values[index + 1]
                low_lambda[column] = left
                high_lambda[column] = right
                valid[column] = True
                break

    return low_phase, high_phase, low_lambda, high_lambda, valid


def refine_phase_boundary(
    kappa_values: np.ndarray,
    low_phase: np.ndarray,
    high_phase: np.ndarray,
    low_lambda: np.ndarray,
    high_lambda: np.ndarray,
    valid: np.ndarray,
    *,
    bisection_iterations: int,
    integration_kwargs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.flatnonzero(valid)
    critical = np.full_like(kappa_values, np.nan)
    residual = np.full_like(kappa_values, np.nan)
    width = np.full_like(kappa_values, np.nan)
    slope = np.full_like(kappa_values, np.nan)

    if active.size == 0:
        return critical, residual, width, slope

    low_p = low_phase.copy()
    high_p = high_phase.copy()
    low_l = low_lambda.copy()
    high_l = high_lambda.copy()

    for iteration in range(bisection_iterations):
        midpoint = 0.5 * (low_p[active] + high_p[active])
        batch = integrate_gain_pairs(
            midpoint,
            kappa_values[active],
            label=f"phase bisection {iteration + 1}/{bisection_iterations}",
            **integration_kwargs,
        )
        midpoint_lambda = batch.lambda_retained

        negative = midpoint_lambda <= 0.0
        negative_indices = active[negative]
        positive_indices = active[~negative]

        low_p[negative_indices] = midpoint[negative]
        low_l[negative_indices] = midpoint_lambda[negative]
        high_p[positive_indices] = midpoint[~negative]
        high_l[positive_indices] = midpoint_lambda[~negative]

    critical[active] = 0.5 * (low_p[active] + high_p[active])
    width[active] = high_p[active] - low_p[active]
    slope[active] = (
        high_l[active] - low_l[active]
    ) / width[active]

    final_batch = integrate_gain_pairs(
        critical[active],
        kappa_values[active],
        label="phase-boundary final residual",
        **integration_kwargs,
    )
    residual[active] = final_batch.lambda_retained

    return critical, residual, width, slope


def scan_one_q(
    *,
    q: float,
    warmed_state: np.ndarray,
    n: float,
    q_start: float,
    ramp_steps: int,
    ramp_periods_per_step: int,
    periods: int,
    discard_periods: int,
    period_reference: float,
    orbit_samples_per_period: int,
    orbit_chunk_periods: int,
    tangent_steps_per_period: int,
    phase_values: np.ndarray,
    kappa_values: np.ndarray,
    bisection_iterations: int,
    initial_x_perturbation: float,
    params: ModelParameters,
    nonlinear_method: str,
    nonlinear_rtol: float,
    nonlinear_atol: float,
    nonlinear_max_step: float,
    block_periods: int,
    progress_every: int,
    open_tolerance: float,
    growth_reference: GrowthReference,
    q550_anchor: float,
    q550_anchor_tolerance: float,
    orbit_cache_path: Path,
    orbit_signature: str,
    resume: bool,
) -> QScanResult:
    print()
    print("=" * 78)
    print(
        f"Q={q:g}: periods={periods}, discard={discard_periods}, "
        f"kappa0={kappa_from_q(q, n):.12e}"
    )

    cached_orbit = (
        load_orbit_cache(orbit_cache_path, orbit_signature)
        if resume
        else None
    )

    if cached_orbit is not None:
        (
            cached_periods,
            cached_discard,
            cached_reference,
            times,
            states,
        ) = cached_orbit
        if (
            cached_periods != periods
            or cached_discard != discard_periods
            or not np.isclose(
                cached_reference,
                period_reference,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            cached_orbit = None
        else:
            print(f"  loaded cached orbit: {orbit_cache_path}")

    if cached_orbit is None:
        hold_start = ramp_to_q(
            warmed_state,
            q_start=q_start,
            q_target=q,
            n=n,
            ramp_steps=ramp_steps,
            ramp_periods_per_step=ramp_periods_per_step,
            period_reference=period_reference,
            params=params,
            method=nonlinear_method,
            rtol=nonlinear_rtol,
            atol=nonlinear_atol,
            max_step=nonlinear_max_step,
        )
        times, states = record_fixed_q_orbit(
            hold_start,
            q_target=q,
            n=n,
            periods=periods,
            samples_per_period=orbit_samples_per_period,
            chunk_periods=orbit_chunk_periods,
            period_reference=period_reference,
            params=params,
            method=nonlinear_method,
            rtol=nonlinear_rtol,
            atol=nonlinear_atol,
            max_step=nonlinear_max_step,
        )
        save_orbit_cache(
            orbit_cache_path,
            signature=orbit_signature,
            q=q,
            periods=periods,
            discard_periods=discard_periods,
            period_reference=period_reference,
            times=times,
            states=states,
        )
        print(f"  saved orbit cache: {orbit_cache_path}")

    initial_physical = internal_to_physical(states[:, 0])
    initial_x = float(initial_physical[8])
    omega_scale = float(
        np.sqrt(
            ddV_barrier(initial_x, params.barrier_B)
            / params.m_x
        )
    )
    initial_direction = normalized_initial_x_direction(
        states[:, 0],
        initial_x_perturbation,
        omega_scale,
    )

    print("  interpolating orbit onto the tangent RK4 lattice...")
    node_states, midpoint_states, time_step = prepare_rk4_orbit_samples(
        times,
        states,
        periods,
        tangent_steps_per_period,
        period_reference,
    )

    phase_mesh, kappa_mesh = np.meshgrid(
        phase_values,
        kappa_values,
        indexing="ij",
    )

    integration_kwargs: dict[str, object] = {
        "node_states": node_states,
        "midpoint_states": midpoint_states,
        "steps_per_period": tangent_steps_per_period,
        "periods": periods,
        "discard_periods": discard_periods,
        "block_periods": block_periods,
        "initial_direction": initial_direction,
        "omega_scale": omega_scale,
        "time_step": time_step,
        "kappa0": kappa_from_q(q, n),
        "params": params,
        "progress_every": progress_every,
    }

    coarse = integrate_gain_pairs(
        phase_mesh.ravel(),
        kappa_mesh.ravel(),
        label="coarse curtain section",
        **integration_kwargs,
    )

    lambda_map = coarse.lambda_retained.reshape(
        phase_values.size,
        kappa_values.size,
    )
    stderr_map = coarse.block_standard_error.reshape(
        phase_values.size,
        kappa_values.size,
    )
    drift_map = coarse.drift.reshape(
        phase_values.size,
        kappa_values.size,
    )

    (
        low_phase,
        high_phase,
        low_lambda,
        high_lambda,
        boundary_valid,
    ) = locate_phase_brackets(
        phase_values,
        lambda_map,
    )

    (
        critical,
        critical_residual,
        bracket_width,
        local_slope,
    ) = refine_phase_boundary(
        kappa_values,
        low_phase,
        high_phase,
        low_lambda,
        high_lambda,
        boundary_valid,
        bisection_iterations=bisection_iterations,
        integration_kwargs=integration_kwargs,
    )

    open_expected = (
        -params.gamma_x
        * period_reference
        / (2.0 * params.m_x)
    )
    open_lambda = float(lambda_map[0, 0])
    open_mismatch = abs(open_lambda - open_expected)
    open_pass = open_mismatch <= open_tolerance

    full_lambda = float(lambda_map[-1, -1])
    reference_lambda = growth_reference_value(growth_reference, q)
    q550_mismatch = (
        abs(full_lambda - q550_anchor)
        if np.isclose(q, 550.0, rtol=0.0, atol=1e-9)
        else float("nan")
    )

    print()
    print(f"  Q={q:g} OPEN measured  = {open_lambda:+.12e}")
    print(f"  Q={q:g} OPEN expected  = {open_expected:+.12e}")
    print(f"  Q={q:g} OPEN mismatch  = {open_mismatch:.3e}")
    print(f"  Q={q:g} OPEN valid     = {open_pass}")
    print(f"  Q={q:g} FULL lambda    = {full_lambda:+.12e}")
    print(f"  Q={q:g} growth ref     = {reference_lambda:+.12e}")
    if np.isfinite(q550_mismatch):
        print(f"  Q=550 anchor mismatch  = {q550_mismatch:.3e}")
        if q550_mismatch > q550_anchor_tolerance:
            print(
                "  WARNING: Q=550 FULL anchor mismatch exceeds the requested "
                "tolerance."
            )

    return QScanResult(
        q=q,
        periods=periods,
        discard_periods=discard_periods,
        phase_values=phase_values.copy(),
        kappa_values=kappa_values.copy(),
        lambda_map=lambda_map,
        stderr_map=stderr_map,
        drift_map=drift_map,
        lambda_phase_zero=lambda_map[0].copy(),
        lambda_phase_one=lambda_map[-1].copy(),
        eta_phase_critical=critical,
        lambda_at_critical=critical_residual,
        phase_bracket_width=bracket_width,
        local_phase_slope=local_slope,
        boundary_valid=boundary_valid,
        open_lambda=open_lambda,
        open_expected=open_expected,
        open_mismatch=open_mismatch,
        open_validation_pass=open_pass,
        full_lambda=full_lambda,
        growth_reference_lambda=reference_lambda,
        q550_anchor_mismatch=q550_mismatch,
    )


def fit_touch_curve(
    q_values: np.ndarray,
    lambda_top: np.ndarray,
    kappa_values: np.ndarray,
    fit_points: int,
) -> TouchResult:
    kappa_count = kappa_values.size
    q_touch = np.full(kappa_count, np.nan)
    fit_slope = np.full(kappa_count, np.nan)
    fit_intercept = np.full(kappa_count, np.nan)
    fit_r_squared = np.full(kappa_count, np.nan)
    bracket_low = np.full(kappa_count, np.nan)
    bracket_high = np.full(kappa_count, np.nan)
    valid = np.zeros(kappa_count, dtype=bool)

    for column in range(kappa_count):
        values = lambda_top[:, column]
        crossing_index = None

        for index in range(q_values.size - 1):
            if values[index] <= 0.0 <= values[index + 1]:
                crossing_index = index
                break

        if crossing_index is None:
            continue

        bracket_low[column] = q_values[crossing_index]
        bracket_high[column] = q_values[crossing_index + 1]

        start = max(0, crossing_index - max(0, fit_points // 2 - 1))
        stop = min(q_values.size, start + fit_points)
        start = max(0, stop - fit_points)

        q_fit = q_values[start:stop]
        lambda_fit = values[start:stop]
        finite = np.isfinite(q_fit) & np.isfinite(lambda_fit)
        q_fit = q_fit[finite]
        lambda_fit = lambda_fit[finite]

        if q_fit.size < 2:
            continue

        slope, intercept = np.polyfit(q_fit, lambda_fit, 1)
        if slope <= 0.0:
            continue

        prediction = slope * q_fit + intercept
        residual = float(np.sum((lambda_fit - prediction) ** 2))
        total = float(
            np.sum((lambda_fit - np.mean(lambda_fit)) ** 2)
        )
        r_squared = (
            1.0 - residual / total
            if total > 0.0
            else float("nan")
        )

        root = -intercept / slope
        if not (
            bracket_low[column] - 1.0
            <= root
            <= bracket_high[column] + 1.0
        ):
            # Fall back to direct interpolation inside the sign bracket.
            q0 = q_values[crossing_index]
            q1 = q_values[crossing_index + 1]
            l0 = values[crossing_index]
            l1 = values[crossing_index + 1]
            root = q0 - l0 * (q1 - q0) / (l1 - l0)

        q_touch[column] = root
        fit_slope[column] = slope
        fit_intercept[column] = intercept
        fit_r_squared[column] = r_squared
        valid[column] = True

    return TouchResult(
        eta_kappa=kappa_values.copy(),
        q_touch=q_touch,
        fit_slope=fit_slope,
        fit_intercept=fit_intercept,
        fit_r_squared=fit_r_squared,
        bracket_q_low=bracket_low,
        bracket_q_high=bracket_high,
        valid=valid,
    )


def save_points_csv(
    output_path: Path,
    results: list[QScanResult],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "periods",
                "discard_periods",
                "eta_kappa",
                "lambda_eta_phase_0",
                "lambda_eta_phase_1",
                "eta_phase_critical",
                "phase_margin_1_minus_critical",
                "lambda_at_critical",
                "phase_bracket_width",
                "local_dlambda_deta_phase",
                "boundary_valid",
                "open_validation_pass",
            ]
        )

        for result in results:
            for index, eta_kappa in enumerate(result.kappa_values):
                critical = result.eta_phase_critical[index]
                margin = (
                    1.0 - critical
                    if np.isfinite(critical)
                    else np.nan
                )
                writer.writerow(
                    [
                        result.q,
                        result.periods,
                        result.discard_periods,
                        eta_kappa,
                        result.lambda_phase_zero[index],
                        result.lambda_phase_one[index],
                        critical,
                        margin,
                        result.lambda_at_critical[index],
                        result.phase_bracket_width[index],
                        result.local_phase_slope[index],
                        int(result.boundary_valid[index]),
                        int(result.open_validation_pass),
                    ]
                )


def save_q_summary_csv(
    output_path: Path,
    results: list[QScanResult],
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "periods",
                "discard_periods",
                "open_lambda",
                "open_expected",
                "open_mismatch",
                "open_validation_pass",
                "full_lambda_eta_phase_1_eta_kappa_1",
                "growth_reference_lambda",
                "full_minus_growth_reference",
                "critical_eta_phase_at_eta_kappa_1",
                "phase_margin_at_eta_kappa_1",
                "q550_anchor_mismatch",
            ]
        )

        for result in results:
            critical = result.eta_phase_critical[-1]
            margin = (
                1.0 - critical
                if np.isfinite(critical)
                else np.nan
            )
            writer.writerow(
                [
                    result.q,
                    result.periods,
                    result.discard_periods,
                    result.open_lambda,
                    result.open_expected,
                    result.open_mismatch,
                    int(result.open_validation_pass),
                    result.full_lambda,
                    result.growth_reference_lambda,
                    result.full_lambda - result.growth_reference_lambda,
                    critical,
                    margin,
                    result.q550_anchor_mismatch,
                ]
            )


def save_touch_csv(
    output_path: Path,
    touch: TouchResult,
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eta_kappa",
                "Q_touch_at_eta_phase_1",
                "local_dlambda_dQ",
                "fit_intercept",
                "fit_R_squared",
                "sign_bracket_Q_low",
                "sign_bracket_Q_high",
                "valid",
            ]
        )
        for index in range(touch.eta_kappa.size):
            writer.writerow(
                [
                    touch.eta_kappa[index],
                    touch.q_touch[index],
                    touch.fit_slope[index],
                    touch.fit_intercept[index],
                    touch.fit_r_squared[index],
                    touch.bracket_q_low[index],
                    touch.bracket_q_high[index],
                    int(touch.valid[index]),
                ]
            )


def save_raw_npz(
    output_path: Path,
    q_values: np.ndarray,
    kappa_values: np.ndarray,
    critical_phase: np.ndarray,
    lambda_top: np.ndarray,
    lambda_bottom: np.ndarray,
    critical_residual: np.ndarray,
    bracket_width: np.ndarray,
    local_phase_slope: np.ndarray,
    open_lambda: np.ndarray,
    full_lambda: np.ndarray,
    touch: TouchResult,
    growth_reference: GrowthReference,
) -> None:
    np.savez_compressed(
        output_path,
        script_version=SCRIPT_VERSION,
        q_values=q_values,
        eta_kappa_values=kappa_values,
        eta_phase_critical=critical_phase,
        lambda_eta_phase_1=lambda_top,
        lambda_eta_phase_0=lambda_bottom,
        lambda_at_critical=critical_residual,
        phase_bracket_width=bracket_width,
        local_phase_slope=local_phase_slope,
        open_lambda=open_lambda,
        full_lambda=full_lambda,
        touch_eta_kappa=touch.eta_kappa,
        touch_q=touch.q_touch,
        touch_fit_slope=touch.fit_slope,
        touch_fit_r_squared=touch.fit_r_squared,
        touch_valid=touch.valid,
        growth_reference_slope=growth_reference.slope,
        growth_reference_qc=growth_reference.q_critical,
        growth_reference_r_squared=growth_reference.r_squared,
    )


def plot_critical_phase_map(
    output_path: Path,
    q_values: np.ndarray,
    kappa_values: np.ndarray,
    critical_phase: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    mesh = axis.pcolormesh(
        kappa_values,
        q_values,
        critical_phase,
        shading="auto",
    )
    figure.colorbar(
        mesh,
        ax=axis,
        label="critical eta_phase",
    )
    axis.set_title("Moving stability curtain: critical phase feedback")
    axis.set_xlabel("eta_kappa")
    axis.set_ylabel("Q")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_critical_phase_vs_q(
    output_path: Path,
    q_values: np.ndarray,
    kappa_values: np.ndarray,
    critical_phase: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    for column, eta_kappa in enumerate(kappa_values):
        axis.plot(
            q_values,
            critical_phase[:, column],
            marker="o",
            label=f"eta_kappa={eta_kappa:.3f}",
        )
    axis.axhline(1.0, linewidth=1.0)
    axis.set_title("Critical phase-return gain versus Q")
    axis.set_xlabel("Q")
    axis.set_ylabel("eta_phase critical")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_top_edge_lambda(
    output_path: Path,
    q_values: np.ndarray,
    kappa_values: np.ndarray,
    lambda_top: np.ndarray,
    touch: TouchResult,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    for column, eta_kappa in enumerate(kappa_values):
        axis.plot(
            q_values,
            lambda_top[:, column],
            marker="o",
            label=f"eta_kappa={eta_kappa:.3f}",
        )
        if touch.valid[column]:
            axis.axvline(
                touch.q_touch[column],
                linewidth=0.8,
                alpha=0.4,
            )
    axis.axhline(0.0, linewidth=1.0)
    axis.set_title("Top edge of the curtain: eta_phase = 1")
    axis.set_xlabel("Q")
    axis.set_ylabel("lambda per reference period")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_touch_curve(
    output_path: Path,
    touch: TouchResult,
    growth_reference: GrowthReference,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    valid = touch.valid
    axis.plot(
        touch.eta_kappa[valid],
        touch.q_touch[valid],
        marker="o",
        label="Q_touch(eta_kappa)",
    )
    axis.axhline(
        growth_reference.q_critical,
        linewidth=1.0,
        label="direct local-growth Qc",
    )
    axis.set_title("Where eta_phase = 1 first pierces the curtain")
    axis.set_xlabel("eta_kappa")
    axis.set_ylabel("Q_touch")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def build_report(
    *,
    args: argparse.Namespace,
    growth_reference: GrowthReference,
    results: list[QScanResult],
    touch: TouchResult,
    q_values: np.ndarray,
    kappa_values: np.ndarray,
    critical_phase: np.ndarray,
    lambda_top: np.ndarray,
    output_paths: dict[str, Path],
    cache_dir: Path,
) -> str:
    open_pass_count = sum(
        int(result.open_validation_pass)
        for result in results
    )
    valid_touch = touch.valid
    physical_index = int(np.argmin(np.abs(kappa_values - 1.0)))
    physical_touch = touch.q_touch[physical_index]
    physical_touch_valid = bool(touch.valid[physical_index])

    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "CONFIGURATION",
        "-------------",
        f"Q values                         = {', '.join(f'{q:g}' for q in q_values)}",
        f"eta_kappa values                 = {', '.join(f'{k:g}' for k in kappa_values)}",
        f"phase minimum                    = {args.phase_min:.12f}",
        f"coarse phase points above minimum = {args.phase_points}",
        f"phase bisection iterations       = {args.bisection_iterations}",
        f"warm-up periods                  = {args.warmup_periods}",
        f"ramp steps                       = {args.ramp_steps}",
        f"ramp periods per step            = {args.ramp_periods_per_step}",
        f"base recorded periods            = {args.base_periods}",
        f"maximum recorded periods         = {args.max_periods}",
        f"orbit samples per period         = {args.orbit_samples_per_period}",
        f"tangent RK4 steps per period     = {args.tangent_steps_per_period}",
        f"cache directory                  = {cache_dir.resolve()}",
        "",
        "LOCAL GROWTH REFERENCE",
        "----------------------",
        f"source                           = {growth_reference.source}",
        f"lambda(Q) slope                  = {growth_reference.slope:.12e}",
        f"direct Qc                        = {growth_reference.q_critical:.12f}",
        f"local fit R^2                    = {growth_reference.r_squared:.9f}",
        "",
        "PER-Q VALIDATION",
        "----------------",
        f"OPEN validations passed          = {open_pass_count}/{len(results)}",
    ]

    for result in results:
        lines.extend(
            [
                f"Q={result.q:g}",
                f"  periods / discard              = {result.periods} / {result.discard_periods}",
                f"  OPEN lambda                    = {result.open_lambda:+.12e}",
                f"  OPEN mismatch                  = {result.open_mismatch:.12e}",
                f"  OPEN passed                    = {result.open_validation_pass}",
                f"  FULL lambda                    = {result.full_lambda:+.12e}",
                f"  growth-reference lambda        = {result.growth_reference_lambda:+.12e}",
                f"  eta_phase_c at eta_kappa=1     = {result.eta_phase_critical[-1]:.12e}",
            ]
        )

    lines.extend(
        [
            "",
            "TOP-EDGE TOUCH CURVE",
            "--------------------",
        ]
    )

    for index, eta_kappa in enumerate(kappa_values):
        lines.extend(
            [
                f"eta_kappa={eta_kappa:.6f}",
                f"  valid                           = {bool(touch.valid[index])}",
                f"  Q_touch                         = {touch.q_touch[index]:.12f}",
                f"  local d lambda / dQ             = {touch.fit_slope[index]:.12e}",
                f"  fit R^2                         = {touch.fit_r_squared[index]:.9f}",
                f"  sign bracket                    = [{touch.bracket_q_low[index]:.9f}, {touch.bracket_q_high[index]:.9f}]",
            ]
        )

    lines.extend(
        [
            "",
            "PHYSICAL PERFORATION",
            "--------------------",
            f"physical eta_kappa index          = {kappa_values[physical_index]:.12f}",
            f"physical Q_touch valid            = {physical_touch_valid}",
            f"physical Q_touch                  = {physical_touch:.12f}",
            f"direct growth Qc                  = {growth_reference.q_critical:.12f}",
            f"touch minus direct Qc             = {physical_touch - growth_reference.q_critical:+.12e}",
        ]
    )

    valid_critical = np.isfinite(critical_phase)
    if np.any(valid_critical):
        lines.extend(
            [
                "",
                "CURTAIN RANGE",
                "-------------",
                f"minimum resolved eta_phase_c      = {np.nanmin(critical_phase):.12f}",
                f"maximum resolved eta_phase_c      = {np.nanmax(critical_phase):.12f}",
                f"maximum top-edge lambda           = {np.nanmax(lambda_top):+.12e}",
                f"minimum top-edge lambda           = {np.nanmin(lambda_top):+.12e}",
            ]
        )

    all_open_pass = open_pass_count == len(results)
    if not all_open_pass:
        primary = (
            "INVALID GLOBAL INTERPRETATION: one or more Q slices failed the "
            "mandatory OPEN damping validation. Inspect those cached slices "
            "before reading the touch curve."
        )
    elif physical_touch_valid:
        primary = (
            "The physical corner eta_phase=1, eta_kappa=1 pierces the moving "
            "stability curtain at the reported physical Q_touch. Compare its "
            "difference from the independently measured local-growth Qc."
        )
    else:
        primary = (
            "The sampled Q range did not bracket the physical top-edge sign "
            "change. Extend --q-values below and/or above the current range."
        )

    lines.extend(
        [
            "",
            "PRIMARY READING",
            "---------------",
            primary,
            "",
            "CAUTION",
            "-------",
            "Every Q slice uses a separately prepared LOW-history central orbit.",
            "The eta gains scale tangent return paths while leaving that central",
            "orbit fixed. Q_touch is therefore a local loop-stability boundary,",
            "not a replacement for a global nonlinear bifurcation continuation.",
            "",
            f"critical phase map PNG   = {output_paths['critical_map'].resolve()}",
            f"critical curves PNG      = {output_paths['critical_curves'].resolve()}",
            f"top-edge lambda PNG      = {output_paths['top_edge'].resolve()}",
            f"touch curve PNG          = {output_paths['touch_curve'].resolve()}",
            f"point CSV                = {output_paths['points_csv'].resolve()}",
            f"Q summary CSV            = {output_paths['q_summary_csv'].resolve()}",
            f"touch CSV                = {output_paths['touch_csv'].resolve()}",
            f"raw NPZ                  = {output_paths['raw'].resolve()}",
        ]
    )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track eta_phase_critical(eta_kappa, Q) and locate the physical "
            "Q where eta_phase=eta_kappa=1 first pierces the curtain."
        )
    )
    parser.add_argument(
        "--q-values",
        default="520,521,522,522.25,522.5,523,524,525,530,540,550",
    )
    parser.add_argument(
        "--kappa-values",
        default="0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--n", type=float, default=10.0)
    parser.add_argument("--q-start", type=float, default=250.0)
    parser.add_argument("--warmup-periods", type=int, default=30)
    parser.add_argument("--ramp-steps", type=int, default=240)
    parser.add_argument("--ramp-periods-per-step", type=int, default=1)

    parser.add_argument("--base-periods", type=int, default=400)
    parser.add_argument("--max-periods", type=int, default=1200)
    parser.add_argument(
        "--target-log-window",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--minimum-rate-scale",
        type=float,
        default=2.0e-3,
    )
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.125,
    )
    parser.add_argument("--minimum-discard-periods", type=int, default=50)

    parser.add_argument("--orbit-samples-per-period", type=int, default=160)
    parser.add_argument("--orbit-chunk-periods", type=int, default=20)
    parser.add_argument("--tangent-steps-per-period", type=int, default=80)

    parser.add_argument("--phase-min", type=float, default=0.80)
    parser.add_argument("--phase-points", type=int, default=9)
    parser.add_argument("--bisection-iterations", type=int, default=8)
    parser.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    parser.add_argument("--block-periods", type=int, default=20)
    parser.add_argument("--touch-fit-points", type=int, default=4)

    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument(
        "--nonlinear-method",
        choices=("LSODA", "DOP853", "RK45", "Radau", "BDF"),
        default="LSODA",
    )
    parser.add_argument("--nonlinear-rtol", type=float, default=1e-8)
    parser.add_argument("--nonlinear-atol", type=float, default=1e-10)
    parser.add_argument("--progress-every", type=int, default=200)

    parser.add_argument("--open-tolerance", type=float, default=5e-3)
    parser.add_argument(
        "--q550-full-anchor",
        type=float,
        default=6.181778686724e-3,
    )
    parser.add_argument(
        "--q550-anchor-tolerance",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--growth-csv",
        default="q_growth_threshold_refined.csv",
    )
    parser.add_argument("--growth-fit-q-min", type=float, default=521.0)
    parser.add_argument("--growth-fit-q-max", type=float, default=524.0)
    parser.add_argument(
        "--fallback-growth-slope",
        type=float,
        default=2.514457591518e-4,
    )
    parser.add_argument(
        "--fallback-growth-qc",
        type=float,
        default=522.039048925052,
    )

    parser.add_argument(
        "--cache-dir",
        default="support_feedback_q_curtain_cache",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore compatible cached Q slices and recompute them",
    )
    parser.add_argument(
        "--output-prefix",
        default="support_feedback_curtain_vs_q",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    q_values = parse_float_list(args.q_values)
    kappa_values = parse_float_list(args.kappa_values)

    if np.any(q_values <= 0.0):
        raise ValueError("All Q values must be positive.")
    if np.any((kappa_values < 0.0) | (kappa_values > 1.0)):
        raise ValueError("All eta_kappa values must lie in [0, 1].")
    if not (
        np.isclose(kappa_values[0], 0.0)
        and np.isclose(kappa_values[-1], 1.0)
    ):
        raise ValueError("--kappa-values must include both 0 and 1.")
    if not 0.0 <= args.phase_min < 1.0:
        raise ValueError("--phase-min must lie in [0, 1).")
    if args.phase_points < 3:
        raise ValueError("--phase-points must be at least 3.")
    if args.bisection_iterations < 1:
        raise ValueError("--bisection-iterations must be positive.")
    if args.base_periods < 100:
        raise ValueError("--base-periods must be at least 100.")
    if args.max_periods < args.base_periods:
        raise ValueError("--max-periods must be >= --base-periods.")
    if not 0.0 <= args.discard_fraction < 0.5:
        raise ValueError("--discard-fraction must lie in [0, 0.5).")
    if args.orbit_samples_per_period < 40:
        raise ValueError("--orbit-samples-per-period must be at least 40.")
    if args.tangent_steps_per_period < 20:
        raise ValueError("--tangent-steps-per-period must be at least 20.")
    if args.touch_fit_points < 2:
        raise ValueError("--touch-fit-points must be at least 2.")

    phase_values = np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                np.linspace(
                    args.phase_min,
                    1.0,
                    args.phase_points,
                ),
            )
        )
    )

    params = ModelParameters(gamma_rho=args.gamma_rho)
    initial_physical = build_initial_physical_state(args.n)
    initial_internal = physical_to_internal(initial_physical)
    period_x, period_delta, period_reference = characteristic_periods(
        initial_physical,
        params,
    )
    nonlinear_max_step = min(period_x, period_delta) / 40.0

    growth_reference = read_growth_reference(
        Path(args.growth_csv),
        args.growth_fit_q_min,
        args.growth_fit_q_max,
        args.fallback_growth_slope,
        args.fallback_growth_qc,
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    common_orbit_config = {
        "script_version": ORBIT_CACHE_COMPAT_VERSION,
        "n": args.n,
        "q_start": args.q_start,
        "warmup_periods": args.warmup_periods,
        "ramp_steps": args.ramp_steps,
        "ramp_periods_per_step": args.ramp_periods_per_step,
        "orbit_samples_per_period": args.orbit_samples_per_period,
        "orbit_chunk_periods": args.orbit_chunk_periods,
        "gamma_rho": args.gamma_rho,
        "nonlinear_method": args.nonlinear_method,
        "nonlinear_rtol": args.nonlinear_rtol,
        "nonlinear_atol": args.nonlinear_atol,
        "period_reference": period_reference,
    }

    result_config = {
        **common_orbit_config,
        "tangent_steps_per_period": args.tangent_steps_per_period,
        "phase_values": phase_values.tolist(),
        "kappa_values": kappa_values.tolist(),
        "bisection_iterations": args.bisection_iterations,
        "initial_x_perturbation": args.initial_x_perturbation,
        "block_periods": args.block_periods,
        "open_tolerance": args.open_tolerance,
    }

    print(f"Script version               = {SCRIPT_VERSION}")
    print(f"Q values                     = {q_values}")
    print(f"eta_kappa values             = {kappa_values}")
    print(f"phase values                 = {phase_values}")
    print(f"reference period             = {period_reference:.12f}")
    print(f"growth reference source      = {growth_reference.source}")
    print(f"growth slope                 = {growth_reference.slope:.12e}")
    print(f"growth Qc                    = {growth_reference.q_critical:.12f}")
    print(f"cache directory              = {cache_dir.resolve()}")
    print(f"resume                       = {resume}")
    print()

    warmed_state = warm_state_at_q_start(
        initial_internal,
        q_start=args.q_start,
        n=args.n,
        warmup_periods=args.warmup_periods,
        period_reference=period_reference,
        params=params,
        method=args.nonlinear_method,
        rtol=args.nonlinear_rtol,
        atol=args.nonlinear_atol,
        max_step=nonlinear_max_step,
    )

    results: list[QScanResult] = []

    for q in q_values:
        periods = adaptive_period_count(
            q,
            growth_reference,
            base_periods=args.base_periods,
            max_periods=args.max_periods,
            target_log_window=args.target_log_window,
            minimum_rate_scale=args.minimum_rate_scale,
        )
        discard_periods = max(
            args.minimum_discard_periods,
            int(round(args.discard_fraction * periods)),
        )
        if discard_periods >= periods - 50:
            raise ValueError(
                f"Q={q:g}: discard settings leave too few retained periods."
            )

        orbit_signature = config_digest(
            {
                **common_orbit_config,
                "q": float(q),
                "periods": periods,
                "discard_periods": discard_periods,
            }
        )
        result_signature = config_digest(
            {
                **result_config,
                "q": float(q),
                "periods": periods,
                "discard_periods": discard_periods,
            }
        )

        token = q_token(float(q))
        orbit_cache_path = cache_dir / f"q_{token}_orbit.npz"
        result_cache_path = cache_dir / f"q_{token}_result.npz"

        cached_result = (
            load_q_result_cache(result_cache_path, result_signature)
            if resume
            else None
        )
        if cached_result is not None:
            print()
            print(
                f"Q={q:g}: loaded completed slice from "
                f"{result_cache_path}"
            )
            results.append(cached_result)
            continue

        result = scan_one_q(
            q=float(q),
            warmed_state=warmed_state,
            n=args.n,
            q_start=args.q_start,
            ramp_steps=args.ramp_steps,
            ramp_periods_per_step=args.ramp_periods_per_step,
            periods=periods,
            discard_periods=discard_periods,
            period_reference=period_reference,
            orbit_samples_per_period=args.orbit_samples_per_period,
            orbit_chunk_periods=args.orbit_chunk_periods,
            tangent_steps_per_period=args.tangent_steps_per_period,
            phase_values=phase_values,
            kappa_values=kappa_values,
            bisection_iterations=args.bisection_iterations,
            initial_x_perturbation=args.initial_x_perturbation,
            params=params,
            nonlinear_method=args.nonlinear_method,
            nonlinear_rtol=args.nonlinear_rtol,
            nonlinear_atol=args.nonlinear_atol,
            nonlinear_max_step=nonlinear_max_step,
            block_periods=args.block_periods,
            progress_every=args.progress_every,
            open_tolerance=args.open_tolerance,
            growth_reference=growth_reference,
            q550_anchor=args.q550_full_anchor,
            q550_anchor_tolerance=args.q550_anchor_tolerance,
            orbit_cache_path=orbit_cache_path,
            orbit_signature=orbit_signature,
            resume=resume,
        )
        save_q_result_cache(
            result_cache_path,
            result_signature,
            result,
        )
        print(f"  saved Q-result cache: {result_cache_path}")
        results.append(result)

    results.sort(key=lambda item: item.q)
    q_values = np.asarray([result.q for result in results], dtype=float)
    kappa_values = results[0].kappa_values.copy()

    for result in results[1:]:
        if not np.array_equal(result.kappa_values, kappa_values):
            raise RuntimeError("Cached Q slices use inconsistent eta_kappa grids.")

    critical_phase = np.vstack(
        [result.eta_phase_critical for result in results]
    )
    lambda_top = np.vstack(
        [result.lambda_phase_one for result in results]
    )
    lambda_bottom = np.vstack(
        [result.lambda_phase_zero for result in results]
    )
    critical_residual = np.vstack(
        [result.lambda_at_critical for result in results]
    )
    bracket_width = np.vstack(
        [result.phase_bracket_width for result in results]
    )
    local_phase_slope = np.vstack(
        [result.local_phase_slope for result in results]
    )
    open_lambda = np.asarray(
        [result.open_lambda for result in results],
        dtype=float,
    )
    full_lambda = np.asarray(
        [result.full_lambda for result in results],
        dtype=float,
    )

    touch = fit_touch_curve(
        q_values,
        lambda_top,
        kappa_values,
        args.touch_fit_points,
    )

    prefix = Path(args.output_prefix)
    output_paths = {
        "critical_map": prefix.with_name(
            prefix.name + "_critical_phase_map.png"
        ),
        "critical_curves": prefix.with_name(
            prefix.name + "_critical_phase_vs_q.png"
        ),
        "top_edge": prefix.with_name(
            prefix.name + "_top_edge_lambda_vs_q.png"
        ),
        "touch_curve": prefix.with_name(
            prefix.name + "_q_touch_vs_kappa.png"
        ),
        "points_csv": prefix.with_name(
            prefix.name + "_points.csv"
        ),
        "q_summary_csv": prefix.with_name(
            prefix.name + "_q_summary.csv"
        ),
        "touch_csv": prefix.with_name(
            prefix.name + "_touch.csv"
        ),
        "raw": prefix.with_name(prefix.name + "_raw.npz"),
        "report": prefix.with_name(prefix.name + "_report.txt"),
    }

    plot_critical_phase_map(
        output_paths["critical_map"],
        q_values,
        kappa_values,
        critical_phase,
    )
    plot_critical_phase_vs_q(
        output_paths["critical_curves"],
        q_values,
        kappa_values,
        critical_phase,
    )
    plot_top_edge_lambda(
        output_paths["top_edge"],
        q_values,
        kappa_values,
        lambda_top,
        touch,
    )
    plot_touch_curve(
        output_paths["touch_curve"],
        touch,
        growth_reference,
    )

    save_points_csv(output_paths["points_csv"], results)
    save_q_summary_csv(output_paths["q_summary_csv"], results)
    save_touch_csv(output_paths["touch_csv"], touch)
    save_raw_npz(
        output_paths["raw"],
        q_values,
        kappa_values,
        critical_phase,
        lambda_top,
        lambda_bottom,
        critical_residual,
        bracket_width,
        local_phase_slope,
        open_lambda,
        full_lambda,
        touch,
        growth_reference,
    )

    report = build_report(
        args=args,
        growth_reference=growth_reference,
        results=results,
        touch=touch,
        q_values=q_values,
        kappa_values=kappa_values,
        critical_phase=critical_phase,
        lambda_top=lambda_top,
        output_paths=output_paths,
        cache_dir=cache_dir,
    )
    output_paths["report"].write_text(report, encoding="utf-8")

    print()
    print(report)

    failed_q = [
        result.q
        for result in results
        if not result.open_validation_pass
    ]
    if failed_q:
        raise RuntimeError(
            "One or more Q slices failed OPEN validation: "
            + ", ".join(f"{q:g}" for q in failed_q)
            + ". Outputs were saved for diagnosis, but the global curtain "
            "must not be interpreted yet."
        )


if __name__ == "__main__":
    main()
