#!/usr/bin/env python3
"""Nonlinear amplitude scaling above the local stability threshold.

This experiment follows the oscillatory branch from high Q down toward the
measured local threshold Q_c and tests the supercritical Hopf-like prediction

    A^2 proportional to Q - Q_c

where A is the asymptotic half peak-to-peak amplitude of x.

For every target Q the script:
1. prepares one common high-amplitude state at q_seed_high;
2. continues that state downward to the selected Q;
3. holds Q fixed in chunks until the per-period amplitude converges, or until
   max_hold_periods is reached;
4. measures the late-time amplitude and phase portrait;
5. fits A^2 versus Q-Q_c and A versus (Q-Q_c)^beta.

The adaptive hold is important near Q_c because critical slowing down makes a
fixed-duration hold unreliable.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp
from scipy.special import expit

ALPHA, BETA, GAMMA = 1.0, 0.8, 0.2
SCRIPT_VERSION = "2026-07-31-hopf-amplitude-scaling-v1"


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
class SaturationResult:
    q_value: float
    converged: bool
    total_periods: int
    period_index: np.ndarray
    period_mean: np.ndarray
    period_std: np.ndarray
    period_amplitude: np.ndarray
    late_amplitude: float
    late_std: float
    late_mean: float
    late_log_slope: float
    late_relative_span: float
    phase_time: np.ndarray
    phase_x: np.ndarray
    phase_xdot: np.ndarray
    final_state: np.ndarray


def dU(rho: float, e_reserve: float = 0.0) -> float:
    return (
        2.0 * ALPHA * rho
        - 3.0 * BETA * rho**2
        + 4.0 * GAMMA * rho**3
        - e_reserve**2 / rho**3
    )


def dV_barrier(x: float, barrier_B: float) -> float:
    return -barrier_B * (1.0 / x - 1.0 / (1.0 - x))


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
        raise ValueError("rho1 and rho2 must be strictly positive.")
    if not 0.0 < x < 1.0:
        raise ValueError("x must be strictly inside (0, 1).")

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

    y1, y1_dot = internal[0], internal[1]
    y2, y2_dot = internal[4], internal[5]
    z, z_dot = internal[8], internal[9]

    rho1 = np.exp(y1)
    rho2 = np.exp(y2)
    x = expit(z)
    g = x * (1.0 - x)

    physical = internal.copy()
    physical[0] = rho1
    physical[1] = rho1 * y1_dot
    physical[4] = rho2
    physical[5] = rho2 * y2_dot
    physical[8] = x
    physical[9] = g * z_dot
    return physical


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
    x = expit(z)
    g = x * (1.0 - x)
    if g < 1e-14:
        raise FloatingPointError(
            "x moved too close to a boundary: the dynamics became numerically singular."
        )

    rho1_dot = rho1 * y1_dot
    rho2_dot = rho2 * y2_dot
    Delta = theta1 + theta2 - params.phi0 - Theta
    cosD, sinD = np.cos(Delta), np.sin(Delta)
    kappa_eff = kappa0 * x

    rho1_ddot = (
        rho1 * theta1_dot**2
        - dU(rho1, params.e_reserve)
        - kappa_eff * rho2 * cosD
        - params.gamma_rho * rho1_dot
    )
    rho2_ddot = (
        rho2 * theta2_dot**2
        - dU(rho2, params.e_reserve)
        - kappa_eff * rho1 * cosD
        - params.gamma_rho * rho2_dot
    )

    y1_ddot = rho1_ddot / rho1 - y1_dot**2
    y2_ddot = rho2_ddot / rho2 - y2_dot**2
    theta1_ddot = kappa_eff * rho2 * sinD / rho1 - 2.0 * y1_dot * theta1_dot
    theta2_ddot = kappa_eff * rho1 * sinD / rho2 - 2.0 * y2_dot * theta2_dot

    activity = rho1 * rho2 * sinD
    x_dot = g * z_dot
    x_ddot = (
        -dV_barrier(x, params.barrier_B)
        + params.mu_x * activity
        - params.gamma_x * x_dot
    ) / params.m_x
    z_ddot = x_ddot / g - (1.0 - 2.0 * x) * z_dot**2

    deriv = np.array(
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
    if not np.all(np.isfinite(deriv)):
        raise FloatingPointError("Non-finite derivative encountered during integration.")
    return deriv


def characteristic_periods(
    initial_physical_state: Sequence[float], params: ModelParameters
) -> tuple[float, float, float]:
    state = np.asarray(initial_physical_state, dtype=float)
    x0 = state[8]
    stiffness_x = params.barrier_B * (1.0 / x0**2 + 1.0 / (1.0 - x0) ** 2)
    omega_n_sq = stiffness_x / params.m_x
    damping_rate = params.gamma_x / (2.0 * params.m_x)
    omega_d_sq = omega_n_sq - damping_rate**2
    if omega_d_sq <= 0.0:
        raise ValueError("No real damped x period exists for these parameters.")

    period_x = 2.0 * np.pi / np.sqrt(omega_d_sq)
    delta_rate0 = state[3] + state[7] - x0
    period_delta = period_x if abs(delta_rate0) < 1e-10 else 2.0 * np.pi / abs(delta_rate0)
    return period_x, period_delta, max(period_x, period_delta)


def integrate_interval(
    state_internal: np.ndarray,
    duration: float,
    kappa0: float,
    params: ModelParameters,
    *,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    sample_times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    solution = solve_ivp(
        derivatives_internal,
        (0.0, duration),
        state_internal,
        args=(kappa0, params),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        t_eval=sample_times,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    if not np.all(np.isfinite(solution.y)):
        raise FloatingPointError("The integration produced non-finite values.")
    return solution.y[:, -1], solution.y if sample_times is not None else None


def advance_at_q(
    state: np.ndarray,
    q_value: float,
    duration: float,
    *,
    n: float,
    coupling_constant: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
) -> np.ndarray:
    kappa0 = coupling_constant * (n / q_value)
    final_state, _ = integrate_interval(
        state,
        duration,
        kappa0,
        params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    return final_state


def ramp_q(
    state: np.ndarray,
    q_start: float,
    q_end: float,
    ramp_steps: int,
    duration_per_step: float,
    *,
    n: float,
    coupling_constant: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    label: str,
) -> np.ndarray:
    if ramp_steps < 2 or abs(q_end - q_start) < 1e-15:
        return state

    q_values = np.linspace(q_start, q_end, ramp_steps + 1)[1:]
    for index, q_value in enumerate(q_values, start=1):
        state = advance_at_q(
            state,
            float(q_value),
            duration_per_step,
            n=n,
            coupling_constant=coupling_constant,
            params=params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if index == 1 or index == ramp_steps or index % max(1, ramp_steps // 5) == 0:
            physical = internal_to_physical(state)
            print(
                f"{label}: [{index:>4}/{ramp_steps}] Q={q_value:9.4f}  "
                f"x={physical[8]:.8f}  rho=({physical[0]:.5f}, {physical[4]:.5f})"
            )
    return state


def chunk_metrics(
    sampled: np.ndarray,
    chunk_periods: int,
    samples_per_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = expit(sampled[8])
    xdot = x * (1.0 - x) * sampled[9]
    means = np.empty(chunk_periods)
    stds = np.empty(chunk_periods)
    amplitudes = np.empty(chunk_periods)

    for period in range(chunk_periods):
        start = period * samples_per_period
        stop = (period + 1) * samples_per_period + 1
        block = x[start:stop]
        means[period] = np.mean(block)
        stds[period] = np.std(block)
        amplitudes[period] = 0.5 * np.ptp(block)

    return means, stds, amplitudes, x, xdot


def convergence_diagnostics(
    amplitude: np.ndarray,
    convergence_window: int,
    amplitude_floor: float,
) -> tuple[float, float, float]:
    if amplitude.size < convergence_window:
        return float("nan"), float("inf"), float("inf")

    recent = np.maximum(amplitude[-convergence_window:], amplitude_floor)
    index = np.arange(convergence_window, dtype=float)
    log_slope = float(np.polyfit(index, np.log(recent), 1)[0])
    mean_amplitude = float(np.mean(recent))
    relative_span = float((np.max(recent) - np.min(recent)) / mean_amplitude)
    return mean_amplitude, log_slope, relative_span


def hold_until_saturated(
    state: np.ndarray,
    *,
    q_value: float,
    period_reference: float,
    n: float,
    coupling_constant: float,
    params: ModelParameters,
    method: str,
    rtol: float,
    atol: float,
    max_step: float,
    chunk_periods: int,
    min_hold_periods: int,
    max_hold_periods: int,
    convergence_window: int,
    convergence_log_slope: float,
    convergence_relative_span: float,
    required_consecutive_chunks: int,
    samples_per_period: int,
    late_phase_periods: int,
    amplitude_floor: float,
) -> SaturationResult:
    kappa0 = coupling_constant * (n / q_value)
    period_means: list[float] = []
    period_stds: list[float] = []
    period_amplitudes: list[float] = []
    total_periods = 0
    consecutive = 0
    converged = False
    last_phase_time = np.empty(0)
    last_phase_x = np.empty(0)
    last_phase_xdot = np.empty(0)

    while total_periods < max_hold_periods:
        this_chunk = min(chunk_periods, max_hold_periods - total_periods)
        duration = this_chunk * period_reference
        sample_count = this_chunk * samples_per_period + 1
        times = np.linspace(0.0, duration, sample_count)
        state, sampled = integrate_interval(
            state,
            duration,
            kappa0,
            params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            sample_times=times,
        )
        assert sampled is not None

        means, stds, amplitudes, x, xdot = chunk_metrics(
            sampled, this_chunk, samples_per_period
        )
        period_means.extend(means.tolist())
        period_stds.extend(stds.tolist())
        period_amplitudes.extend(amplitudes.tolist())
        total_periods += this_chunk

        keep_periods = min(late_phase_periods, this_chunk)
        keep_samples = keep_periods * samples_per_period + 1
        last_phase_time = times[-keep_samples:] / period_reference
        last_phase_time -= last_phase_time[0]
        last_phase_x = x[-keep_samples:]
        last_phase_xdot = xdot[-keep_samples:]

        amplitude_array = np.asarray(period_amplitudes)
        mean_amp, log_slope, relative_span = convergence_diagnostics(
            amplitude_array,
            convergence_window,
            amplitude_floor,
        )

        condition = (
            total_periods >= min_hold_periods
            and np.isfinite(log_slope)
            and abs(log_slope) <= convergence_log_slope
            and relative_span <= convergence_relative_span
        )
        consecutive = consecutive + 1 if condition else 0

        print(
            f"Q={q_value:8.3f} hold={total_periods:5d}/{max_hold_periods} T_ref  "
            f"A={mean_amp:.7e}  dlogA/dN={log_slope:+.3e}  "
            f"relative_span={relative_span:.3e}  stable_chunks={consecutive}"
        )

        if consecutive >= required_consecutive_chunks:
            converged = True
            break

    period_mean_array = np.asarray(period_means)
    period_std_array = np.asarray(period_stds)
    period_amplitude_array = np.asarray(period_amplitudes)
    late_count = min(convergence_window, period_amplitude_array.size)
    late_amplitude = float(np.median(period_amplitude_array[-late_count:]))
    late_std = float(np.median(period_std_array[-late_count:]))
    late_mean = float(np.mean(period_mean_array[-late_count:]))
    _, late_log_slope, late_relative_span = convergence_diagnostics(
        period_amplitude_array,
        late_count,
        amplitude_floor,
    )

    return SaturationResult(
        q_value=q_value,
        converged=converged,
        total_periods=total_periods,
        period_index=np.arange(1, total_periods + 1, dtype=int),
        period_mean=period_mean_array,
        period_std=period_std_array,
        period_amplitude=period_amplitude_array,
        late_amplitude=late_amplitude,
        late_std=late_std,
        late_mean=late_mean,
        late_log_slope=late_log_slope,
        late_relative_span=late_relative_span,
        phase_time=last_phase_time,
        phase_x=last_phase_x,
        phase_xdot=last_phase_xdot,
        final_state=state.copy(),
    )


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    if x.size < 2:
        return float("nan"), float("nan"), float("nan"), np.full_like(y, np.nan)
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    residual = np.sum((y - prediction) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return float(slope), float(intercept), float(r_squared), prediction


def origin_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        return float("nan"), float("nan"), np.full_like(y, np.nan)
    slope = float(np.dot(x, y) / denominator)
    prediction = slope * x
    residual = np.sum((y - prediction) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return slope, float(r_squared), prediction


def add_time_colored_phase_curve(ax, x: np.ndarray, y: np.ndarray, t: np.ndarray) -> LineCollection:
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t_mid = 0.5 * (t[:-1] + t[1:])
    collection = LineCollection(segments, array=t_mid, linewidths=1.4)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def plot_scaling_summary(
    results: list[SaturationResult],
    q_critical: float,
    output: str | Path,
) -> Path:
    output_path = Path(output)
    ordered = sorted(results, key=lambda result: result.q_value)
    q = np.asarray([result.q_value for result in ordered])
    mu = q - q_critical
    amplitude = np.asarray([result.late_amplitude for result in ordered])
    amplitude_squared = amplitude**2
    converged = np.asarray([result.converged for result in ordered], dtype=bool)
    fit_mask = converged & (mu > 0.0) & (amplitude > 0.0)
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = (mu > 0.0) & (amplitude > 0.0)

    x_fit = mu[fit_mask]
    y_fit = amplitude_squared[fit_mask]
    origin_slope, origin_r2, _ = origin_fit(x_fit, y_fit)
    free_slope, free_intercept, free_r2, _ = linear_fit(q[fit_mask], y_fit)
    q_critical_fit = -free_intercept / free_slope if free_slope > 0.0 else float("nan")

    log_slope, log_intercept, log_r2, _ = linear_fit(
        np.log(x_fit), np.log(amplitude[fit_mask])
    )
    beta = log_slope

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    for result in ordered:
        ax1.plot(
            result.period_index,
            result.period_amplitude,
            label=f"Q={result.q_value:g}" + ("" if result.converged else " (max hold)"),
            linewidth=1.5,
        )
    ax1.set_yscale("log")
    ax1.set_title("Adaptive convergence of the cycle amplitude")
    ax1.set_xlabel("fixed-Q hold time [reference periods]")
    ax1.set_ylabel("half peak-to-peak amplitude A")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.scatter(mu, amplitude_squared, label="measured asymptotic A^2")
    mu_line = np.linspace(0.0, max(mu) * 1.05, 300)
    if np.isfinite(origin_slope):
        ax2.plot(
            mu_line,
            origin_slope * mu_line,
            linestyle="--",
            label=f"fixed Qc fit: A^2={origin_slope:.3e}(Q-Qc), R^2={origin_r2:.5f}",
        )
    if np.isfinite(free_slope):
        q_line = q_critical + mu_line
        ax2.plot(
            mu_line,
            free_slope * q_line + free_intercept,
            linestyle=":",
            label=f"free intercept: Qc={q_critical_fit:.4f}, R^2={free_r2:.5f}",
        )
    ax2.set_title("Supercritical scaling test")
    ax2.set_xlabel("Q - Qc")
    ax2.set_ylabel("A^2")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.scatter(mu[fit_mask], amplitude[fit_mask], label="measured amplitude")
    if np.isfinite(beta):
        log_x_line = np.linspace(np.min(np.log(x_fit)), np.max(np.log(x_fit)), 300)
        ax3.plot(
            np.exp(log_x_line),
            np.exp(log_intercept + beta * log_x_line),
            linestyle="--",
            label=f"A proportional to (Q-Qc)^beta: beta={beta:.4f}, R^2={log_r2:.5f}",
        )
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title("Critical exponent estimate")
    ax3.set_xlabel("Q - Qc")
    ax3.set_ylabel("asymptotic amplitude A")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(q, [result.total_periods for result in ordered], marker="o", label="hold periods used")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(
        q,
        [abs(result.late_log_slope) for result in ordered],
        marker="s",
        linestyle="--",
        label="|late dlogA/dN|",
    )
    ax4.set_title("Convergence cost and residual drift")
    ax4.set_xlabel("Parameter Q")
    ax4.set_ylabel("hold periods")
    ax4_twin.set_ylabel("absolute late log-amplitude slope")
    ax4.grid(True, alpha=0.3)
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.suptitle(
        f"Nonlinear oscillation-amplitude scaling above Qc={q_critical:.4f}\n"
        f"beta={beta:.4f}, fitted Qc={q_critical_fit:.4f}",
        fontsize=17,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Scaling summary saved to: {output_path.resolve()}")
    return output_path


def plot_phase_portraits(results: list[SaturationResult], output: str | Path) -> Path:
    output_path = Path(output)
    ordered = sorted(results, key=lambda result: result.q_value)
    cols = 3
    rows = int(np.ceil(len(ordered) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    first_collection: LineCollection | None = None

    for ax, result in zip(axes, ordered):
        collection = add_time_colored_phase_curve(
            ax,
            result.phase_x - result.late_mean,
            result.phase_xdot,
            result.phase_time,
        )
        if first_collection is None:
            first_collection = collection
        ax.scatter(
            [result.phase_x[0] - result.late_mean],
            [result.phase_xdot[0]],
            marker="o",
            s=28,
            label="start",
        )
        ax.scatter(
            [result.phase_x[-1] - result.late_mean],
            [result.phase_xdot[-1]],
            marker="x",
            s=36,
            label="end",
        )
        status = "converged" if result.converged else "max hold reached"
        ax.set_title(
            f"Q={result.q_value:g} | A={result.late_amplitude:.3e}\n"
            f"{status} after {result.total_periods} T_ref"
        )
        ax.set_xlabel("x - late mean")
        ax.set_ylabel("x_dot")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(ordered):]:
        ax.axis("off")

    if first_collection is not None:
        colorbar = fig.colorbar(first_collection, ax=axes.tolist(), shrink=0.85)
        colorbar.set_label("local time [reference periods]")

    fig.suptitle("Late-time phase portraits along the nonlinear branch", fontsize=17)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Phase portraits saved to: {output_path.resolve()}")
    return output_path


def save_results_csv(
    results: list[SaturationResult],
    q_critical: float,
    output: str | Path,
) -> Path:
    output_path = Path(output)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "Q_minus_Qc",
                "converged",
                "hold_periods",
                "late_amplitude_half_peak_to_peak",
                "late_amplitude_squared",
                "late_std",
                "late_mean",
                "late_log_slope_per_reference_period",
                "late_relative_span",
            ]
        )
        for result in sorted(results, key=lambda item: item.q_value):
            writer.writerow(
                [
                    f"{result.q_value:.12g}",
                    f"{result.q_value - q_critical:.12g}",
                    int(result.converged),
                    result.total_periods,
                    f"{result.late_amplitude:.12e}",
                    f"{result.late_amplitude**2:.12e}",
                    f"{result.late_std:.12e}",
                    f"{result.late_mean:.12e}",
                    f"{result.late_log_slope:.12e}",
                    f"{result.late_relative_span:.12e}",
                ]
            )
    print(f"Scaling CSV saved to: {output_path.resolve()}")
    return output_path


def parse_q_values(text: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Q values must be comma-separated numbers.") from exc
    if len(values) < 3:
        raise argparse.ArgumentTypeError("Provide at least three Q values.")
    return values


def run_experiment(
    *,
    q_values: list[float],
    q_critical: float,
    n: float,
    q_low: float,
    q_seed_high: float,
    warmup_periods: int,
    seed_ramp_steps: int,
    seed_ramp_periods_per_step: float,
    seed_hold_periods: int,
    continuation_steps_per_100q: float,
    continuation_periods_per_step: float,
    chunk_periods: int,
    min_hold_periods: int,
    max_hold_periods: int,
    convergence_window: int,
    convergence_log_slope: float,
    convergence_relative_span: float,
    required_consecutive_chunks: int,
    samples_per_period: int,
    late_phase_periods: int,
    gamma_rho: float,
    method: str,
    rtol: float,
    atol: float,
    summary_output: str | Path,
    phase_output: str | Path,
    csv_output: str | Path,
) -> tuple[Path, Path, Path]:
    if any(q <= q_critical for q in q_values):
        raise ValueError("All Q values must be strictly above q_critical.")
    if q_seed_high < max(q_values):
        raise ValueError("q_seed_high must be greater than or equal to all target Q values.")

    coupling_constant = 55.0 / 6.0
    params = ModelParameters(gamma_rho=gamma_rho)
    initial_physical = build_initial_physical_state(n)
    period_x, period_delta, period_reference = characteristic_periods(initial_physical, params)
    max_step = min(period_x, period_delta) / 40.0
    state = physical_to_internal(initial_physical)

    print(f"Script version            = {SCRIPT_VERSION}")
    print(f"Measured Qc               = {q_critical:.6f}")
    print(f"Target Q values           = {sorted(q_values)}")
    print(f"Reference period          = {period_reference:.8f}")
    print(f"Adaptive hold             = {min_hold_periods}..{max_hold_periods} T_ref")
    print(f"Convergence log-slope tol = {convergence_log_slope:.3e} per T_ref")
    print(f"Convergence span tol      = {convergence_relative_span:.3e}")

    state = advance_at_q(
        state,
        q_low,
        warmup_periods * period_reference,
        n=n,
        coupling_constant=coupling_constant,
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    state = ramp_q(
        state,
        q_low,
        q_seed_high,
        seed_ramp_steps,
        seed_ramp_periods_per_step * period_reference,
        n=n,
        coupling_constant=coupling_constant,
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        label="SEED upward ramp",
    )

    state = advance_at_q(
        state,
        q_seed_high,
        seed_hold_periods * period_reference,
        n=n,
        coupling_constant=coupling_constant,
        params=params,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    print(f"High-Q seed prepared at Q={q_seed_high:g} for {seed_hold_periods} T_ref")

    results: list[SaturationResult] = []
    current_q = q_seed_high
    for q_target in sorted(q_values, reverse=True):
        delta_q = abs(current_q - q_target)
        ramp_steps = max(2, int(np.ceil(delta_q * continuation_steps_per_100q / 100.0)))
        state = ramp_q(
            state,
            current_q,
            q_target,
            ramp_steps,
            continuation_periods_per_step * period_reference,
            n=n,
            coupling_constant=coupling_constant,
            params=params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            label=f"Continuation to Q={q_target:g}",
        )

        result = hold_until_saturated(
            state,
            q_value=q_target,
            period_reference=period_reference,
            n=n,
            coupling_constant=coupling_constant,
            params=params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            chunk_periods=chunk_periods,
            min_hold_periods=min_hold_periods,
            max_hold_periods=max_hold_periods,
            convergence_window=convergence_window,
            convergence_log_slope=convergence_log_slope,
            convergence_relative_span=convergence_relative_span,
            required_consecutive_chunks=required_consecutive_chunks,
            samples_per_period=samples_per_period,
            late_phase_periods=late_phase_periods,
            amplitude_floor=1e-14,
        )
        results.append(result)
        state = result.final_state.copy()
        current_q = q_target
        print(
            f"Q={q_target:g} result: A={result.late_amplitude:.9e}, "
            f"A^2={result.late_amplitude**2:.9e}, converged={result.converged}, "
            f"hold={result.total_periods} T_ref\n"
        )

    summary_path = plot_scaling_summary(results, q_critical, summary_output)
    phase_path = plot_phase_portraits(results, phase_output)
    csv_path = save_results_csv(results, q_critical, csv_output)

    ordered = sorted(results, key=lambda result: result.q_value)
    q = np.asarray([result.q_value for result in ordered])
    mu = q - q_critical
    amplitude = np.asarray([result.late_amplitude for result in ordered])
    mask = np.asarray([result.converged for result in ordered]) & (amplitude > 0.0)
    if np.count_nonzero(mask) < 3:
        mask = amplitude > 0.0

    fixed_slope, fixed_r2, _ = origin_fit(mu[mask], amplitude[mask] ** 2)
    free_slope, free_intercept, free_r2, _ = linear_fit(q[mask], amplitude[mask] ** 2)
    fitted_qc = -free_intercept / free_slope if free_slope > 0.0 else float("nan")
    beta, _, beta_r2, _ = linear_fit(np.log(mu[mask]), np.log(amplitude[mask]))

    print("\nNonlinear scaling diagnostics")
    print("-----------------------------")
    print(f"Fixed-Qc A^2 slope       = {fixed_slope:.9e}")
    print(f"Fixed-Qc A^2 fit R^2     = {fixed_r2:.9f}")
    print(f"Free-intercept fitted Qc = {fitted_qc:.9f}")
    print(f"Free-intercept fit R^2   = {free_r2:.9f}")
    print(f"Critical exponent beta   = {beta:.9f}")
    print(f"Log-log fit R^2          = {beta_r2:.9f}")
    print("Supercritical Hopf-like expectation: beta approximately 0.5.")

    return summary_path, phase_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the saturated oscillation amplitude above Qc and test A^2 proportional to Q-Qc."
    )
    parser.add_argument("--q-values", type=parse_q_values, default=parse_q_values("523,525,530,540,550"))
    parser.add_argument("--q-critical", type=float, default=522.03)
    parser.add_argument("--n", type=float, default=10.0)
    parser.add_argument("--q-low", type=float, default=250.0)
    parser.add_argument("--q-seed-high", type=float, default=700.0)
    parser.add_argument("--warmup-periods", type=int, default=30)
    parser.add_argument("--seed-ramp-steps", type=int, default=240)
    parser.add_argument("--seed-ramp-periods-per-step", type=float, default=1.0)
    parser.add_argument("--seed-hold-periods", type=int, default=500)
    parser.add_argument("--continuation-steps-per-100q", type=float, default=80.0)
    parser.add_argument("--continuation-periods-per-step", type=float, default=1.0)
    parser.add_argument("--chunk-periods", type=int, default=100)
    parser.add_argument("--min-hold-periods", type=int, default=600)
    parser.add_argument("--max-hold-periods", type=int, default=8000)
    parser.add_argument("--convergence-window", type=int, default=200)
    parser.add_argument("--convergence-log-slope", type=float, default=2e-5)
    parser.add_argument("--convergence-relative-span", type=float, default=0.02)
    parser.add_argument("--required-consecutive-chunks", type=int, default=2)
    parser.add_argument("--samples-per-period", type=int, default=96)
    parser.add_argument("--late-phase-periods", type=int, default=20)
    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument(
        "--method",
        choices=("LSODA", "DOP853", "RK45", "Radau", "BDF"),
        default="LSODA",
    )
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--summary-output", default="hopf_amplitude_scaling.png")
    parser.add_argument("--phase-output", default="hopf_phase_portraits.png")
    parser.add_argument("--csv-output", default="hopf_amplitude_scaling.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        q_values=args.q_values,
        q_critical=args.q_critical,
        n=args.n,
        q_low=args.q_low,
        q_seed_high=args.q_seed_high,
        warmup_periods=args.warmup_periods,
        seed_ramp_steps=args.seed_ramp_steps,
        seed_ramp_periods_per_step=args.seed_ramp_periods_per_step,
        seed_hold_periods=args.seed_hold_periods,
        continuation_steps_per_100q=args.continuation_steps_per_100q,
        continuation_periods_per_step=args.continuation_periods_per_step,
        chunk_periods=args.chunk_periods,
        min_hold_periods=args.min_hold_periods,
        max_hold_periods=args.max_hold_periods,
        convergence_window=args.convergence_window,
        convergence_log_slope=args.convergence_log_slope,
        convergence_relative_span=args.convergence_relative_span,
        required_consecutive_chunks=args.required_consecutive_chunks,
        samples_per_period=args.samples_per_period,
        late_phase_periods=args.late_phase_periods,
        gamma_rho=args.gamma_rho,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        summary_output=args.summary_output,
        phase_output=args.phase_output,
        csv_output=args.csv_output,
    )


if __name__ == "__main__":
    main()
