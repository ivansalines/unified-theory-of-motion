#!/usr/bin/env python3
"""Hysteresis test with adiabatic sweep, three phase spirals, and transition views.

Main features:
- rho1 and rho2 are integrated as exp(y), so they remain strictly positive;
- x is integrated as sigmoid(z), so it remains strictly inside (0, 1);
- each Q step has a separate settling window and measurement window;
- x_mean and x_std are computed only on the measurement window;
- the script saves:
  1) the hysteresis loop,
  2) three phase portraits x vs x_dot,
  3) three local time traces showing the adiabatic transition.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit

# ─────────────────────────────────────────────
# Model constants
# ─────────────────────────────────────────────
ALPHA, BETA, GAMMA = 1.0, 0.8, 0.2


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
class PhaseTarget:
    label: str
    segment: str
    q_target: float


@dataclass
class PhaseCapture:
    label: str
    segment: str
    q_target: float
    q_selected: float
    local_times: np.ndarray
    x_samples: np.ndarray
    xdot_samples: np.ndarray


# ─────────────────────────────────────────────
# Potentials
# ─────────────────────────────────────────────
def dU(rho: float, e_reserve: float = 0.0) -> float:
    """Derivative of the local radial potential."""
    return (
        2.0 * ALPHA * rho
        - 3.0 * BETA * rho**2
        + 4.0 * GAMMA * rho**3
        - e_reserve**2 / rho**3
    )


def dV_barrier(x: float, barrier_B: float) -> float:
    """Derivative of the barrier potential that diverges at x=0 and x=1."""
    return -barrier_B * (1.0 / x - 1.0 / (1.0 - x))


# Physical state for reading:
# [rho1, rho1_dot, theta1, theta1_dot,
#  rho2, rho2_dot, theta2, theta2_dot,
#  x, x_dot, Theta]
#
# Integrated state:
# [y1, y1_dot, theta1, theta1_dot,
#  y2, y2_dot, theta2, theta2_dot,
#  z, z_dot, Theta]
# with rho = exp(y), x = sigmoid(z)


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


# ─────────────────────────────────────────────
# Equations of motion in transformed coordinates
# ─────────────────────────────────────────────
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

    # rho_ddot = rho * (y_ddot + y_dot^2)
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

    # x_ddot = g * z_ddot + g * (1 - 2x) * z_dot^2
    z_ddot = x_ddot / g - (1.0 - 2.0 * x) * z_dot**2

    Theta_dot = x

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
            Theta_dot,
        ],
        dtype=float,
    )

    if not np.all(np.isfinite(deriv)):
        raise FloatingPointError("Non-finite derivative encountered during integration.")

    return deriv


# ─────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────
def characteristic_periods(
    initial_physical_state: Sequence[float], params: ModelParameters
) -> tuple[float, float, float]:
    """Return the damped x period, initial Delta period, and reference period."""
    state = np.asarray(initial_physical_state, dtype=float)
    x0 = state[8]

    stiffness_x = params.barrier_B * (1.0 / x0**2 + 1.0 / (1.0 - x0) ** 2)
    omega_n_sq = stiffness_x / params.m_x
    damping_rate = params.gamma_x / (2.0 * params.m_x)
    omega_d_sq = omega_n_sq - damping_rate**2
    if omega_d_sq <= 0.0:
        raise ValueError(
            "The x degree of freedom is not underdamped: no real damped natural period exists."
        )

    period_x = 2.0 * np.pi / np.sqrt(omega_d_sq)
    delta_rate0 = state[3] + state[7] - x0
    period_delta = period_x if abs(delta_rate0) < 1e-10 else 2.0 * np.pi / abs(delta_rate0)
    period_reference = max(period_x, period_delta)
    return period_x, period_delta, period_reference


# ─────────────────────────────────────────────
# Numerical integration helper
# ─────────────────────────────────────────────
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
    if duration <= 0.0:
        raise ValueError("Integration duration must be positive.")

    if sample_times is not None:
        sample_times = np.asarray(sample_times, dtype=float)
        if sample_times.ndim != 1 or sample_times.size == 0:
            raise ValueError("sample_times must be a non-empty 1D array.")
        if sample_times[0] < 0.0 or sample_times[-1] > duration:
            raise ValueError("sample_times must lie inside the integration interval.")

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

    final_state = solution.y[:, -1]
    sampled_states = solution.y if sample_times is not None else None
    return final_state, sampled_states


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def plot_three_phase_portraits(captures: list[PhaseCapture], output: str | Path) -> Path:
    output_path = Path(output)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    for ax, capture in zip(axes, captures):
        xs = capture.x_samples
        xds = capture.xdot_samples
        for i in range(len(xs) - 1):
            alpha = 0.12 + 0.80 * (i / max(1, len(xs) - 2))
            ax.plot(xs[i:i+2], xds[i:i+2], linewidth=1.25, alpha=alpha)

        ax.scatter(xs[0], xds[0], s=36, marker="o", label="start")
        ax.scatter(xs[-1], xds[-1], s=48, marker="x", label="end")
        ax.set_title(
            f"{capture.label}\n"
            f"{capture.segment.capitalize()} branch | target Q={capture.q_target:.1f}\n"
            f"used Q={capture.q_selected:.3f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("x_dot")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Three phase spirals of the third plane: x vs x_dot", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Three phase portraits saved to: {output_path.resolve()}")
    return output_path


def plot_adiabatic_transition_panels(
    captures: list[PhaseCapture],
    settle_duration: float,
    output: str | Path,
) -> Path:
    output_path = Path(output)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)

    for ax, capture in zip(axes, captures):
        t = capture.local_times
        x = capture.x_samples

        ax.axvspan(0.0, settle_duration, alpha=0.08, label="settling window")
        ax.axvspan(settle_duration, t[-1], alpha=0.05, label="measurement window")
        ax.axvline(settle_duration, linestyle="--", linewidth=1.3)
        ax.plot(t, x, linewidth=1.8)
        ax.scatter(t[0], x[0], s=28, marker="o", label="start")
        ax.scatter(t[-1], x[-1], s=36, marker="x", label="end")
        ax.set_title(
            f"{capture.label}\n"
            f"{capture.segment.capitalize()} branch | target Q={capture.q_target:.1f}\n"
            f"used Q={capture.q_selected:.3f}"
        )
        ax.set_xlabel("local time")
        ax.set_ylabel("x")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(
        "Adiabatic transition snapshots: local response at selected Q steps",
        fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Adiabatic transition panels saved to: {output_path.resolve()}")
    return output_path


# ─────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────
def run_hysteresis(
    *,
    n: float = 10.0,
    Q_min: float = 400.0,
    Q_max: float = 1200.0,
    Q_steps: int = 150,
    settle_periods: float = 1.0,
    measure_periods: float = 2.0,
    warmup_periods: float = 3.0,
    samples_per_period: int = 160,
    gamma_rho: float = 0.3,
    method: str = "LSODA",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    hysteresis_output: str | Path = "hysteresis_test.png",
    spirals_output: str | Path = "three_phase_spirals.png",
    transition_output: str | Path = "adiabatic_transition_panels.png",
) -> tuple[Path, Path, Path]:
    if Q_steps < 2:
        raise ValueError("Q_steps must be at least 2.")
    if Q_min <= 0.0 or Q_max <= Q_min:
        raise ValueError("You need 0 < Q_min < Q_max.")
    if settle_periods < 0.0 or measure_periods <= 0.0:
        raise ValueError("Settling and measurement periods must be valid.")
    if warmup_periods < 0.0 or samples_per_period < 16:
        raise ValueError("Invalid timing parameters.")
    if gamma_rho < 0.0:
        raise ValueError("gamma_rho cannot be negative.")

    C = 55.0 / 6.0
    omega0, rho0, delta0, x0 = 0.05, 1.2, 0.3, 0.5
    params = ModelParameters(gamma_rho=gamma_rho)

    initial_physical_state = np.array(
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
    state = physical_to_internal(initial_physical_state)

    period_x, period_delta, period_ref = characteristic_periods(initial_physical_state, params)
    settle_duration = settle_periods * period_ref
    measure_duration = measure_periods * period_ref
    warmup_duration = warmup_periods * period_ref

    # Keep at least ~40 maximum internal steps over the fastest period.
    max_step = min(period_x, period_delta) / 40.0

    Q_up = np.linspace(Q_min, Q_max, Q_steps)
    Q_down = Q_up[-2::-1]
    Q_path = np.concatenate((Q_up, Q_down))

    q_mid = 0.5 * (Q_min + Q_max)
    targets = [
        PhaseTarget("Spiral 1: low upward sweep", "upward", Q_min),
        PhaseTarget("Spiral 2: mid upward sweep", "upward", q_mid),
        PhaseTarget("Spiral 3: mid downward sweep", "downward", q_mid),
    ]
    best_distance = [float("inf")] * len(targets)
    best_capture: list[PhaseCapture | None] = [None] * len(targets)

    kappa0_init = C * (n / Q_min)
    if warmup_duration > 0.0:
        state, _ = integrate_interval(
            state,
            warmup_duration,
            kappa0_init,
            params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

    print(f"n={n:g}; Q: {Q_min:g} -> {Q_max:g} -> {Q_min:g}")
    print(f"Radial damping gamma_rho = {gamma_rho:g}")
    if gamma_rho == 0.0:
        print(
            "WARNING: without radial damping, the rho modes do not truly settle "
            "and long sweeps may grow without bound."
        )
    print(f"Damped x period       = {period_x:.6f}")
    print(f"Initial Delta period  = {period_delta:.6f}")
    print(f"Reference period      = {period_ref:.6f}")
    print(
        f"Per Q: {settle_periods:g} reference periods for settling + "
        f"{measure_periods:g} reference periods for measurement"
    )

    total_duration = settle_duration + measure_duration
    total_sample_count = max(
        64, int(np.ceil((settle_periods + measure_periods) * samples_per_period)) + 1
    )
    local_sample_times = np.linspace(0.0, total_duration, total_sample_count)
    measure_mask = local_sample_times >= (settle_duration - 1e-12)
    measure_times = local_sample_times[measure_mask] - settle_duration

    x_mean_history: list[float] = []
    x_std_history: list[float] = []

    total = Q_path.size
    for index, Q_current in enumerate(Q_path, start=1):
        kappa0 = C * (n / Q_current)
        state, sampled = integrate_interval(
            state,
            total_duration,
            kappa0,
            params,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            sample_times=local_sample_times,
        )
        assert sampled is not None

        x_samples = expit(sampled[8])
        xdot_samples = x_samples * (1.0 - x_samples) * sampled[9]

        x_measure = x_samples[measure_mask]
        x_mean = np.trapezoid(x_measure, measure_times) / measure_duration
        x_mean_history.append(float(x_mean))
        x_std_history.append(float(np.std(x_measure)))

        current_segment = "upward" if index <= Q_steps else "downward"
        for i, target in enumerate(targets):
            if current_segment != target.segment:
                continue
            distance = abs(Q_current - target.q_target)
            if distance < best_distance[i]:
                best_distance[i] = distance
                best_capture[i] = PhaseCapture(
                    label=target.label,
                    segment=current_segment,
                    q_target=target.q_target,
                    q_selected=float(Q_current),
                    local_times=local_sample_times.copy(),
                    x_samples=x_samples.copy(),
                    xdot_samples=xdot_samples.copy(),
                )

        if index == 1 or index == total or index % max(1, total // 10) == 0:
            physical = internal_to_physical(state)
            print(
                f"[{index:>3}/{total}] Q={Q_current:8.3f}  "
                f"x_mean={x_mean:.7f}  x_std={x_std_history[-1]:.3e}  "
                f"rho=({physical[0]:.4f}, {physical[4]:.4f})"
            )

    x_mean_array = np.asarray(x_mean_history)
    x_std_array = np.asarray(x_std_history)

    upward_slice = slice(0, Q_steps)
    downward_slice = slice(Q_steps, None)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(Q_up, x_mean_array[upward_slice], label="Upward Q sweep", linewidth=2.0)
    ax.plot(
        Q_down,
        x_mean_array[downward_slice],
        label="Downward Q sweep",
        linestyle="--",
        linewidth=2.0,
    )
    ax.fill_between(
        Q_up,
        np.clip(x_mean_array[upward_slice] - x_std_array[upward_slice], 0.0, 1.0),
        np.clip(x_mean_array[upward_slice] + x_std_array[upward_slice], 0.0, 1.0),
        alpha=0.12,
    )
    ax.fill_between(
        Q_down,
        np.clip(x_mean_array[downward_slice] - x_std_array[downward_slice], 0.0, 1.0),
        np.clip(x_mean_array[downward_slice] + x_std_array[downward_slice], 0.0, 1.0),
        alpha=0.12,
    )
    ax.set_title(
        f"Topological hysteresis test (n={n:g})\n"
        f"mean over {measure_periods:g} reference periods after "
        f"{settle_periods:g} settling periods"
    )
    ax.set_xlabel("Parameter Q")
    ax.set_ylabel("Mean third-plane position (x_mean)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    hysteresis_path = Path(hysteresis_output)
    fig.savefig(hysteresis_path, dpi=160)
    plt.close(fig)

    captures = [capture for capture in best_capture if capture is not None]
    if len(captures) != 3:
        raise RuntimeError("Could not capture all three requested spirals.")

    spirals_path = plot_three_phase_portraits(captures, spirals_output)
    transition_path = plot_adiabatic_transition_panels(
        captures, settle_duration, transition_output
    )

    print(f"Hysteresis plot saved to: {hysteresis_path.resolve()}")
    return hysteresis_path, spirals_path, transition_path


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hysteresis test with phase spirals and adiabatic transition views."
    )
    parser.add_argument("--n", type=float, default=10.0)
    parser.add_argument("--q-min", type=float, default=400.0)
    parser.add_argument("--q-max", type=float, default=1200.0)
    parser.add_argument("--q-steps", type=int, default=150)
    parser.add_argument("--settle-periods", type=float, default=1.0)
    parser.add_argument("--measure-periods", type=float, default=2.0)
    parser.add_argument("--warmup-periods", type=float, default=3.0)
    parser.add_argument("--samples-per-period", type=int, default=160)
    parser.add_argument(
        "--gamma-rho",
        type=float,
        default=0.3,
        help="radial viscous damping; use 0 to recover the original radial dynamics",
    )
    parser.add_argument(
        "--method",
        choices=("LSODA", "DOP853", "RK45", "Radau", "BDF"),
        default="LSODA",
    )
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--hysteresis-output", default="hysteresis_test.png")
    parser.add_argument("--spirals-output", default="three_phase_spirals.png")
    parser.add_argument(
        "--transition-output",
        default="adiabatic_transition_panels.png",
        help="output file for the local adiabatic transition panels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_hysteresis(
        n=args.n,
        Q_min=args.q_min,
        Q_max=args.q_max,
        Q_steps=args.q_steps,
        settle_periods=args.settle_periods,
        measure_periods=args.measure_periods,
        warmup_periods=args.warmup_periods,
        samples_per_period=args.samples_per_period,
        gamma_rho=args.gamma_rho,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        hysteresis_output=args.hysteresis_output,
        spirals_output=args.spirals_output,
        transition_output=args.transition_output,
    )


if __name__ == "__main__":
    main()
