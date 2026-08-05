#!/usr/bin/env python3
"""Extract the continuous rocking generator from the measured B(s) stride.

The input is the raw NPZ produced by
``test_support_stride_involution_audit.py``.

No nonlinear orbit is regenerated.  The script works directly with the
exported reduced 9x9 rotation-time matrices B(s) over two consecutive
P_line strokes.

For every Q it tests the model

    R(s) = B(s) - C
    dR/ds ~= G R

inside the measured two-dimensional active matrix plane, where C is the
affine stride center.  It also tests the harmonic representation

    R(s) ~= A cos(phi) + D sin(phi)
    phi(s + P_line) ~= phi(s) + pi.

The principal outputs are:

* the direct 2D generator G and its normalized complex structure J = G/omega;
* exp(P_line G) versus -I and exp(2 P_line G) versus I;
* phase speed and its nonuniformity;
* odd/even Fourier content of the two-stroke loop;
* the matrix entries that carry the fundamental rocking mode;
* direct plots and machine-readable CSV/NPZ exports.

All numerical conclusions are based on the centered moving part of B(s), not
on the much larger static Frobenius norm of the complete matrix.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


SCRIPT_VERSION = "2026-08-05-stride-generator-audit-v1"

REDUCED_NAMES = (
    "y1",
    "v1",
    "omega1",
    "y2",
    "v2",
    "omega2",
    "z",
    "vz",
    "Delta",
)


@dataclass
class CaseResult:
    q: float
    token: str
    p_line: float
    p_oriented: float
    anchor_count: int
    phase_points: int
    matrix_dimension: int
    affine_centers: np.ndarray
    cycles: np.ndarray
    centered_cycles: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_explained: np.ndarray
    scores: np.ndarray
    harmonic_scores: np.ndarray
    inferred_phase: np.ndarray
    phase_speed: np.ndarray
    pointwise_generator_residual: np.ndarray
    generator: np.ndarray
    generator_offset: np.ndarray
    generator_eigenvalues: np.ndarray
    omega_measured: float
    omega_expected: float
    normalized_generator: np.ndarray
    half_propagator: np.ndarray
    full_propagator: np.ndarray
    harmonic_r2: np.ndarray
    harmonic3_r2: np.ndarray
    fundamental_fraction: np.ndarray
    even_harmonic_fraction: np.ndarray
    harmonic_energy: np.ndarray
    aligned_complex_modes: np.ndarray
    mean_complex_mode: np.ndarray
    anatomy_amplitude: np.ndarray
    anatomy_phase: np.ndarray
    summary: dict[str, float | str | bool]


def q_token(q: float) -> str:
    return (
        f"{q:.10f}"
        .rstrip("0")
        .rstrip(".")
        .replace("-", "m")
        .replace(".", "p")
    )


def case_prefixes(archive: np.lib.npyio.NpzFile) -> list[str]:
    prefixes = []
    for name in archive.files:
        if name.startswith("q_") and name.endswith("_q"):
            prefixes.append(name[:-2])
    return sorted(prefixes, key=lambda prefix: float(archive[f"{prefix}_q"]))


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(a, ord="fro")),
        float(np.linalg.norm(b, ord="fro")),
        1e-300,
    )
    return float(np.linalg.norm(a - b, ord="fro") / denominator)


def run_pca(vectors: np.ndarray):
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    _, singular_values, components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    variance = singular_values**2 / max(vectors.shape[0] - 1, 1)
    total = float(np.sum(variance))
    explained = variance / total if total > 0.0 else np.zeros_like(variance)
    scores = centered @ components.T
    return mean, components, singular_values, explained, scores


def periodic_derivative(values: np.ndarray, step: float) -> np.ndarray:
    return (
        np.roll(values, -1, axis=1)
        - np.roll(values, 1, axis=1)
    ) / (2.0 * step)


def fit_planar_generator(
    scores: np.ndarray,
    ds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    derivatives = periodic_derivative(scores, ds)
    x = scores.reshape(-1, 2)
    dx = derivatives.reshape(-1, 2)

    design = np.column_stack((x, np.ones(x.shape[0], dtype=float)))
    coefficients, _, _, _ = np.linalg.lstsq(design, dx, rcond=None)
    generator = coefficients[:2]
    offset = coefficients[2]

    predicted = x @ generator + offset
    residual = np.linalg.norm(dx - predicted, axis=1)
    scale = np.linalg.norm(dx, axis=1)
    pointwise = residual / np.maximum(scale, 1e-300)
    return generator, offset, derivatives, pointwise.reshape(scores.shape[:2])


def harmonic_fit(
    vectors: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit first and first-three harmonics anchor by anchor.

    Parameters
    ----------
    vectors
        Array with shape (anchors, points_per_two_stroke, coordinates).

    Returns
    -------
    fitted_first
        First-harmonic reconstruction.
    r2_first
        First-harmonic explained variance per anchor.
    r2_third
        Explained variance using harmonics 1, 2, and 3.
    fundamental_fraction
        Fraction of non-DC Fourier energy in harmonic k=1.
    even_fraction
        Fraction of non-DC Fourier energy in even harmonics.
    harmonic_energy
        Normalized harmonic energy spectrum per anchor.
    """
    anchor_count, point_count, coordinate_count = vectors.shape
    theta = 2.0 * np.pi * np.arange(point_count, dtype=float) / point_count
    design_first = np.column_stack((np.cos(theta), np.sin(theta)))
    design_third = np.column_stack(
        [
            function(harmonic * theta)
            for harmonic in range(1, 4)
            for function in (np.cos, np.sin)
        ]
    )

    fitted = np.empty_like(vectors)
    r2_first = np.empty(anchor_count, dtype=float)
    r2_third = np.empty(anchor_count, dtype=float)

    fft_count = point_count // 2 + 1
    harmonic_energy = np.zeros((anchor_count, fft_count), dtype=float)
    fundamental_fraction = np.empty(anchor_count, dtype=float)
    even_fraction = np.empty(anchor_count, dtype=float)

    for anchor in range(anchor_count):
        current = vectors[anchor]

        beta_first, _, _, _ = np.linalg.lstsq(
            design_first,
            current,
            rcond=None,
        )
        fit_first = design_first @ beta_first
        fitted[anchor] = fit_first

        beta_third, _, _, _ = np.linalg.lstsq(
            design_third,
            current,
            rcond=None,
        )
        fit_third = design_third @ beta_third

        total_sum = float(np.sum(current * current))
        if total_sum <= 0.0:
            r2_first[anchor] = float("nan")
            r2_third[anchor] = float("nan")
        else:
            r2_first[anchor] = 1.0 - float(
                np.sum((current - fit_first) ** 2)
            ) / total_sum
            r2_third[anchor] = 1.0 - float(
                np.sum((current - fit_third) ** 2)
            ) / total_sum

        fourier = np.fft.rfft(current, axis=0)
        weights = np.ones(fourier.shape[0], dtype=float)
        if point_count % 2 == 0:
            weights[1:-1] = 2.0
        else:
            weights[1:] = 2.0

        energy = weights * np.sum(np.abs(fourier) ** 2, axis=1)
        dynamic_total = float(np.sum(energy[1:]))
        if dynamic_total > 0.0:
            normalized = energy / dynamic_total
            fundamental_fraction[anchor] = normalized[1]
            even_fraction[anchor] = float(np.sum(normalized[2::2]))
            harmonic_energy[anchor] = normalized
        else:
            fundamental_fraction[anchor] = float("nan")
            even_fraction[anchor] = float("nan")
            harmonic_energy[anchor] = np.nan

    return (
        fitted,
        r2_first,
        r2_third,
        fundamental_fraction,
        even_fraction,
        harmonic_energy,
    )


def infer_phase_and_speed(
    scores: np.ndarray,
    ds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_count, point_count, _ = scores.shape
    theta = 2.0 * np.pi * np.arange(point_count, dtype=float) / point_count
    design = np.column_stack((np.cos(theta), np.sin(theta)))

    fitted_scores = np.empty_like(scores)
    inferred = np.empty((anchor_count, point_count), dtype=float)
    speed = np.empty((anchor_count, point_count), dtype=float)

    for anchor in range(anchor_count):
        beta, _, _, _ = np.linalg.lstsq(
            design,
            scores[anchor],
            rcond=None,
        )
        fitted_scores[anchor] = design @ beta

        if abs(np.linalg.det(beta)) <= 1e-14 * max(
            float(np.linalg.norm(beta, ord="fro") ** 2),
            1.0,
        ):
            raise RuntimeError(
                "The fitted active-plane harmonic basis is singular."
            )

        circular = scores[anchor] @ np.linalg.inv(beta)
        complex_phase = circular[:, 0] + 1j * circular[:, 1]

        unwrapped = np.unwrap(np.angle(complex_phase))
        slope = float(
            np.polyfit(
                np.arange(point_count, dtype=float) * ds,
                unwrapped,
                1,
            )[0]
        )
        if slope < 0.0:
            complex_phase = np.conjugate(complex_phase)
            unwrapped = np.unwrap(np.angle(complex_phase))

        unwrapped -= unwrapped[0]
        local_speed = np.gradient(
            unwrapped,
            ds,
            edge_order=2,
        )

        # The sampled cycle is endpoint-exclusive.  The first and last two
        # derivative values would otherwise use a one-sided seam estimate and
        # artificially inflate the phase-speed variation.
        local_speed[:2] = np.nan
        local_speed[-2:] = np.nan

        inferred[anchor] = unwrapped
        speed[anchor] = local_speed

    return fitted_scores, inferred, speed


def aligned_fundamental_modes(
    centered_cycles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_count, point_count, rows, columns = centered_cycles.shape
    flattened = centered_cycles.reshape(anchor_count, point_count, -1)
    theta = 2.0 * np.pi * np.arange(point_count, dtype=float) / point_count
    design = np.column_stack((np.cos(theta), np.sin(theta)))

    complex_modes = np.empty(
        (anchor_count, rows * columns),
        dtype=complex,
    )
    for anchor in range(anchor_count):
        beta, _, _, _ = np.linalg.lstsq(
            design,
            flattened[anchor],
            rcond=None,
        )
        complex_modes[anchor] = beta[0] - 1j * beta[1]

    reference = complex_modes[0]
    aligned = np.empty_like(complex_modes)
    for anchor in range(anchor_count):
        overlap = np.vdot(reference, complex_modes[anchor])
        phase = float(np.angle(overlap))
        aligned[anchor] = complex_modes[anchor] * np.exp(-1j * phase)

    mean_mode = np.mean(aligned, axis=0).reshape(rows, columns)
    return aligned.reshape(anchor_count, rows, columns), mean_mode


def anatomy_rows(
    result: CaseResult,
) -> list[dict[str, float | str | int]]:
    amplitude = result.anatomy_amplitude
    phase = result.anatomy_phase
    total = float(np.linalg.norm(amplitude))
    maximum = max(float(np.max(amplitude)), 1e-300)

    rows = []
    for output_index, output_name in enumerate(REDUCED_NAMES):
        for input_index, input_name in enumerate(REDUCED_NAMES):
            value = float(amplitude[output_index, input_index])
            rows.append(
                {
                    "Q": result.q,
                    "output_index": output_index,
                    "output_coordinate": output_name,
                    "input_index": input_index,
                    "input_coordinate": input_name,
                    "fundamental_amplitude": value,
                    "fraction_of_mode_frobenius_norm": value / max(total, 1e-300),
                    "fraction_of_maximum_entry": value / maximum,
                    "aligned_phase_degrees": math.degrees(
                        float(phase[output_index, input_index])
                    ),
                }
            )

    rows.sort(
        key=lambda row: float(row["fundamental_amplitude"]),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def analyze_case(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    maximum_harmonic: int,
) -> CaseResult:
    q = float(archive[f"{prefix}_q"])
    p_line = float(archive[f"{prefix}_p_line"])
    p_oriented = float(archive[f"{prefix}_p_oriented"])

    matrices_0 = np.asarray(archive[f"{prefix}_matrices_0"], dtype=float)
    matrices_p = np.asarray(archive[f"{prefix}_matrices_p"], dtype=float)
    affine_centers = np.asarray(
        archive[f"{prefix}_affine_centers"],
        dtype=float,
    )

    if matrices_0.shape != matrices_p.shape:
        raise ValueError(f"Q={q:g}: incompatible matrix-stroke shapes.")
    if matrices_0.ndim != 4:
        raise ValueError(f"Q={q:g}: expected anchor x phase x row x column.")

    anchor_count, phase_points, rows, columns = matrices_0.shape
    if rows != columns:
        raise ValueError(f"Q={q:g}: reduced B(s) matrices are not square.")
    if affine_centers.shape != (anchor_count, rows, columns):
        raise ValueError(f"Q={q:g}: incompatible affine-center array.")

    cycles = np.concatenate((matrices_0, matrices_p), axis=1)
    centered_cycles = cycles - affine_centers[:, None, :, :]
    point_count = centered_cycles.shape[1]
    ds = p_line / phase_points

    vectors = centered_cycles.reshape(anchor_count * point_count, -1)
    (
        pca_mean,
        pca_components,
        _,
        pca_explained,
        all_scores,
    ) = run_pca(vectors)

    scores = all_scores[:, :2].reshape(anchor_count, point_count, 2)
    # Remove the tiny residual anchor-wise score offset.
    scores = scores - np.mean(scores, axis=1, keepdims=True)

    (
        generator,
        generator_offset,
        derivatives,
        pointwise_generator_residual,
    ) = fit_planar_generator(scores, ds)

    eigenvalues = np.linalg.eigvals(generator)
    omega_measured = float(np.mean(np.abs(np.imag(eigenvalues))))
    omega_expected = math.pi / p_line
    if omega_measured <= 0.0 or not np.isfinite(omega_measured):
        raise RuntimeError(f"Q={q:g}: invalid generator frequency.")

    normalized_generator = generator / omega_measured
    half_propagator = expm(p_line * generator)
    full_propagator = expm(2.0 * p_line * generator)

    flattened_cycles = centered_cycles.reshape(
        anchor_count,
        point_count,
        rows * columns,
    )
    (
        harmonic_matrix_fit,
        harmonic_r2,
        harmonic3_r2,
        fundamental_fraction,
        even_fraction,
        harmonic_energy,
    ) = harmonic_fit(flattened_cycles)

    (
        harmonic_scores,
        inferred_phase,
        phase_speed,
    ) = infer_phase_and_speed(scores, ds)

    aligned_modes, mean_mode = aligned_fundamental_modes(centered_cycles)
    anatomy_amplitude = np.abs(mean_mode)
    anatomy_phase = np.angle(mean_mode)

    expected_identity = np.eye(2)
    generator_prediction = (
        scores.reshape(-1, 2) @ generator + generator_offset
    )
    derivative_flat = derivatives.reshape(-1, 2)

    global_generator_residual = float(
        np.linalg.norm(derivative_flat - generator_prediction)
        / max(np.linalg.norm(derivative_flat), 1e-300)
    )
    normalized_square_error = relative_frobenius(
        normalized_generator @ normalized_generator,
        -expected_identity,
    )
    half_turn_error = relative_frobenius(
        half_propagator,
        -expected_identity,
    )
    full_turn_error = relative_frobenius(
        full_propagator,
        expected_identity,
    )

    positive_speed = phase_speed[np.isfinite(phase_speed)]
    phase_speed_mean = float(np.mean(positive_speed))
    phase_speed_std = float(np.std(positive_speed, ddof=1))
    phase_speed_cv = phase_speed_std / max(abs(phase_speed_mean), 1e-300)

    cumulative = np.cumsum(pca_explained)
    active_plane_variance = float(cumulative[1])

    median_fundamental = float(np.nanmedian(fundamental_fraction))
    median_even = float(np.nanmedian(even_fraction))
    median_harmonic_r2 = float(np.nanmedian(harmonic_r2))
    median_harmonic3_r2 = float(np.nanmedian(harmonic3_r2))

    supported = bool(
        active_plane_variance >= 0.999
        and median_fundamental >= 0.99
        and median_even <= 1e-3
        and global_generator_residual <= 0.10
        and normalized_square_error <= 0.01
        and half_turn_error <= 0.01
        and full_turn_error <= 0.02
        and phase_speed_cv <= 0.02
    )

    if supported:
        classification = (
            "Continuous planar rocking generator supported: the centered B(s) "
            "stride is overwhelmingly first-harmonic and odd under a half "
            "cycle, while one constant two-dimensional generator produces an "
            "approximately pi rotation over P_line and identity over 2 P_line."
        )
    else:
        classification = (
            "A two-stroke active plane is present, but the selected thresholds "
            "do not support a single nearly uniform continuous rocking "
            "generator. Inspect generator residual, harmonic content, and "
            "phase-speed modulation."
        )

    summary: dict[str, float | str | bool] = {
        "p_line": p_line,
        "p_oriented": p_oriented,
        "anchor_count": anchor_count,
        "phase_points_per_stroke": phase_points,
        "matrix_dimension": rows,
        "active_plane_variance": active_plane_variance,
        "pc1_explained": float(pca_explained[0]),
        "pc2_explained": float(pca_explained[1]),
        "median_first_harmonic_R2": median_harmonic_r2,
        "median_first_three_harmonics_R2": median_harmonic3_r2,
        "median_fundamental_energy_fraction": median_fundamental,
        "median_even_harmonic_energy_fraction": median_even,
        "generator_g00": float(generator[0, 0]),
        "generator_g01": float(generator[0, 1]),
        "generator_g10": float(generator[1, 0]),
        "generator_g11": float(generator[1, 1]),
        "generator_offset_norm": float(np.linalg.norm(generator_offset)),
        "generator_trace": float(np.trace(generator)),
        "generator_determinant": float(np.linalg.det(generator)),
        "generator_eigenvalue_0_real": float(np.real(eigenvalues[0])),
        "generator_eigenvalue_0_imag": float(np.imag(eigenvalues[0])),
        "generator_eigenvalue_1_real": float(np.real(eigenvalues[1])),
        "generator_eigenvalue_1_imag": float(np.imag(eigenvalues[1])),
        "omega_measured": omega_measured,
        "omega_expected_pi_over_P": omega_expected,
        "omega_relative_error": abs(omega_measured - omega_expected)
        / omega_expected,
        "global_generator_derivative_residual": global_generator_residual,
        "normalized_generator_square_error": normalized_square_error,
        "half_step_exponential_error_to_minus_I": half_turn_error,
        "full_step_exponential_error_to_I": full_turn_error,
        "mean_inferred_phase_speed": phase_speed_mean,
        "inferred_phase_speed_relative_error": abs(
            phase_speed_mean - omega_expected
        ) / omega_expected,
        "phase_speed_coefficient_of_variation": phase_speed_cv,
        "median_pointwise_generator_residual": float(
            np.nanmedian(pointwise_generator_residual)
        ),
        "maximum_pointwise_generator_residual": float(
            np.nanmax(pointwise_generator_residual)
        ),
        "continuous_generator_test_passed": supported,
        "classification": classification,
    }

    return CaseResult(
        q=q,
        token=q_token(q),
        p_line=p_line,
        p_oriented=p_oriented,
        anchor_count=anchor_count,
        phase_points=phase_points,
        matrix_dimension=rows,
        affine_centers=affine_centers,
        cycles=cycles,
        centered_cycles=centered_cycles,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_explained=pca_explained,
        scores=scores,
        harmonic_scores=harmonic_scores,
        inferred_phase=inferred_phase,
        phase_speed=phase_speed,
        pointwise_generator_residual=pointwise_generator_residual,
        generator=generator,
        generator_offset=generator_offset,
        generator_eigenvalues=eigenvalues,
        omega_measured=omega_measured,
        omega_expected=omega_expected,
        normalized_generator=normalized_generator,
        half_propagator=half_propagator,
        full_propagator=full_propagator,
        harmonic_r2=harmonic_r2,
        harmonic3_r2=harmonic3_r2,
        fundamental_fraction=fundamental_fraction,
        even_harmonic_fraction=even_fraction,
        harmonic_energy=harmonic_energy[:, : maximum_harmonic + 1],
        aligned_complex_modes=aligned_modes,
        mean_complex_mode=mean_mode,
        anatomy_amplitude=anatomy_amplitude,
        anatomy_phase=anatomy_phase,
        summary=summary,
    )


def write_summary_csv(path: Path, results: list[CaseResult]) -> None:
    keys = list(results[0].summary.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Q", *keys])
        for result in results:
            writer.writerow(
                [result.q, *[result.summary[key] for key in keys]]
            )


def write_anatomy_csv(path: Path, results: list[CaseResult]) -> None:
    all_rows = []
    for result in results:
        all_rows.extend(anatomy_rows(result))

    fieldnames = [
        "Q",
        "rank",
        "output_index",
        "output_coordinate",
        "input_index",
        "input_coordinate",
        "fundamental_amplitude",
        "fraction_of_mode_frobenius_norm",
        "fraction_of_maximum_entry",
        "aligned_phase_degrees",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def write_profiles_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "anchor_index",
                "phase_index",
                "phase_in_P_line_units",
                "support_rotation_offset",
                "PC1",
                "PC2",
                "harmonic_PC1",
                "harmonic_PC2",
                "inferred_phase_radians",
                "inferred_phase_speed",
                "pointwise_generator_residual",
            ]
        )
        for result in results:
            point_count = 2 * result.phase_points
            for anchor in range(result.anchor_count):
                for index in range(point_count):
                    writer.writerow(
                        [
                            result.q,
                            anchor,
                            index,
                            index / result.phase_points,
                            index * result.p_line / result.phase_points,
                            result.scores[anchor, index, 0],
                            result.scores[anchor, index, 1],
                            result.harmonic_scores[anchor, index, 0],
                            result.harmonic_scores[anchor, index, 1],
                            result.inferred_phase[anchor, index],
                            result.phase_speed[anchor, index],
                            result.pointwise_generator_residual[anchor, index],
                        ]
                    )


def write_harmonics_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "harmonic",
                "median_energy_fraction",
                "percentile_10",
                "percentile_90",
                "parity",
            ]
        )
        for result in results:
            for harmonic in range(1, result.harmonic_energy.shape[1]):
                values = result.harmonic_energy[:, harmonic]
                writer.writerow(
                    [
                        result.q,
                        harmonic,
                        np.nanmedian(values),
                        np.nanpercentile(values, 10.0),
                        np.nanpercentile(values, 90.0),
                        "odd" if harmonic % 2 else "even",
                    ]
                )


def write_comparison_csv(path: Path, results: list[CaseResult]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q_A",
                "Q_B",
                "principal_angle_1_degrees",
                "principal_angle_2_degrees",
                "minimum_subspace_cosine",
                "maximum_subspace_cosine",
                "omega_ratio_B_over_A",
            ]
        )
        for first_index in range(len(results)):
            for second_index in range(first_index + 1, len(results)):
                first = results[first_index]
                second = results[second_index]
                overlap = (
                    first.pca_components[:2]
                    @ second.pca_components[:2].T
                )
                singular_values = np.linalg.svd(
                    overlap,
                    compute_uv=False,
                )
                singular_values = np.clip(singular_values, 0.0, 1.0)
                angles = np.degrees(np.arccos(singular_values))
                writer.writerow(
                    [
                        first.q,
                        second.q,
                        angles[0],
                        angles[1],
                        np.min(singular_values),
                        np.max(singular_values),
                        second.omega_measured / first.omega_measured,
                    ]
                )


def write_raw_npz(path: Path, results: list[CaseResult]) -> None:
    payload: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
    }
    for result in results:
        prefix = f"q_{result.token}"
        payload[f"{prefix}_q"] = result.q
        payload[f"{prefix}_p_line"] = result.p_line
        payload[f"{prefix}_p_oriented"] = result.p_oriented
        payload[f"{prefix}_pca_mean"] = result.pca_mean
        payload[f"{prefix}_pca_components"] = result.pca_components
        payload[f"{prefix}_pca_explained"] = result.pca_explained
        payload[f"{prefix}_scores"] = result.scores
        payload[f"{prefix}_harmonic_scores"] = result.harmonic_scores
        payload[f"{prefix}_inferred_phase"] = result.inferred_phase
        payload[f"{prefix}_phase_speed"] = result.phase_speed
        payload[
            f"{prefix}_pointwise_generator_residual"
        ] = result.pointwise_generator_residual
        payload[f"{prefix}_generator"] = result.generator
        payload[f"{prefix}_generator_offset"] = result.generator_offset
        payload[
            f"{prefix}_generator_eigenvalues"
        ] = result.generator_eigenvalues
        payload[
            f"{prefix}_normalized_generator"
        ] = result.normalized_generator
        payload[f"{prefix}_half_propagator"] = result.half_propagator
        payload[f"{prefix}_full_propagator"] = result.full_propagator
        payload[f"{prefix}_harmonic_r2"] = result.harmonic_r2
        payload[f"{prefix}_harmonic3_r2"] = result.harmonic3_r2
        payload[
            f"{prefix}_fundamental_fraction"
        ] = result.fundamental_fraction
        payload[
            f"{prefix}_even_harmonic_fraction"
        ] = result.even_harmonic_fraction
        payload[f"{prefix}_harmonic_energy"] = result.harmonic_energy
        payload[
            f"{prefix}_aligned_complex_modes"
        ] = result.aligned_complex_modes
        payload[f"{prefix}_mean_complex_mode"] = result.mean_complex_mode
        payload[
            f"{prefix}_anatomy_amplitude"
        ] = result.anatomy_amplitude
        payload[f"{prefix}_anatomy_phase"] = result.anatomy_phase

    np.savez_compressed(path, **payload)


def save_phase_plane(path: Path, result: CaseResult) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))

    for anchor in range(result.anchor_count):
        axis.plot(
            result.scores[anchor, :, 0],
            result.scores[anchor, :, 1],
            linewidth=1.0,
            alpha=0.35,
        )

    middle = result.anchor_count // 2
    measured = result.scores[middle]
    harmonic = result.harmonic_scores[middle]

    axis.plot(
        measured[:, 0],
        measured[:, 1],
        linewidth=2.8,
        label="measured median-anchor stride",
    )
    axis.plot(
        harmonic[:, 0],
        harmonic[:, 1],
        linestyle="--",
        linewidth=2.0,
        label="first-harmonic fit",
    )
    axis.scatter(
        measured[0, 0],
        measured[0, 1],
        s=65,
        marker="o",
        label="0",
    )
    axis.scatter(
        measured[result.phase_points, 0],
        measured[result.phase_points, 1],
        s=65,
        marker="^",
        label="P_line",
    )

    axis.set_title(
        f"Continuous active-plane rocking loop, Q={result.q:g}\n"
        f"first-harmonic R²={np.nanmedian(result.harmonic_r2):.9f}"
    )
    axis.set_xlabel(
        f"PC1 ({100*result.pca_explained[0]:.5f}%)"
    )
    axis.set_ylabel(
        f"PC2 ({100*result.pca_explained[1]:.5f}%)"
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_phase_speed(path: Path, result: CaseResult) -> None:
    phase_units = (
        np.arange(2 * result.phase_points, dtype=float)
        / result.phase_points
    )
    valid_columns = np.any(np.isfinite(result.phase_speed), axis=0)
    phase_units = phase_units[valid_columns]
    speed_values = result.phase_speed[:, valid_columns]

    median = np.nanmedian(speed_values, axis=0)
    low = np.nanpercentile(speed_values, 10.0, axis=0)
    high = np.nanpercentile(speed_values, 90.0, axis=0)

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        phase_units,
        median,
        linewidth=2.2,
        label="measured geometric phase speed",
    )
    axis.fill_between(
        phase_units,
        low,
        high,
        alpha=0.18,
        label="10–90 percentile across anchors",
    )
    axis.axhline(
        result.omega_expected,
        linestyle="--",
        linewidth=1.5,
        label="pi / P_line",
    )
    axis.set_title(
        f"Rocking phase speed, Q={result.q:g}\n"
        f"coefficient of variation="
        f"{result.summary['phase_speed_coefficient_of_variation']:.3e}"
    )
    axis.set_xlabel("position within the two-stroke cycle [P_line]")
    axis.set_ylabel("phase speed [radians / support rotation]")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_harmonic_spectrum(path: Path, result: CaseResult) -> None:
    harmonics = np.arange(1, result.harmonic_energy.shape[1])
    median = np.nanmedian(result.harmonic_energy[:, 1:], axis=0)
    low = np.nanpercentile(result.harmonic_energy[:, 1:], 10.0, axis=0)
    high = np.nanpercentile(result.harmonic_energy[:, 1:], 90.0, axis=0)

    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        harmonics,
        median,
        marker="o",
        linewidth=1.8,
        label="median harmonic energy fraction",
    )
    axis.fill_between(harmonics, low, high, alpha=0.18)
    axis.set_yscale("log")
    axis.set_title(
        f"Fourier anatomy of the two-stroke B(s) loop, Q={result.q:g}\n"
        f"even-harmonic fraction="
        f"{np.nanmedian(result.even_harmonic_fraction):.3e}"
    )
    axis.set_xlabel("harmonic number over the 2 P_line cycle")
    axis.set_ylabel("fraction of non-DC matrix energy")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_anatomy_heatmap(path: Path, result: CaseResult) -> None:
    amplitude = result.anatomy_amplitude
    normalized = amplitude / max(float(np.max(amplitude)), 1e-300)

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, aspect="equal")
    figure.colorbar(
        image,
        ax=axis,
        label="fundamental amplitude / maximum entry",
    )

    axis.set_xticks(np.arange(len(REDUCED_NAMES)))
    axis.set_xticklabels(REDUCED_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(REDUCED_NAMES)))
    axis.set_yticklabels(REDUCED_NAMES)
    axis.set_xlabel("input perturbation coordinate")
    axis.set_ylabel("output derivative coordinate")
    axis.set_title(
        f"Anatomy of the fundamental rocking mode, Q={result.q:g}"
    )

    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_top_couplings(path: Path, result: CaseResult, count: int) -> None:
    rows = anatomy_rows(result)[:count]
    labels = [
        f"d({row['output_coordinate']})/ds ← δ{row['input_coordinate']}"
        for row in rows
    ]
    values = [
        float(row["fraction_of_maximum_entry"])
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(11, 8))
    y = np.arange(len(rows))
    axis.barh(y, values)
    axis.set_yticks(y)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("fundamental amplitude / maximum matrix entry")
    axis.set_title(
        f"Strongest matrix couplings in the rocking mode, Q={result.q:g}"
    )
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_generator_matrix(path: Path, result: CaseResult) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(result.normalized_generator, aspect="equal")
    figure.colorbar(image, ax=axis, label="J = G / omega")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{result.normalized_generator[row, column]:+.6f}",
                ha="center",
                va="center",
            )

    axis.set_xticks([0, 1], ["PC1", "PC2"])
    axis.set_yticks([0, 1], ["dPC1/ds", "dPC2/ds"])
    axis.set_title(
        f"Normalized continuous generator, Q={result.q:g}\n"
        f"||J²+I||={result.summary['normalized_generator_square_error']:.3e}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def build_report(
    source: Path,
    results: list[CaseResult],
    outputs: dict[str, Path],
    comparison_path: Path,
) -> str:
    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "CONTINUOUS ROCKING GENERATOR AUDIT",
        "==================================",
        "",
        f"Source raw archive: {source.resolve()}",
        "",
        "The analysis uses the actual centered reduced 9x9 matrices B(s) over",
        "two consecutive P_line strokes. No nonlinear orbit is regenerated.",
        "",
        "Tested model:",
        "",
        "    R(s) = B(s) - C",
        "    dR/ds ~= G R",
        "    R(s) ~= A cos(phi) + D sin(phi)",
        "    phi(s + P_line) ~= phi(s) + pi",
        "",
    ]

    for result in results:
        s = result.summary
        lines.extend(
            [
                f"Q={result.q:g}",
                f"  P_line                                = {result.p_line:.12f}",
                f"  P_oriented                            = {result.p_oriented:.12f}",
                f"  active-plane variance                 = {s['active_plane_variance']:.12e}",
                f"  PC1 explained                         = {s['pc1_explained']:.12e}",
                f"  PC2 explained                         = {s['pc2_explained']:.12e}",
                f"  median first-harmonic R^2             = {s['median_first_harmonic_R2']:.12e}",
                f"  median first-three-harmonic R^2       = {s['median_first_three_harmonics_R2']:.12e}",
                f"  median fundamental energy fraction    = {s['median_fundamental_energy_fraction']:.12e}",
                f"  median even-harmonic energy fraction  = {s['median_even_harmonic_energy_fraction']:.12e}",
                "",
                "  fitted active-plane generator G",
                f"    [{result.generator[0,0]:+.12e}  {result.generator[0,1]:+.12e}]",
                f"    [{result.generator[1,0]:+.12e}  {result.generator[1,1]:+.12e}]",
                f"  generator offset norm                 = {s['generator_offset_norm']:.12e}",
                f"  generator trace                       = {s['generator_trace']:+.12e}",
                f"  generator determinant                 = {s['generator_determinant']:+.12e}",
                f"  measured omega                        = {s['omega_measured']:.12e}",
                f"  expected pi/P_line                    = {s['omega_expected_pi_over_P']:.12e}",
                f"  omega relative error                  = {s['omega_relative_error']:.12e}",
                f"  global derivative-fit residual        = {s['global_generator_derivative_residual']:.12e}",
                f"  normalized J^2 + I error              = {s['normalized_generator_square_error']:.12e}",
                f"  exp(P_line G) distance to -I          = {s['half_step_exponential_error_to_minus_I']:.12e}",
                f"  exp(2 P_line G) distance to I         = {s['full_step_exponential_error_to_I']:.12e}",
                f"  mean inferred phase speed             = {s['mean_inferred_phase_speed']:.12e}",
                f"  inferred phase-speed relative error   = {s['inferred_phase_speed_relative_error']:.12e}",
                f"  phase-speed coefficient of variation  = {s['phase_speed_coefficient_of_variation']:.12e}",
                f"  continuous-generator test passed      = {s['continuous_generator_test_passed']}",
                f"  classification                        = {s['classification']}",
                "",
                "  strongest fundamental matrix couplings",
            ]
        )

        for row in anatomy_rows(result)[:10]:
            lines.append(
                "    "
                f"{int(row['rank']):>2}. "
                f"d({row['output_coordinate']})/ds <- "
                f"delta {row['input_coordinate']}: "
                f"amplitude={float(row['fundamental_amplitude']):.12e}, "
                f"relative={float(row['fraction_of_maximum_entry']):.6f}"
            )
        lines.append("")

    lines.extend(
        [
            "FILES",
            "-----",
            f"summary CSV       = {outputs['summary'].resolve()}",
            f"anatomy CSV       = {outputs['anatomy'].resolve()}",
            f"profiles CSV      = {outputs['profiles'].resolve()}",
            f"harmonics CSV     = {outputs['harmonics'].resolve()}",
            f"comparison CSV    = {comparison_path.resolve()}",
            f"raw NPZ            = {outputs['raw'].resolve()}",
        ]
    )

    for result in results:
        token = result.token
        lines.extend(
            [
                f"Q={result.q:g} phase plane PNG    = "
                f"{outputs[f'{token}_phase_plane'].resolve()}",
                f"Q={result.q:g} phase speed PNG    = "
                f"{outputs[f'{token}_phase_speed'].resolve()}",
                f"Q={result.q:g} harmonics PNG      = "
                f"{outputs[f'{token}_harmonics'].resolve()}",
                f"Q={result.q:g} anatomy PNG        = "
                f"{outputs[f'{token}_anatomy'].resolve()}",
                f"Q={result.q:g} top couplings PNG  = "
                f"{outputs[f'{token}_top'].resolve()}",
                f"Q={result.q:g} generator PNG      = "
                f"{outputs[f'{token}_generator'].resolve()}",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("support_stride_involution_audit_raw.npz"),
        help="raw NPZ produced by the stride-involution audit",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=float,
        help="optional Q value to analyze; repeat for multiple cases",
    )
    parser.add_argument(
        "--maximum-harmonic",
        type=int,
        default=12,
        help="highest Fourier harmonic exported and plotted",
    )
    parser.add_argument(
        "--top-couplings",
        type=int,
        default=14,
        help="number of matrix entries shown in the coupling ranking plot",
    )
    parser.add_argument(
        "--output-prefix",
        default="support_stride_generator_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input archive not found: {args.input}")
    if args.maximum_harmonic < 3:
        raise ValueError("--maximum-harmonic must be at least 3.")
    if args.top_couplings < 1:
        raise ValueError("--top-couplings must be positive.")

    archive = np.load(args.input)
    prefixes = case_prefixes(archive)
    if not prefixes:
        raise KeyError("No Q cases were found in the raw archive.")

    if args.case:
        requested = np.asarray(args.case, dtype=float)
        prefixes = [
            prefix
            for prefix in prefixes
            if np.any(
                np.isclose(
                    float(archive[f"{prefix}_q"]),
                    requested,
                    rtol=0.0,
                    atol=1e-8,
                )
            )
        ]
        if not prefixes:
            raise ValueError("None of the requested Q values exist in the archive.")

    print(f"Script version = {SCRIPT_VERSION}")
    print(f"Input archive  = {args.input.resolve()}")
    print("Cases:")
    for prefix in prefixes:
        print(f"  Q={float(archive[f'{prefix}_q']):g}")

    results = []
    for prefix in prefixes:
        q = float(archive[f"{prefix}_q"])
        print("=" * 78)
        print(f"Continuous rocking generator audit for Q={q:g}")
        result = analyze_case(
            archive,
            prefix,
            maximum_harmonic=args.maximum_harmonic,
        )
        results.append(result)
        print(
            f"  active-plane variance       = "
            f"{result.summary['active_plane_variance']:.12e}"
        )
        print(
            f"  fundamental energy fraction = "
            f"{result.summary['median_fundamental_energy_fraction']:.12e}"
        )
        print(
            f"  even-harmonic fraction      = "
            f"{result.summary['median_even_harmonic_energy_fraction']:.12e}"
        )
        print(
            f"  omega measured / expected   = "
            f"{result.omega_measured:.12e} / {result.omega_expected:.12e}"
        )
        print(
            f"  exp(P G) error to -I        = "
            f"{result.summary['half_step_exponential_error_to_minus_I']:.12e}"
        )
        print(
            f"  classification              = "
            f"{result.summary['classification']}"
        )

    prefix_path = Path(args.output_prefix)
    outputs: dict[str, Path] = {
        "summary": prefix_path.with_name(prefix_path.name + "_summary.csv"),
        "anatomy": prefix_path.with_name(prefix_path.name + "_anatomy.csv"),
        "profiles": prefix_path.with_name(prefix_path.name + "_profiles.csv"),
        "harmonics": prefix_path.with_name(prefix_path.name + "_harmonics.csv"),
        "raw": prefix_path.with_name(prefix_path.name + "_raw.npz"),
        "report": prefix_path.with_name(prefix_path.name + "_report.txt"),
    }
    comparison_path = prefix_path.with_name(
        prefix_path.name + "_comparison.csv"
    )

    write_summary_csv(outputs["summary"], results)
    write_anatomy_csv(outputs["anatomy"], results)
    write_profiles_csv(outputs["profiles"], results)
    write_harmonics_csv(outputs["harmonics"], results)
    write_comparison_csv(comparison_path, results)
    write_raw_npz(outputs["raw"], results)

    for result in results:
        token = result.token
        outputs[f"{token}_phase_plane"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_phase_plane.png"
        )
        outputs[f"{token}_phase_speed"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_phase_speed.png"
        )
        outputs[f"{token}_harmonics"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_harmonics.png"
        )
        outputs[f"{token}_anatomy"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_anatomy.png"
        )
        outputs[f"{token}_top"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_top_couplings.png"
        )
        outputs[f"{token}_generator"] = prefix_path.with_name(
            prefix_path.name + f"_Q{token}_generator.png"
        )

        save_phase_plane(outputs[f"{token}_phase_plane"], result)
        save_phase_speed(outputs[f"{token}_phase_speed"], result)
        save_harmonic_spectrum(outputs[f"{token}_harmonics"], result)
        save_anatomy_heatmap(outputs[f"{token}_anatomy"], result)
        save_top_couplings(
            outputs[f"{token}_top"],
            result,
            args.top_couplings,
        )
        save_generator_matrix(outputs[f"{token}_generator"], result)

    report = build_report(
        args.input,
        results,
        outputs,
        comparison_path,
    )
    outputs["report"].write_text(report, encoding="utf-8")

    print()
    print(report)


if __name__ == "__main__":
    main()
