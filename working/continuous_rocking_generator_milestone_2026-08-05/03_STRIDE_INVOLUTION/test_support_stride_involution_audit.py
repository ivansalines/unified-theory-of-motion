#!/usr/bin/env python3
"""Alternating-stride / antipodal-involution audit for the support system.

The previous direct B(s) export showed that the reduced 9x9 tangent matrices
live almost entirely in a two-dimensional affine plane.  It also showed that
one P_line interval does not close the centered matrix loop: it appears to
move approximately to the antipodal point, while two P_line intervals close
the full stride.

This script tests that statement directly, without normalizing by the large
static Frobenius norm of B.

For each Q and each anchor it samples three consecutive P_line intervals and
measures

    R(s + P_line)  ~= -R(s)
    R(s + 2P_line) ~=  R(s)

where

    R(s) = B(s) - C

and C is the least-squares affine center obtained from the paired half-step
midpoints 0.5 * [B(s) + B(s + P_line)].

It also measures the tangent-direction return

    w(s + P_line)  ~= -w(s)
    w(s + 2P_line) ~=  w(s)

and fits the best orthogonal map between the first and second matrix strokes
inside the direct two-dimensional PCA plane.  A map close to -I is a genuine
half-turn in the active coefficient geometry; det=-1 would instead indicate
a reflection.

No nonlinear orbit is regenerated.  Existing orbit caches are reused through
the validated functions in test_support_mobius_monodromy_audit.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_VERSION = "2026-08-05-stride-involution-audit-v1"


@dataclass
class AnchorAudit:
    q: float
    anchor_index: int
    anchor_s: float
    p_line: float
    affine_center: np.ndarray
    cycle_mean: np.ndarray
    amplitude_scale: float
    phase: np.ndarray
    matrices_0: np.ndarray
    matrices_p: np.ndarray
    matrices_2p: np.ndarray
    residual_0: np.ndarray
    residual_p: np.ndarray
    residual_2p: np.ndarray
    antipodal_cosine: np.ndarray
    antipodal_residual: np.ndarray
    antipodal_amplitude_ratio: np.ndarray
    full_return_cosine: np.ndarray
    full_return_residual: np.ndarray
    full_return_amplitude_ratio: np.ndarray
    midpoint_drift: np.ndarray
    tangent_signed_p: np.ndarray
    tangent_signed_2p: np.ndarray
    two_stride_matrices: np.ndarray


@dataclass
class CaseAudit:
    q: float
    source: Path
    p_line: float
    p_oriented: float
    anchors: list[AnchorAudit]
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_singular_values: np.ndarray
    pca_explained: np.ndarray
    pca_scores: np.ndarray
    pca_metadata: np.ndarray
    procrustes_matrix_2d: np.ndarray
    procrustes_determinant: float
    procrustes_trace: float
    procrustes_distance_to_minus_identity: float
    procrustes_angle_radians: float
    summary: dict[str, float | str | bool]


def parse_case(raw: str) -> tuple[float, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--case must use Q=orbit_cache.npz")
    q_text, path_text = raw.split("=", 1)
    try:
        q = float(q_text.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid Q value: {q_text!r}") from exc
    path = Path(path_text.strip())
    if not str(path):
        raise argparse.ArgumentTypeError("Orbit path is empty.")
    return q, path


def load_module(path: Path) -> ModuleType:
    """Import the audit script with Python 3.13 dataclass compatibility."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audit script not found: {path}")

    module_name = "mobius_audit_stride_runtime"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import audit script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def metric_cosine(a: np.ndarray, b: np.ndarray, metric: np.ndarray) -> float:
    numerator = float(a @ metric @ b)
    norm_a = math.sqrt(max(float(a @ metric @ a), 0.0))
    norm_b = math.sqrt(max(float(b @ metric @ b), 0.0))
    denominator = norm_a * norm_b
    if denominator <= 1e-300:
        return float("nan")
    return numerator / denominator


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ni,ni->n", a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denominator = norm_a * norm_b
    output = np.full(a.shape[0], np.nan, dtype=float)
    valid = denominator > 1e-300
    output[valid] = numerator[valid] / denominator[valid]
    return output


def symmetric_relative_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ||a-b|| / sqrt(||a||^2 + ||b||^2), row by row."""
    numerator = np.linalg.norm(a - b, axis=1)
    denominator = np.sqrt(
        np.sum(a * a, axis=1) + np.sum(b * b, axis=1)
    )
    output = np.full(a.shape[0], np.nan, dtype=float)
    valid = denominator > 1e-300
    output[valid] = numerator[valid] / denominator[valid]
    return output


def amplitude_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return ||b||/||a|| row by row."""
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    output = np.full(a.shape[0], np.nan, dtype=float)
    valid = norm_a > 1e-300
    output[valid] = norm_b[valid] / norm_a[valid]
    return output


def run_pca(vectors: np.ndarray):
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    _, singular_values, components = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    scores = centered @ components.T
    variance = singular_values**2 / max(vectors.shape[0] - 1, 1)
    total = float(np.sum(variance))
    explained = variance / total if total > 0.0 else np.zeros_like(variance)
    return mean, components, singular_values, explained, scores


def orthogonal_procrustes_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Find orthogonal R minimizing ||x @ R - y||_F for row vectors."""
    cross = x.T @ y
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def estimate_periods(audit: ModuleType, tail, args) -> tuple[float, float]:
    lag, signed, line = audit.recurrence_curves(
        tail.reduced_direction,
        tail.metric,
        samples_per_rotation=args.samples_per_rotation,
        max_lag_rotations=args.max_recurrence_lag,
    )
    p_line, _ = audit.estimate_line_period(
        lag,
        line,
        min_period=args.min_line_period,
        max_period=args.max_line_period,
    )
    p_oriented = audit.estimate_oriented_period(lag, signed, p_line)
    return float(p_line), float(p_oriented)


def choose_anchors(
    tail,
    *,
    p_line: float,
    anchor_count: int,
    anchor_span_rotations: float,
) -> np.ndarray:
    # Three P_line intervals are needed so that every first-stroke phase has
    # partners at +P_line and +2P_line.
    available_end = float(tail.s[-1] - 3.0 * p_line)
    available_start = max(
        float(tail.s[0]),
        available_end - anchor_span_rotations,
    )
    if available_end <= available_start:
        raise RuntimeError("Tail is too short for a three-stroke audit.")
    return np.linspace(available_start, available_end, anchor_count)


def sample_anchor(
    audit: ModuleType,
    tail,
    state_spline,
    *,
    q: float,
    anchor_index: int,
    anchor_s: float,
    p_line: float,
    phase_points: int,
    kappa0: float,
    params,
) -> AnchorAudit:
    # endpoint=False prevents duplicate phase points inside each stroke.
    phase = np.arange(phase_points, dtype=float) / phase_points
    offsets = np.arange(3 * phase_points + 1, dtype=float) / phase_points
    s_values = anchor_s + offsets * p_line

    matrices = np.empty(
        (
            s_values.size,
            audit.REDUCED_DIMENSION,
            audit.REDUCED_DIMENSION,
        ),
        dtype=float,
    )

    for index, s_value in enumerate(s_values):
        state = np.asarray(state_spline(s_value), dtype=float)
        matrices[index] = audit.reduced_jacobian_rotation(
            state,
            kappa0=kappa0,
            params=params,
            embedding_name="symmetric",
        )

    if not np.all(np.isfinite(matrices)):
        raise FloatingPointError(
            f"Q={q:g}, anchor={anchor_index}: non-finite B(s)."
        )

    n = phase_points
    b0 = matrices[0:n]
    bp = matrices[n : 2 * n]
    b2p = matrices[2 * n : 3 * n]

    # Optimal affine center for the hypothesis B(s+P) = 2C - B(s).
    pair_midpoints = 0.5 * (b0 + bp)
    affine_center = np.mean(pair_midpoints, axis=0)
    cycle_mean = np.mean(matrices[0 : 2 * n], axis=0)

    r0 = (b0 - affine_center).reshape(n, -1)
    rp = (bp - affine_center).reshape(n, -1)
    r2p = (b2p - affine_center).reshape(n, -1)

    amplitude_scale = float(
        np.sqrt(0.5 * np.mean(np.sum(r0 * r0, axis=1) + np.sum(rp * rp, axis=1)))
    )
    if amplitude_scale <= 0.0 or not np.isfinite(amplitude_scale):
        raise FloatingPointError("Invalid centered matrix amplitude scale.")

    antipodal_cosine = row_cosine(rp, -r0)
    antipodal_residual = symmetric_relative_residual(rp, -r0)
    antipodal_amplitude_ratio = amplitude_ratio(r0, rp)

    full_return_cosine = row_cosine(r2p, r0)
    full_return_residual = symmetric_relative_residual(r2p, r0)
    full_return_amplitude_ratio = amplitude_ratio(r0, r2p)

    midpoint_drift = np.linalg.norm(
        (pair_midpoints - affine_center).reshape(n, -1),
        axis=1,
    ) / amplitude_scale

    tangent_signed_p = np.empty(n, dtype=float)
    tangent_signed_2p = np.empty(n, dtype=float)

    for index, phi in enumerate(phase):
        s0 = anchor_s + phi * p_line
        u0 = audit.interpolate_direction(tail, s0)
        up = audit.interpolate_direction(tail, s0 + p_line)
        u2p = audit.interpolate_direction(tail, s0 + 2.0 * p_line)
        tangent_signed_p[index] = metric_cosine(u0, up, tail.metric)
        tangent_signed_2p[index] = metric_cosine(u0, u2p, tail.metric)

    return AnchorAudit(
        q=q,
        anchor_index=anchor_index,
        anchor_s=float(anchor_s),
        p_line=p_line,
        affine_center=affine_center,
        cycle_mean=cycle_mean,
        amplitude_scale=amplitude_scale,
        phase=phase,
        matrices_0=b0,
        matrices_p=bp,
        matrices_2p=b2p,
        residual_0=r0,
        residual_p=rp,
        residual_2p=r2p,
        antipodal_cosine=antipodal_cosine,
        antipodal_residual=antipodal_residual,
        antipodal_amplitude_ratio=antipodal_amplitude_ratio,
        full_return_cosine=full_return_cosine,
        full_return_residual=full_return_residual,
        full_return_amplitude_ratio=full_return_amplitude_ratio,
        midpoint_drift=midpoint_drift,
        tangent_signed_p=tangent_signed_p,
        tangent_signed_2p=tangent_signed_2p,
        two_stride_matrices=matrices[0 : 2 * n + 1],
    )


def pooled_values(anchors: list[AnchorAudit], name: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray(getattr(anchor, name), dtype=float) for anchor in anchors]
    )


def analyze_case(
    audit: ModuleType,
    *,
    q: float,
    orbit_path: Path,
    args,
) -> CaseAudit:
    params = audit.ModelParameters(gamma_rho=args.gamma_rho)
    orbit = audit.load_orbit(q, orbit_path)
    tail = audit.collect_uniform_tail(
        orbit,
        tail_periods=args.tail_periods,
        steps_per_period=args.steps_per_period,
        samples_per_rotation=args.samples_per_rotation,
        initial_x_perturbation=args.initial_x_perturbation,
        params=params,
        progress_every=args.progress_every,
    )

    p_line, p_oriented = estimate_periods(audit, tail, args)
    state_spline = audit.build_state_spline(tail)
    kappa0 = (55.0 / 6.0) * (10.0 / q)

    anchor_values = choose_anchors(
        tail,
        p_line=p_line,
        anchor_count=args.anchor_count,
        anchor_span_rotations=args.anchor_span_rotations,
    )

    anchors: list[AnchorAudit] = []
    for anchor_index, anchor_s in enumerate(anchor_values):
        record = sample_anchor(
            audit,
            tail,
            state_spline,
            q=q,
            anchor_index=anchor_index,
            anchor_s=float(anchor_s),
            p_line=p_line,
            phase_points=args.phase_points,
            kappa0=kappa0,
            params=params,
        )
        anchors.append(record)
        print(
            f"  Q={q:g}: anchor {anchor_index + 1}/{args.anchor_count}, "
            f"anti residual={np.nanmedian(record.antipodal_residual):.6e}, "
            f"2P residual={np.nanmedian(record.full_return_residual):.6e}"
        )

    # Direct PCA of all actual B(s) matrices over two strokes.
    vectors = []
    metadata = []
    for anchor in anchors:
        flattened = anchor.two_stride_matrices.reshape(
            anchor.two_stride_matrices.shape[0],
            -1,
        )
        vectors.append(flattened)
        for index in range(flattened.shape[0]):
            metadata.append(
                [
                    anchor.anchor_index,
                    anchor.anchor_s,
                    index,
                    index / args.phase_points,
                    anchor.anchor_s + (index / args.phase_points) * p_line,
                ]
            )

    pooled = np.vstack(vectors)
    pca_metadata = np.asarray(metadata, dtype=float)
    (
        pca_mean,
        pca_components,
        pca_singular_values,
        pca_explained,
        pca_scores,
    ) = run_pca(pooled)

    # Fit the half-step map in the measured active PCA plane.
    first_scores = []
    second_scores = []
    points_per_two_stride = 2 * args.phase_points + 1
    for anchor_index in range(len(anchors)):
        start = anchor_index * points_per_two_stride
        score = pca_scores[start : start + points_per_two_stride, :2]
        first_scores.append(score[0 : args.phase_points])
        second_scores.append(
            score[args.phase_points : 2 * args.phase_points]
        )

    x = np.vstack(first_scores)
    y = np.vstack(second_scores)
    # Remove any tiny pooled offset before the orthogonal fit.
    x = x - np.mean(x, axis=0)
    y = y - np.mean(y, axis=0)
    procrustes = orthogonal_procrustes_rows(x, y)
    determinant = float(np.linalg.det(procrustes))
    trace = float(np.trace(procrustes))
    distance_minus_identity = float(
        np.linalg.norm(procrustes + np.eye(2), ord="fro")
        / np.linalg.norm(np.eye(2), ord="fro")
    )
    if determinant > 0.0:
        angle = float(math.atan2(procrustes[1, 0], procrustes[0, 0]))
        if angle < 0.0:
            angle += 2.0 * math.pi
    else:
        angle = float("nan")

    anti_cos = pooled_values(anchors, "antipodal_cosine")
    anti_res = pooled_values(anchors, "antipodal_residual")
    anti_ratio = pooled_values(anchors, "antipodal_amplitude_ratio")
    full_cos = pooled_values(anchors, "full_return_cosine")
    full_res = pooled_values(anchors, "full_return_residual")
    full_ratio = pooled_values(anchors, "full_return_amplitude_ratio")
    midpoint = pooled_values(anchors, "midpoint_drift")
    tangent_p = pooled_values(anchors, "tangent_signed_p")
    tangent_2p = pooled_values(anchors, "tangent_signed_2p")

    center_difference = np.asarray(
        [
            np.linalg.norm(anchor.affine_center - anchor.cycle_mean, ord="fro")
            / anchor.amplitude_scale
            for anchor in anchors
        ],
        dtype=float,
    )

    anti_passed = bool(
        np.nanmedian(anti_cos) >= args.minimum_cosine
        and np.nanmedian(anti_res) <= args.maximum_residual
    )
    full_passed = bool(
        np.nanmedian(full_cos) >= args.minimum_cosine
        and np.nanmedian(full_res) <= args.maximum_residual
    )
    tangent_passed = bool(
        np.nanmedian(tangent_p) <= -args.minimum_tangent_overlap
        and np.nanmedian(tangent_2p) >= args.minimum_tangent_overlap
    )
    active_plane_passed = bool(
        determinant > 0.0
        and distance_minus_identity <= args.maximum_map_distance
    )

    if anti_passed and full_passed and tangent_passed and active_plane_passed:
        classification = (
            "Alternating two-stroke affine involution supported: the active "
            "matrix geometry executes an approximately antipodal half-turn at "
            "P_line and closes at 2 P_line, while the tangent orientation "
            "reverses and then returns."
        )
    elif anti_passed and full_passed and tangent_passed:
        classification = (
            "Antipodal two-stroke recurrence supported, but the fitted active-"
            "plane map is not sufficiently close to -I under the selected "
            "threshold. Inspect whether the half-step is a reflection or an "
            "oblique orthogonal map."
        )
    elif full_passed and tangent_passed:
        classification = (
            "Two-stroke closure and tangent alternation are supported, but the "
            "one-stroke matrix map is not a sufficiently accurate affine "
            "antipode under the selected threshold."
        )
    else:
        classification = (
            "Standing-stride hypothesis not established under the selected "
            "thresholds; inspect the phase-resolved residuals and active-plane "
            "map."
        )

    cumulative = np.cumsum(pca_explained)
    summary: dict[str, float | str | bool] = {
        "p_line": p_line,
        "p_oriented": p_oriented,
        "median_antipodal_cosine": float(np.nanmedian(anti_cos)),
        "minimum_antipodal_cosine": float(np.nanmin(anti_cos)),
        "median_antipodal_residual": float(np.nanmedian(anti_res)),
        "maximum_antipodal_residual": float(np.nanmax(anti_res)),
        "median_antipodal_amplitude_ratio": float(np.nanmedian(anti_ratio)),
        "median_full_return_cosine": float(np.nanmedian(full_cos)),
        "minimum_full_return_cosine": float(np.nanmin(full_cos)),
        "median_full_return_residual": float(np.nanmedian(full_res)),
        "maximum_full_return_residual": float(np.nanmax(full_res)),
        "median_full_return_amplitude_ratio": float(np.nanmedian(full_ratio)),
        "median_pair_midpoint_drift": float(np.nanmedian(midpoint)),
        "maximum_pair_midpoint_drift": float(np.nanmax(midpoint)),
        "median_center_difference": float(np.nanmedian(center_difference)),
        "median_tangent_signed_p": float(np.nanmedian(tangent_p)),
        "median_tangent_signed_2p": float(np.nanmedian(tangent_2p)),
        "pc1_explained": float(pca_explained[0]),
        "pc2_explained": float(pca_explained[1]),
        "pc3_explained": float(pca_explained[2]),
        "pc1_pc2_cumulative": float(cumulative[1]),
        "pc1_pc2_pc3_cumulative": float(cumulative[2]),
        "procrustes_r00": float(procrustes[0, 0]),
        "procrustes_r01": float(procrustes[0, 1]),
        "procrustes_r10": float(procrustes[1, 0]),
        "procrustes_r11": float(procrustes[1, 1]),
        "procrustes_determinant": determinant,
        "procrustes_trace": trace,
        "procrustes_distance_to_minus_identity": distance_minus_identity,
        "procrustes_angle_radians": angle,
        "procrustes_angle_degrees": math.degrees(angle) if np.isfinite(angle) else float("nan"),
        "antipodal_test_passed": anti_passed,
        "full_return_test_passed": full_passed,
        "tangent_alternation_passed": tangent_passed,
        "active_plane_half_turn_passed": active_plane_passed,
        "classification": classification,
    }

    return CaseAudit(
        q=q,
        source=orbit_path,
        p_line=p_line,
        p_oriented=p_oriented,
        anchors=anchors,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_singular_values=pca_singular_values,
        pca_explained=pca_explained,
        pca_scores=pca_scores,
        pca_metadata=pca_metadata,
        procrustes_matrix_2d=procrustes,
        procrustes_determinant=determinant,
        procrustes_trace=trace,
        procrustes_distance_to_minus_identity=distance_minus_identity,
        procrustes_angle_radians=angle,
        summary=summary,
    )


def q_token(q: float) -> str:
    return f"{q:.8f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def save_embedding_plot(path: Path, result: CaseAudit, phase_points: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 8))
    points = 2 * phase_points + 1

    for anchor_index in range(len(result.anchors)):
        start = anchor_index * points
        score = result.pca_scores[start : start + points]
        axis.plot(score[:, 0], score[:, 1], alpha=0.45, linewidth=1.1)

    middle = len(result.anchors) // 2
    start = middle * points
    score = result.pca_scores[start : start + points]
    axis.plot(
        score[:, 0],
        score[:, 1],
        linewidth=3.0,
        label="median-anchor two-stroke loop",
    )
    axis.scatter(score[0, 0], score[0, 1], s=70, marker="o", label="0")
    axis.scatter(
        score[phase_points, 0],
        score[phase_points, 1],
        s=70,
        marker="^",
        label="P_line",
    )
    axis.scatter(score[-1, 0], score[-1, 1], s=70, marker="x", label="2 P_line")

    exp = result.pca_explained
    axis.set_title(
        f"Direct two-stroke B(s) loop, Q={result.q:g}\n"
        f"active-plane variance={100*np.sum(exp[:2]):.8f}%"
    )
    axis.set_xlabel(f"PC1 ({100*exp[0]:.4f}%)")
    axis.set_ylabel(f"PC2 ({100*exp[1]:.4f}%)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def percentile_profile(result: CaseAudit, name: str):
    values = np.stack(
        [np.asarray(getattr(anchor, name), dtype=float) for anchor in result.anchors],
        axis=0,
    )
    return (
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 10.0, axis=0),
        np.nanpercentile(values, 90.0, axis=0),
    )


def save_residual_plot(path: Path, result: CaseAudit) -> None:
    phase = result.anchors[0].phase
    figure, axis = plt.subplots(figsize=(11, 7))

    for name, label in [
        ("antipodal_residual", "P_line antipodal residual"),
        ("full_return_residual", "2 P_line closure residual"),
        ("midpoint_drift", "pair-midpoint drift"),
    ]:
        median, low, high = percentile_profile(result, name)
        axis.plot(phase, median, linewidth=2.0, label=label)
        axis.fill_between(phase, low, high, alpha=0.16)

    axis.set_title(f"Centered stride residuals, Q={result.q:g}")
    axis.set_xlabel("phase within the first P_line stroke")
    axis.set_ylabel("dimensionless centered residual")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_overlap_plot(path: Path, result: CaseAudit) -> None:
    phase = result.anchors[0].phase
    figure, axis = plt.subplots(figsize=(11, 7))

    for name, label in [
        ("antipodal_cosine", "matrix: B(s+P)-C vs -(B(s)-C)"),
        ("full_return_cosine", "matrix: B(s+2P)-C vs B(s)-C"),
        ("tangent_signed_p", "tangent signed return at P_line"),
        ("tangent_signed_2p", "tangent signed return at 2 P_line"),
    ]:
        median, _, _ = percentile_profile(result, name)
        axis.plot(phase, median, linewidth=2.0, label=label)

    axis.axhline(0.0, linewidth=1.0)
    axis.axhline(1.0, linewidth=0.8, linestyle=":")
    axis.axhline(-1.0, linewidth=0.8, linestyle=":")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(f"Alternating matrix and tangent returns, Q={result.q:g}")
    axis.set_xlabel("phase within the first P_line stroke")
    axis.set_ylabel("cosine / signed overlap")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_map_plot(path: Path, result: CaseAudit, phase_points: int) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    points = 2 * phase_points + 1

    first = []
    second = []
    for anchor_index in range(len(result.anchors)):
        start = anchor_index * points
        score = result.pca_scores[start : start + points, :2]
        first.append(score[0:phase_points])
        second.append(score[phase_points : 2 * phase_points])

    x = np.vstack(first)
    y = np.vstack(second)
    sample_indices = np.linspace(0, x.shape[0] - 1, min(80, x.shape[0]), dtype=int)
    axis.scatter(x[:, 0], x[:, 1], s=7, alpha=0.22, label="first stroke")
    axis.scatter(y[:, 0], y[:, 1], s=7, alpha=0.22, label="second stroke")

    for index in sample_indices:
        axis.plot(
            [x[index, 0], y[index, 0]],
            [x[index, 1], y[index, 1]],
            linewidth=0.5,
            alpha=0.25,
        )

    axis.set_title(
        f"Measured half-step map in the active B(s) plane, Q={result.q:g}\n"
        f"det={result.procrustes_determinant:+.6f}, "
        f"distance to -I={result.procrustes_distance_to_minus_identity:.3e}"
    )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_profiles_csv(path: Path, results: list[CaseAudit]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Q",
                "anchor_index",
                "anchor_s",
                "phase_index",
                "stroke_phase",
                "antipodal_cosine",
                "antipodal_residual",
                "antipodal_amplitude_ratio",
                "full_return_cosine",
                "full_return_residual",
                "full_return_amplitude_ratio",
                "pair_midpoint_drift",
                "tangent_signed_return_P",
                "tangent_signed_return_2P",
            ]
        )
        for result in results:
            for anchor in result.anchors:
                for index, phase in enumerate(anchor.phase):
                    writer.writerow(
                        [
                            result.q,
                            anchor.anchor_index,
                            anchor.anchor_s,
                            index,
                            phase,
                            anchor.antipodal_cosine[index],
                            anchor.antipodal_residual[index],
                            anchor.antipodal_amplitude_ratio[index],
                            anchor.full_return_cosine[index],
                            anchor.full_return_residual[index],
                            anchor.full_return_amplitude_ratio[index],
                            anchor.midpoint_drift[index],
                            anchor.tangent_signed_p[index],
                            anchor.tangent_signed_2p[index],
                        ]
                    )


def save_summary_csv(path: Path, results: list[CaseAudit]) -> None:
    keys = list(results[0].summary.keys())
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Q", "source", *keys])
        for result in results:
            writer.writerow(
                [result.q, str(result.source.resolve()), *[result.summary[key] for key in keys]]
            )


def save_raw_npz(path: Path, results: list[CaseAudit]) -> None:
    payload: dict[str, object] = {"script_version": SCRIPT_VERSION}
    for result in results:
        prefix = f"q_{q_token(result.q)}"
        payload[f"{prefix}_q"] = result.q
        payload[f"{prefix}_p_line"] = result.p_line
        payload[f"{prefix}_p_oriented"] = result.p_oriented
        payload[f"{prefix}_pca_mean"] = result.pca_mean
        payload[f"{prefix}_pca_components"] = result.pca_components
        payload[f"{prefix}_pca_singular_values"] = result.pca_singular_values
        payload[f"{prefix}_pca_explained"] = result.pca_explained
        payload[f"{prefix}_pca_scores"] = result.pca_scores
        payload[f"{prefix}_pca_metadata"] = result.pca_metadata
        payload[f"{prefix}_procrustes_matrix_2d"] = result.procrustes_matrix_2d
        payload[f"{prefix}_anchor_s"] = np.asarray([a.anchor_s for a in result.anchors])
        payload[f"{prefix}_phase"] = result.anchors[0].phase
        for name in [
            "antipodal_cosine",
            "antipodal_residual",
            "antipodal_amplitude_ratio",
            "full_return_cosine",
            "full_return_residual",
            "full_return_amplitude_ratio",
            "midpoint_drift",
            "tangent_signed_p",
            "tangent_signed_2p",
        ]:
            payload[f"{prefix}_{name}"] = np.stack(
                [np.asarray(getattr(a, name)) for a in result.anchors],
                axis=0,
            )
        payload[f"{prefix}_matrices_0"] = np.stack([a.matrices_0 for a in result.anchors])
        payload[f"{prefix}_matrices_p"] = np.stack([a.matrices_p for a in result.anchors])
        payload[f"{prefix}_matrices_2p"] = np.stack([a.matrices_2p for a in result.anchors])
        payload[f"{prefix}_affine_centers"] = np.stack([a.affine_center for a in result.anchors])
        payload[f"{prefix}_cycle_means"] = np.stack([a.cycle_mean for a in result.anchors])
    np.savez_compressed(path, **payload)


def build_report(results: list[CaseAudit], outputs: dict[str, Path]) -> str:
    lines = [
        f"Script version: {SCRIPT_VERSION}",
        "",
        "ALTERNATING-STRIDE / ANTIPODAL-INVOLUTION AUDIT",
        "================================================",
        "",
        "The audit uses the centered moving part of the actual reduced 9x9",
        "matrices B(s).  It does not normalize loop closure by the large static",
        "Frobenius norm of the complete matrix.",
        "",
        "Tested relations:",
        "",
        "    B(s + P_line) - C  ~= -[B(s) - C]",
        "    B(s + 2 P_line) - C ~=  [B(s) - C]",
        "    w(s + P_line)       ~= -w(s)",
        "    w(s + 2 P_line)     ~=  w(s)",
        "",
    ]

    for result in results:
        s = result.summary
        lines.extend(
            [
                f"Q={result.q:g}",
                f"  source orbit                         = {result.source.resolve()}",
                f"  P_line                               = {result.p_line:.12f}",
                f"  P_oriented                           = {result.p_oriented:.12f}",
                f"  median antipodal cosine              = {s['median_antipodal_cosine']:+.12e}",
                f"  minimum antipodal cosine             = {s['minimum_antipodal_cosine']:+.12e}",
                f"  median antipodal residual            = {s['median_antipodal_residual']:.12e}",
                f"  maximum antipodal residual           = {s['maximum_antipodal_residual']:.12e}",
                f"  median antipodal amplitude ratio     = {s['median_antipodal_amplitude_ratio']:.12e}",
                f"  median 2P return cosine              = {s['median_full_return_cosine']:+.12e}",
                f"  minimum 2P return cosine             = {s['minimum_full_return_cosine']:+.12e}",
                f"  median 2P return residual            = {s['median_full_return_residual']:.12e}",
                f"  maximum 2P return residual           = {s['maximum_full_return_residual']:.12e}",
                f"  median 2P amplitude ratio            = {s['median_full_return_amplitude_ratio']:.12e}",
                f"  median pair-midpoint drift           = {s['median_pair_midpoint_drift']:.12e}",
                f"  median affine-center/cycle-mean diff = {s['median_center_difference']:.12e}",
                f"  median tangent signed return P       = {s['median_tangent_signed_p']:+.12e}",
                f"  median tangent signed return 2P      = {s['median_tangent_signed_2p']:+.12e}",
                f"  PC1+PC2 cumulative variance          = {s['pc1_pc2_cumulative']:.12e}",
                f"  active-plane map determinant         = {s['procrustes_determinant']:+.12e}",
                f"  active-plane map trace               = {s['procrustes_trace']:+.12e}",
                f"  active-plane distance to -I          = {s['procrustes_distance_to_minus_identity']:.12e}",
                f"  active-plane rotation angle degrees  = {s['procrustes_angle_degrees']:.9f}",
                f"  antipodal test passed                = {s['antipodal_test_passed']}",
                f"  full-return test passed              = {s['full_return_test_passed']}",
                f"  tangent alternation passed           = {s['tangent_alternation_passed']}",
                f"  active-plane half-turn passed        = {s['active_plane_half_turn_passed']}",
                f"  classification                       = {s['classification']}",
                "",
            ]
        )

    lines.extend(["FILES", "-----"])
    for label, path in outputs.items():
        lines.append(f"{label:24s} = {path.resolve()}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-script",
        type=Path,
        default=Path("test_support_mobius_monodromy_audit.py"),
    )
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--tail-periods", type=int, default=70)
    parser.add_argument("--steps-per-period", type=int, default=160)
    parser.add_argument("--samples-per-rotation", type=int, default=720)
    parser.add_argument("--max-recurrence-lag", type=float, default=0.65)
    parser.add_argument("--min-line-period", type=float, default=0.12)
    parser.add_argument("--max-line-period", type=float, default=0.32)
    parser.add_argument("--anchor-count", type=int, default=9)
    parser.add_argument("--anchor-span-rotations", type=float, default=4.0)
    parser.add_argument("--phase-points", type=int, default=240)
    parser.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--minimum-cosine", type=float, default=0.99)
    parser.add_argument("--maximum-residual", type=float, default=0.10)
    parser.add_argument("--minimum-tangent-overlap", type=float, default=0.99)
    parser.add_argument("--maximum-map-distance", type=float, default=0.15)
    parser.add_argument(
        "--output-prefix",
        default="support_stride_involution_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.case or [
        (
            522.25,
            Path(
                "support_feedback_q_curtain_focus_cache/"
                "q_522p25_orbit.npz"
            ),
        ),
        (
            550.0,
            Path("support_feedback_q_curtain_cache/q_550_orbit.npz"),
        ),
    ]

    if args.tail_periods < 20:
        raise ValueError("--tail-periods must be at least 20.")
    if args.steps_per_period < 40:
        raise ValueError("--steps-per-period must be at least 40.")
    if args.samples_per_rotation < 180:
        raise ValueError("--samples-per-rotation must be at least 180.")
    if args.anchor_count < 1:
        raise ValueError("--anchor-count must be positive.")
    if args.phase_points < 60:
        raise ValueError("--phase-points must be at least 60.")

    audit = load_module(args.audit_script)
    required = [
        "REDUCED_DIMENSION",
        "ModelParameters",
        "load_orbit",
        "collect_uniform_tail",
        "recurrence_curves",
        "estimate_line_period",
        "estimate_oriented_period",
        "build_state_spline",
        "reduced_jacobian_rotation",
        "interpolate_direction",
    ]
    missing = [name for name in required if not hasattr(audit, name)]
    if missing:
        raise AttributeError("Audit script is missing: " + ", ".join(missing))

    print(f"Script version = {SCRIPT_VERSION}")
    print(f"Audit script   = {args.audit_script.resolve()}")
    print(f"Anchors        = {args.anchor_count}")
    print(f"Phase points   = {args.phase_points} per P_line")

    results: list[CaseAudit] = []
    for q, orbit_path in sorted(cases, key=lambda item: item[0]):
        print("=" * 78)
        print(f"Alternating-stride audit for Q={q:g}")
        results.append(
            analyze_case(
                audit,
                q=q,
                orbit_path=orbit_path,
                args=args,
            )
        )

    prefix = Path(args.output_prefix)
    outputs = {
        "profiles CSV": prefix.with_name(prefix.name + "_profiles.csv"),
        "summary CSV": prefix.with_name(prefix.name + "_summary.csv"),
        "raw NPZ": prefix.with_name(prefix.name + "_raw.npz"),
        "report": prefix.with_name(prefix.name + "_report.txt"),
    }

    save_profiles_csv(outputs["profiles CSV"], results)
    save_summary_csv(outputs["summary CSV"], results)
    save_raw_npz(outputs["raw NPZ"], results)

    for result in results:
        token = q_token(result.q)
        embedding = prefix.with_name(prefix.name + f"_Q{token}_embedding.png")
        residuals = prefix.with_name(prefix.name + f"_Q{token}_residuals.png")
        overlaps = prefix.with_name(prefix.name + f"_Q{token}_overlaps.png")
        half_map = prefix.with_name(prefix.name + f"_Q{token}_half_map.png")
        save_embedding_plot(embedding, result, args.phase_points)
        save_residual_plot(residuals, result)
        save_overlap_plot(overlaps, result)
        save_map_plot(half_map, result, args.phase_points)
        outputs[f"Q={result.q:g} embedding PNG"] = embedding
        outputs[f"Q={result.q:g} residuals PNG"] = residuals
        outputs[f"Q={result.q:g} overlaps PNG"] = overlaps
        outputs[f"Q={result.q:g} half-map PNG"] = half_map

    report = build_report(results, outputs)
    outputs["report"].write_text(report, encoding="utf-8")
    print()
    print(report)


if __name__ == "__main__":
    main()
