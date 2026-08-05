#!/usr/bin/env python3
"""Export a direct PCA embedding of the reduced tangent matrices B(s).

This script imports the validated machinery from
``test_support_mobius_monodromy_audit.py``.  It samples the actual reduced
rotation-time Jacobian B(s) (9x9) along several P_line loops, flattens every
matrix to 81 dimensions, performs PCA, and exports both the raw matrices and
the embedding.

No proxy geometry is used: the PCA input is the matrix sequence itself.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_VERSION = "2026-08-04-direct-B-loop-embedding-v2-python313-import-fix"


def parse_case(raw: str) -> tuple[float, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--case must use Q=orbit_cache.npz")
    q_text, path_text = raw.split("=", 1)
    return float(q_text.strip()), Path(path_text.strip())


def load_module(path: Path) -> ModuleType:
    """Load the audit script as a real registered Python module.

    Python 3.13 dataclasses resolve annotation metadata through
    sys.modules[cls.__module__] while the class decorator is executing.
    A module created with module_from_spec() must therefore be registered
    before exec_module() is called.
    """
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audit script not found: {path}")

    module_name = "mobius_audit_runtime"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(a, ord="fro")),
        float(np.linalg.norm(b, ord="fro")),
        1e-15,
    )
    return float(np.linalg.norm(a - b, ord="fro") / denominator)


def run_pca(vectors: np.ndarray):
    mean = np.mean(vectors, axis=0)
    centered = vectors - mean
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ components.T
    variance = singular_values**2 / max(vectors.shape[0] - 1, 1)
    explained = variance / np.sum(variance)
    return mean, components, singular_values, explained, scores


def estimate_periods(audit, tail, args):
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


def analyze_case(audit, q: float, orbit_path: Path, args):
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
    spline = audit.build_state_spline(tail)
    kappa0 = (55.0 / 6.0) * (10.0 / q)

    anchor_end = float(tail.s[-1] - p_line)
    anchor_start = max(float(tail.s[0]), anchor_end - args.anchor_span_rotations)
    anchors = np.linspace(anchor_start, anchor_end, args.anchor_count)
    phase = np.linspace(0.0, 1.0, args.phase_points)

    all_matrices = []
    metadata = []
    closures = []

    for anchor_index, anchor_s in enumerate(anchors):
        s_values = anchor_s + phase * p_line
        matrices = np.empty(
            (args.phase_points, audit.REDUCED_DIMENSION, audit.REDUCED_DIMENSION),
            dtype=float,
        )
        for index, s_value in enumerate(s_values):
            state = np.asarray(spline(s_value), dtype=float)
            matrices[index] = audit.reduced_jacobian_rotation(
                state,
                kappa0=kappa0,
                params=params,
                embedding_name="symmetric",
            )
        if not np.all(np.isfinite(matrices)):
            raise FloatingPointError(f"Q={q:g}: non-finite B(s) samples")
        closure = relative_frobenius(matrices[0], matrices[-1])
        closures.append(closure)
        all_matrices.append(matrices)
        for phase_index, (phi, s_value) in enumerate(zip(phase, s_values)):
            metadata.append([anchor_index, anchor_s, phase_index, phi, s_value])
        print(
            f"  Q={q:g}: anchor {anchor_index + 1}/{args.anchor_count}, "
            f"direct closure={closure:.6e}"
        )

    matrices = np.stack(all_matrices, axis=0)
    vectors = matrices.reshape(-1, audit.REDUCED_DIMENSION**2)
    metadata = np.asarray(metadata, dtype=float)
    mean, components, singular_values, explained, scores = run_pca(vectors)

    return {
        "q": q,
        "source": orbit_path,
        "p_line": p_line,
        "p_oriented": p_oriented,
        "anchors": anchors,
        "phase": phase,
        "matrices": matrices,
        "vectors": vectors,
        "metadata": metadata,
        "closures": np.asarray(closures),
        "mean": mean,
        "components": components,
        "singular_values": singular_values,
        "explained": explained,
        "scores": scores,
    }


def write_matrix_csv(path: Path, results, dim: int):
    matrix_names = [f"B_{i}_{j}" for i in range(dim) for j in range(dim)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Q", "anchor_index", "anchor_s", "phase_index", "loop_phase",
            "s_value", "anchor_closure_relative", "matrix_frobenius_norm",
            *matrix_names,
        ])
        for result in results:
            points = result["phase"].size
            for anchor_index, anchor_s in enumerate(result["anchors"]):
                for phase_index in range(points):
                    matrix = result["matrices"][anchor_index, phase_index]
                    writer.writerow([
                        result["q"], anchor_index, anchor_s, phase_index,
                        result["phase"][phase_index],
                        anchor_s + result["phase"][phase_index] * result["p_line"],
                        result["closures"][anchor_index],
                        np.linalg.norm(matrix, ord="fro"),
                        *matrix.reshape(-1),
                    ])


def write_embedding_csv(path: Path, results):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Q", "anchor_index", "anchor_s", "phase_index", "loop_phase",
            "s_value", "PC1", "PC2", "PC3",
        ])
        for result in results:
            for index, meta in enumerate(result["metadata"]):
                writer.writerow([
                    result["q"], int(meta[0]), meta[1], int(meta[2]), meta[3],
                    meta[4], *result["scores"][index, :3],
                ])


def write_components_csv(path: Path, results, dim: int):
    names = [f"B_{i}_{j}" for i in range(dim) for j in range(dim)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Q", "component", "singular_value", "explained_variance_ratio",
            "cumulative_explained_variance", *names,
        ])
        for result in results:
            cumulative = np.cumsum(result["explained"])
            for index in range(min(12, result["components"].shape[0])):
                writer.writerow([
                    result["q"], index + 1, result["singular_values"][index],
                    result["explained"][index], cumulative[index],
                    *result["components"][index],
                ])


def write_summary_csv(path: Path, results):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Q", "source", "P_line", "P_oriented", "anchor_count",
            "phase_points", "median_direct_closure", "maximum_direct_closure",
            "PC1_explained", "PC2_explained", "PC3_explained",
            "PC1_PC2_cumulative", "PC1_PC2_PC3_cumulative",
            "dimension_99_percent", "dimension_99_9_percent",
        ])
        for result in results:
            cumulative = np.cumsum(result["explained"])
            writer.writerow([
                result["q"], str(result["source"].resolve()), result["p_line"],
                result["p_oriented"], result["anchors"].size,
                result["phase"].size, np.median(result["closures"]),
                np.max(result["closures"]), result["explained"][0],
                result["explained"][1], result["explained"][2], cumulative[1],
                cumulative[2], int(np.searchsorted(cumulative, 0.99) + 1),
                int(np.searchsorted(cumulative, 0.999) + 1),
            ])


def write_npz(path: Path, results):
    payload = {"script_version": SCRIPT_VERSION}
    for result in results:
        token = str(result["q"]).replace(".", "p")
        prefix = f"q_{token}"
        for name in [
            "p_line", "p_oriented", "anchors", "phase", "matrices",
            "vectors", "metadata", "closures", "mean", "components",
            "singular_values", "explained", "scores",
        ]:
            payload[f"{prefix}_{name}"] = result[name]
    np.savez_compressed(path, **payload)


def save_embedding_plot(path: Path, results):
    figure = plt.figure(figsize=(15, 7))
    for panel, result in enumerate(results, start=1):
        axis = figure.add_subplot(1, len(results), panel, projection="3d")
        points = result["phase"].size
        for anchor_index in range(result["anchors"].size):
            start = anchor_index * points
            stop = start + points
            score = result["scores"][start:stop]
            axis.plot(score[:, 0], score[:, 1], score[:, 2], alpha=0.55)
        mid = result["anchors"].size // 2
        start = mid * points
        stop = start + points
        score = result["scores"][start:stop]
        axis.plot(score[:, 0], score[:, 1], score[:, 2], linewidth=3.0,
                  label="median anchor loop")
        axis.scatter(*score[0, :3], s=55, marker="o", label="phase 0")
        axis.scatter(*score[-1, :3], s=55, marker="x", label="phase 1")
        exp = result["explained"]
        axis.set_title(
            f"Q={result['q']:g}\nPC1+PC2+PC3={100*np.sum(exp[:3]):.6f}%"
        )
        axis.set_xlabel(f"PC1 ({100*exp[0]:.3f}%)")
        axis.set_ylabel(f"PC2 ({100*exp[1]:.3f}%)")
        axis.set_zlabel(f"PC3 ({100*exp[2]:.3f}%)")
        axis.legend()
    figure.suptitle(
        "Direct PCA embedding of the actual reduced matrices B(s)\n"
        "Each curve is one sampled P_line coefficient loop",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def save_variance_plot(path: Path, results):
    figure, axis = plt.subplots(figsize=(11, 7))
    for result in results:
        count = min(12, result["explained"].size)
        axis.plot(
            np.arange(1, count + 1),
            np.cumsum(result["explained"][:count]),
            marker="o", label=f"Q={result['q']:g}",
        )
    axis.axhline(0.99, linestyle=":")
    axis.axhline(0.999, linestyle="--")
    axis.set_xlabel("number of principal components")
    axis.set_ylabel("cumulative explained variance")
    axis.set_ylim(0.0, 1.005)
    axis.set_title("Intrinsic dimension of the direct B(s) loop")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def build_report(results, outputs):
    lines = [
        f"Script version: {SCRIPT_VERSION}", "",
        "DIRECT B(s) MATRIX LOOP EMBEDDING", "=================================", "",
        "The PCA input consists of the actual reduced 9x9 rotation-time",
        "coefficient matrices B(s), flattened to 81 dimensions.", "",
    ]
    for result in results:
        cumulative = np.cumsum(result["explained"])
        lines.extend([
            f"Q={result['q']:g}",
            f"  source orbit                = {result['source'].resolve()}",
            f"  P_line                      = {result['p_line']:.12f}",
            f"  P_oriented                  = {result['p_oriented']:.12f}",
            f"  anchors                     = {result['anchors'].size}",
            f"  phase points per loop       = {result['phase'].size}",
            f"  median direct closure       = {np.median(result['closures']):.12e}",
            f"  maximum direct closure      = {np.max(result['closures']):.12e}",
            f"  PC1 explained               = {result['explained'][0]:.12e}",
            f"  PC2 explained               = {result['explained'][1]:.12e}",
            f"  PC3 explained               = {result['explained'][2]:.12e}",
            f"  PC1+PC2 cumulative          = {cumulative[1]:.12e}",
            f"  PC1+PC2+PC3 cumulative      = {cumulative[2]:.12e}", "",
        ])
    lines.extend(["FILES", "-----"])
    for key, value in outputs.items():
        lines.append(f"{key:18s} = {value.resolve()}")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-script", type=Path,
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
    parser.add_argument("--phase-points", type=int, default=241)
    parser.add_argument("--initial-x-perturbation", type=float, default=1e-6)
    parser.add_argument("--gamma-rho", type=float, default=0.3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--output-prefix", default="support_direct_B_loop_embedding"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases = args.case or [
        (522.25, Path("support_feedback_q_curtain_focus_cache/q_522p25_orbit.npz")),
        (550.0, Path("support_feedback_q_curtain_cache/q_550_orbit.npz")),
    ]
    audit = load_module(args.audit_script)
    required = [
        "REDUCED_DIMENSION", "ModelParameters", "load_orbit",
        "collect_uniform_tail", "recurrence_curves", "estimate_line_period",
        "estimate_oriented_period", "build_state_spline",
        "reduced_jacobian_rotation",
    ]
    missing = [name for name in required if not hasattr(audit, name)]
    if missing:
        raise AttributeError("Audit script is missing: " + ", ".join(missing))

    print(f"Script version = {SCRIPT_VERSION}")
    print(f"Audit script   = {args.audit_script.resolve()}")
    results = []
    for q, orbit_path in sorted(cases):
        print("=" * 78)
        print(f"Direct B(s) export for Q={q:g}")
        results.append(analyze_case(audit, q, orbit_path, args))

    prefix = Path(args.output_prefix)
    outputs = {
        "matrices CSV": prefix.with_name(prefix.name + "_matrices.csv"),
        "embedding CSV": prefix.with_name(prefix.name + "_embedding.csv"),
        "components CSV": prefix.with_name(prefix.name + "_components.csv"),
        "summary CSV": prefix.with_name(prefix.name + "_summary.csv"),
        "raw NPZ": prefix.with_name(prefix.name + "_raw.npz"),
        "embedding PNG": prefix.with_name(prefix.name + "_embedding.png"),
        "variance PNG": prefix.with_name(prefix.name + "_variance.png"),
        "report": prefix.with_name(prefix.name + "_report.txt"),
    }
    write_matrix_csv(outputs["matrices CSV"], results, audit.REDUCED_DIMENSION)
    write_embedding_csv(outputs["embedding CSV"], results)
    write_components_csv(outputs["components CSV"], results, audit.REDUCED_DIMENSION)
    write_summary_csv(outputs["summary CSV"], results)
    write_npz(outputs["raw NPZ"], results)
    save_embedding_plot(outputs["embedding PNG"], results)
    save_variance_plot(outputs["variance PNG"], results)
    report = build_report(results, outputs)
    outputs["report"].write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
