#!/usr/bin/env python3
"""
plot_windings_3d_v2.py

Versione migliorata di plot_windings_3d.py:
- caricamento dati piu' robusto (validazione colonne, gestione errori)
- CLI (argparse) al posto dei parametri hardcoded, ma con default identici
- output interattivo con Plotly (ruota/zoom/hover con i valori esatti),
  molto piu' comodo di matplotlib per esplorare dati 3D, specie se lavori
  da server remoto (niente bisogno di un display per vedere il plot: si
  salva un file HTML che apri nel browser)
- fallback automatico a matplotlib se plotly non e' installato
- log invece di print, cosi' sai sempre cosa e' stato caricato/scartato
- colore continuo sul winding (o sullo step, a scelta) invece di due soli
  colori piatti, per leggere meglio landamento lungo l'asse che ti interessa
"""

import argparse
import glob
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Step", "Numberline_Q", "Type"}


def parse_args():
    p = argparse.ArgumentParser(description="Plot 3D dei dati di winding (Step / Numberline_Q / Winding).")
    p.add_argument("--data-folder", default=".", help="Cartella con i CSV (default: .)")
    p.add_argument("--winding-min", type=int, default=1)
    p.add_argument("--winding-max", type=int, default=20)
    p.add_argument("--sort-mode", choices=["asc", "desc"], default="asc")
    p.add_argument("--pattern", default="summary_winding_{w}_Q1-Q2000.csv",
                    help="Pattern del nome file, con {w} al posto del numero di winding")
    p.add_argument("--color-by", choices=["winding", "step"], default="winding",
                    help="Variabile usata per il colore continuo dei punti 'Stiffness'")
    p.add_argument("--elev", type=float, default=20, help="Elevazione (solo modalita' matplotlib)")
    p.add_argument("--azim", type=float, default=35, help="Azimuth (solo modalita' matplotlib)")
    p.add_argument("--engine", choices=["plotly", "matplotlib"], default="plotly",
                    help="Motore di rendering. plotly = HTML interattivo, matplotlib = finestra statica")
    p.add_argument("--out", default="windings_3d.html",
                    help="File di output (HTML per plotly, PNG per matplotlib se --no-show)")
    p.add_argument("--no-show", action="store_true",
                    help="Non aprire una finestra/interattivo: salva solo su file")
    return p.parse_args()


def load_data(data_folder, winding_min, winding_max, pattern):
    all_data = []
    for w in range(winding_min, winding_max + 1):
        filename = pattern.format(w=w)
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            log.warning("File non trovato: %s", filename)
            continue
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            log.error("Errore leggendo %s: %s", filename, e)
            continue

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            log.error("File %s scartato: mancano colonne %s", filename, missing)
            continue

        df["Winding"] = w
        all_data.append(df)

    if not all_data:
        raise RuntimeError(
            "Nessun file caricato. Controlla --data-folder e --pattern "
            f"(atteso qualcosa tipo '{pattern.format(w=winding_min)}')."
        )

    data = pd.concat(all_data, ignore_index=True)
    log.info("Caricati %d punti da %d file.", len(data), len(all_data))
    return data


def plot_plotly(data, color_by, out_path, show):
    import plotly.graph_objects as go

    stiff = data[data["Type"] == "Stiffness"]
    conv = data[data["Type"] == "Converged"]

    fig = go.Figure()

    color_column = {"winding": "Winding", "step": "Step"}[color_by]

    fig.add_trace(go.Scatter3d(
        x=stiff["Winding"], y=stiff["Numberline_Q"], z=stiff["Step"],
        mode="markers",
        name="Stiffness",
        marker=dict(
            size=3,
            color=stiff[color_column],
            colorscale="Viridis",
            colorbar=dict(title=color_column),
            opacity=0.7,
        ),
        hovertemplate="Winding=%{x}<br>Numberline_Q=%{y}<br>Step=%{z}<extra>Stiffness</extra>",
    ))

    fig.add_trace(go.Scatter3d(
        x=conv["Winding"], y=conv["Numberline_Q"], z=conv["Step"],
        mode="markers",
        name="Converged",
        marker=dict(size=5, color="red", symbol="diamond"),
        hovertemplate="Winding=%{x}<br>Numberline_Q=%{y}<br>Step=%{z}<extra>Converged</extra>",
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Winding",
            yaxis_title="Numberline_Q",
            zaxis_title="Step",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0),
        title="Windings 3D — Stiffness vs Converged",
    )

    fig.write_html(out_path)
    log.info("Salvato HTML interattivo in: %s", out_path)
    if show:
        fig.show()


def plot_matplotlib(data, elev, azim, out_path, show):
    import matplotlib.pyplot as plt  # noqa: F401 (registra il backend 3d)

    stiff = data[data["Type"] == "Stiffness"]
    conv = data[data["Type"] == "Converged"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        stiff["Winding"], stiff["Numberline_Q"], stiff["Step"],
        c=stiff["Winding"], cmap="viridis", s=6, alpha=0.7, label="Stiffness"
    )
    ax.scatter(
        conv["Winding"], conv["Numberline_Q"], conv["Step"],
        c="red", s=12, label="Converged"
    )

    ax.set_xlabel("Winding")
    ax.set_ylabel("Numberline_Q")
    ax.set_zlabel("Step")
    ax.grid(True)
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Winding")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        log.info("Salvato PNG in: %s", out_path)
    if show:
        plt.show()


def main():
    args = parse_args()

    data = load_data(args.data_folder, args.winding_min, args.winding_max, args.pattern)
    data = data.sort_values(by="Step", ascending=(args.sort_mode == "asc"))

    if args.engine == "plotly":
        try:
            out_path = args.out if args.out.endswith(".html") else args.out + ".html"
            plot_plotly(data, args.color_by, out_path, show=not args.no_show)
        except ImportError:
            log.warning("plotly non installato (pip install plotly --break-system-packages). Uso matplotlib.")
            out_path = os.path.splitext(args.out)[0] + ".png"
            plot_matplotlib(data, args.elev, args.azim, out_path, show=not args.no_show)
    else:
        out_path = os.path.splitext(args.out)[0] + ".png"
        plot_matplotlib(data, args.elev, args.azim, out_path, show=not args.no_show)


if __name__ == "__main__":
    main()
