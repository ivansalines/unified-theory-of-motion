#!/usr/bin/env python3
"""
Dashboard Plotly 3D sincronizzata per i dati di winding.

Mostra quattro viste dello stesso oggetto informativo:
- Step
- dt
- E
- ||F||

Una vista e' in primo piano e le altre tre sono laterali. Facendo clic su una
vista laterale, questa diventa principale. Rotazione e zoom della camera 3D
sono sincronizzati in tempo reale fra tutti i plot.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import webbrowser
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

METRICS = ("Step", "dt", "E", "||F||")
REQUIRED_COLUMNS = {"Numberline_Q", "Type", *METRICS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dashboard 3D sincronizzata: Step, dt, E e ||F|| rispetto a "
            "Winding e Numberline_Q."
        )
    )
    parser.add_argument("--data-folder", default=".", help="Cartella contenente i CSV")
    parser.add_argument("--winding-min", type=int, default=1)
    parser.add_argument("--winding-max", type=int, default=20)
    parser.add_argument(
        "--pattern",
        default="summary_winding_{w}_Q1-Q2000.csv",
        help="Pattern dei CSV; usare {w} per il numero di winding",
    )
    parser.add_argument(
        "--main-plot",
        choices=METRICS,
        default="Step",
        help="Vista inizialmente in primo piano",
    )
    parser.add_argument(
        "--color-by",
        choices=("winding", "metric"),
        default="winding",
        help="Colore dei punti Stiffness: winding oppure metrica della vista",
    )
    parser.add_argument("--out", default="windings_3d_dashboard.html")
    parser.add_argument("--stiff-size", type=float, default=1.5)
    parser.add_argument("--stiff-opacity", type=float, default=0.55)
    parser.add_argument("--conv-size", type=float, default=2.5)
    parser.add_argument("--conv-opacity", type=float, default=0.50)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Genera l'HTML senza aprirlo nel browser",
    )
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for winding in range(args.winding_min, args.winding_max + 1):
        filename = args.pattern.format(w=winding)
        path = Path(args.data_folder) / filename

        if not path.exists():
            log.warning("File non trovato: %s", path)
            continue

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            log.error("Errore leggendo %s: %s", path, exc)
            continue

        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            log.error("File %s scartato: mancano colonne %s", path, sorted(missing))
            continue

        frame = frame.copy()
        frame["Winding"] = winding
        frames.append(frame)

    if not frames:
        example = args.pattern.format(w=args.winding_min)
        raise RuntimeError(
            "Nessun CSV valido caricato. Controlla --data-folder e --pattern "
            f"(esempio atteso: {example})."
        )

    data = pd.concat(frames, ignore_index=True)

    for column in ("Winding", "Numberline_Q", *METRICS):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    before = len(data)
    data = data.dropna(subset=["Winding", "Numberline_Q", *METRICS, "Type"])
    dropped = before - len(data)
    if dropped:
        log.warning("Scartate %d righe con valori mancanti o non numerici.", dropped)

    log.info("Caricati %d punti da %d file.", len(data), len(frames))
    return data


def build_figure(data: pd.DataFrame, metric: str, args: argparse.Namespace):
    import plotly.graph_objects as go

    stiff = data[data["Type"] == "Stiffness"]
    conv = data[data["Type"] == "Converged"]
    color_values = stiff["Winding"] if args.color_by == "winding" else stiff[metric]
    color_title = "Winding" if args.color_by == "winding" else metric

    custom_columns = ["Step", "dt", "E", "||F||", "Type"]
    stiff_custom = stiff[custom_columns].to_numpy()
    conv_custom = conv[custom_columns].to_numpy()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=stiff["Winding"],
            y=stiff["Numberline_Q"],
            z=stiff[metric],
            mode="markers",
            name="Stiffness",
            customdata=stiff_custom,
            marker={
                "size": args.stiff_size,
                "color": color_values,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": color_title, "thickness": 12, "len": 0.55},
                "opacity": args.stiff_opacity,
                "line": {"width": 0},
            },
            hovertemplate=(
                "Winding=%{x}<br>Numberline_Q=%{y}"
                "<br>Step=%{customdata[0]}"
                "<br>dt=%{customdata[1]}"
                "<br>E=%{customdata[2]}"
                "<br>||F||=%{customdata[3]}"
                "<extra>Stiffness</extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=conv["Winding"],
            y=conv["Numberline_Q"],
            z=conv[metric],
            mode="markers",
            name="Converged",
            customdata=conv_custom,
            marker={
                "size": args.conv_size,
                "color": "red",
                "symbol": "diamond",
                "opacity": args.conv_opacity,
                "line": {"width": 0},
            },
            hovertemplate=(
                "Winding=%{x}<br>Numberline_Q=%{y}"
                "<br>Step=%{customdata[0]}"
                "<br>dt=%{customdata[1]}"
                "<br>E=%{customdata[2]}"
                "<br>||F||=%{customdata[3]}"
                "<extra>Converged</extra>"
            ),
        )
    )

    figure.update_layout(
        title={"text": metric, "x": 0.5, "xanchor": "center"},
        scene={
            "xaxis_title": "Winding",
            "yaxis_title": "Numberline_Q",
            "zaxis_title": metric,
            "aspectmode": "auto",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.10}},
        },
        legend={"itemsizing": "constant", "orientation": "h", "y": 0.99, "x": 0.01},
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        paper_bgcolor="#10141c",
        plot_bgcolor="#10141c",
        font={"color": "#e8edf5"},
        uirevision="windings-synchronized-view",
    )
    return figure


def build_html(figures: dict[str, object], main_plot: str) -> str:
    figure_json = {
        metric: json.loads(figure.to_json())
        for metric, figure in figures.items()
    }
    payload = json.dumps(figure_json, ensure_ascii=False)
    metrics_json = json.dumps(list(METRICS), ensure_ascii=False)
    main_json = json.dumps(main_plot, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Windings 3D — viste sincronizzate</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; }}
    body {{
      background: #090c12;
      color: #e8edf5;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #dashboard {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      grid-template-rows: repeat(3, minmax(0, 1fr));
      gap: 10px;
      width: 100vw;
      height: 100vh;
      padding: 10px;
    }}
    .plot-card {{
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
      border: 1px solid #293244;
      border-radius: 12px;
      background: #10141c;
      box-shadow: 0 8px 24px rgba(0,0,0,.22);
      transition: border-color .18s ease, box-shadow .18s ease;
    }}
    .plot-card:hover {{ border-color: #6581ad; }}
    .plot-card.main {{
      grid-column: 1;
      grid-row: 1 / 4;
      border-color: #6f91c7;
      box-shadow: 0 10px 36px rgba(0,0,0,.38);
    }}
    .plot-card.side {{ cursor: pointer; }}
    .plot-card.side[data-side-index="0"] {{ grid-column: 2; grid-row: 1; }}
    .plot-card.side[data-side-index="1"] {{ grid-column: 2; grid-row: 2; }}
    .plot-card.side[data-side-index="2"] {{ grid-column: 2; grid-row: 3; }}
    .plot {{ width: 100%; height: 100%; }}
    .badge {{
      position: absolute;
      z-index: 5;
      right: 10px;
      top: 8px;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(9,12,18,.76);
      border: 1px solid #34415a;
      color: #c8d3e6;
      font-size: 11px;
      pointer-events: none;
    }}
    .main .badge::after {{ content: " · principale"; color: #8fb5ee; }}
    @media (max-width: 900px) {{
      body {{ overflow: auto; }}
      #dashboard {{
        height: auto;
        min-height: 100vh;
        grid-template-columns: 1fr;
        grid-template-rows: 70vh repeat(3, 38vh);
      }}
      .plot-card.main {{ grid-column: 1; grid-row: 1; }}
      .plot-card.side[data-side-index="0"] {{ grid-column: 1; grid-row: 2; }}
      .plot-card.side[data-side-index="1"] {{ grid-column: 1; grid-row: 3; }}
      .plot-card.side[data-side-index="2"] {{ grid-column: 1; grid-row: 4; }}
    }}
  </style>
</head>
<body>
  <main id="dashboard"></main>
  <script>
    const figures = {payload};
    const metrics = {metrics_json};
    let mainMetric = {main_json};
    let syncingCamera = false;

    const dashboard = document.getElementById('dashboard');

    function makeCard(metric) {{
      const card = document.createElement('section');
      card.className = 'plot-card';
      card.dataset.metric = metric;

      const badge = document.createElement('div');
      badge.className = 'badge';
      badge.textContent = metric;

      const plot = document.createElement('div');
      plot.className = 'plot';
      plot.id = `plot-${{metrics.indexOf(metric)}}`;

      card.appendChild(badge);
      card.appendChild(plot);
      card.addEventListener('click', () => {{
        if (metric !== mainMetric) setMain(metric);
      }});
      dashboard.appendChild(card);
      return plot;
    }}

    function applyLayoutClasses() {{
      let sideIndex = 0;
      document.querySelectorAll('.plot-card').forEach(card => {{
        const isMain = card.dataset.metric === mainMetric;
        card.classList.toggle('main', isMain);
        card.classList.toggle('side', !isMain);
        if (isMain) {{
          delete card.dataset.sideIndex;
        }} else {{
          card.dataset.sideIndex = sideIndex++;
        }}
      }});
    }}

    function resizeAll() {{
      requestAnimationFrame(() => {{
        metrics.forEach(metric => Plotly.Plots.resize(document.getElementById(`plot-${{metrics.indexOf(metric)}}`)));
      }});
    }}

    function setMain(metric) {{
      mainMetric = metric;
      applyLayoutClasses();
      resizeAll();
    }}

    function extractCamera(update) {{
      if (update['scene.camera']) return update['scene.camera'];
      const camera = {{}};
      let found = false;
      for (const [key, value] of Object.entries(update)) {{
        if (!key.startsWith('scene.camera.')) continue;
        found = true;
        const path = key.substring('scene.camera.'.length).split('.');
        let cursor = camera;
        path.forEach((part, index) => {{
          if (index === path.length - 1) cursor[part] = value;
          else cursor = cursor[part] ||= {{}};
        }});
      }}
      return found ? camera : null;
    }}

    async function synchronizeCamera(sourceMetric, update) {{
      const camera = extractCamera(update);
      if (!camera || syncingCamera) return;

      syncingCamera = true;
      try {{
        await Promise.all(metrics
          .filter(metric => metric !== sourceMetric)
          .map(metric => Plotly.relayout(
            document.getElementById(`plot-${{metrics.indexOf(metric)}}`),
            {{'scene.camera': camera}}
          )));
      }} finally {{
        syncingCamera = false;
      }}
    }}

    async function init() {{
      metrics.forEach(makeCard);
      applyLayoutClasses();

      await Promise.all(metrics.map(metric => {{
        const div = document.getElementById(`plot-${{metrics.indexOf(metric)}}`);
        const figure = figures[metric];
        return Plotly.newPlot(div, figure.data, figure.layout, {{
          responsive: true,
          displaylogo: false,
          scrollZoom: true,
          modeBarButtonsToRemove: ['toImage']
        }}).then(() => {{
          div.on('plotly_relayout', update => synchronizeCamera(metric, update));
        }});
      }}));

      resizeAll();
    }}

    window.addEventListener('resize', resizeAll);
    init();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data = load_data(args)

    try:
        figures = {metric: build_figure(data, metric, args) for metric in METRICS}
    except ImportError as exc:
        raise SystemExit("Plotly non installato. Esegui: pip install plotly pandas") from exc

    output = Path(args.out)
    if output.suffix.lower() != ".html":
        output = output.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(figures, args.main_plot), encoding="utf-8")

    absolute = output.resolve()
    log.info("Dashboard salvata in: %s", absolute)
    if not args.no_show:
        webbrowser.open(absolute.as_uri())


if __name__ == "__main__":
    main()
