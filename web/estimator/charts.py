"""Geometria del scatter 'esfuerzo real vs. estimado' (pantalla 'Modelo y
evaluacion'). Escala logaritmica en ambos ejes: el esfuerzo real va de ~6 a
~11400 persona-mes (3 ordenes de magnitud), una escala lineal aplastaria casi
todos los puntos contra el origen.
"""

import math

WIDTH = 640
HEIGHT = 360
MARGIN = {"left": 56, "right": 16, "top": 16, "bottom": 40}
PLOT_W = WIDTH - MARGIN["left"] - MARGIN["right"]
PLOT_H = HEIGHT - MARGIN["top"] - MARGIN["bottom"]

NICE_TICKS = [1, 10, 100, 1000, 10000, 100000]


def build_scatter(df):
    all_vals = list(df["effort_pm"]) + list(df["pred_mlp"]) + list(df["pred_cocomo"])
    log_min = math.log10(min(all_vals)) - 0.15
    log_max = math.log10(max(all_vals)) + 0.15
    span = log_max - log_min

    def sx(v):
        return MARGIN["left"] + (math.log10(v) - log_min) / span * PLOT_W

    def sy(v):
        return MARGIN["top"] + PLOT_H - (math.log10(v) - log_min) / span * PLOT_H

    def points(column):
        return [
            {
                "cx": round(sx(row.effort_pm), 1),
                "cy": round(sy(getattr(row, column)), 1),
                "real": round(row.effort_pm, 1),
                "pred": round(getattr(row, column), 1),
                "name": row.project_id,
            }
            for row in df.itertuples()
        ]

    ticks = [t for t in NICE_TICKS if log_min <= math.log10(t) <= log_max]

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "plot_left": MARGIN["left"],
        "plot_top": MARGIN["top"],
        "plot_right": WIDTH - MARGIN["right"],
        "plot_bottom": HEIGHT - MARGIN["bottom"],
        "points_mlp": points("pred_mlp"),
        "points_cocomo": points("pred_cocomo"),
        "diag": {"x1": sx(10 ** log_min), "y1": sy(10 ** log_min), "x2": sx(10 ** log_max), "y2": sy(10 ** log_max)},
        "xticks": [{"pos": round(sx(t), 1), "label": f"{t:,}"} for t in ticks],
        "yticks": [{"pos": round(sy(t), 1), "label": f"{t:,}"} for t in ticks],
    }
