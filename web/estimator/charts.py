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


LOSS_WIDTH = 640
LOSS_HEIGHT = 280
LOSS_MARGIN = {"left": 48, "right": 16, "top": 16, "bottom": 32}
LOSS_PLOT_W = LOSS_WIDTH - LOSS_MARGIN["left"] - LOSS_MARGIN["right"]
LOSS_PLOT_H = LOSS_HEIGHT - LOSS_MARGIN["top"] - LOSS_MARGIN["bottom"]


def build_loss_curve(df):
    """Curva real de entrenamiento (loss/val_loss por epoca) del ultimo
    reentrenamiento del modelo final, leida de train_history.csv. Eje Y
    logaritmico (el error baja ~2 ordenes de magnitud), eje X = epoca."""
    epochs = df["epoch"].tolist()
    losses = df["loss"].tolist()
    val_losses = df["val_loss"].tolist()
    n_max = max(epochs)

    all_vals = losses + val_losses
    log_min = math.log10(min(all_vals)) - 0.1
    log_max = math.log10(max(all_vals)) + 0.1
    span = log_max - log_min

    def sx(e):
        return LOSS_MARGIN["left"] + ((e - 1) / (n_max - 1) if n_max > 1 else 0.5) * LOSS_PLOT_W

    def sy(v):
        return LOSS_MARGIN["top"] + LOSS_PLOT_H - (math.log10(v) - log_min) / span * LOSS_PLOT_H

    def path_for(values):
        return "M " + " L ".join(f"{round(sx(e), 1)},{round(sy(v), 1)}" for e, v in zip(epochs, values))

    yticks = [t for t in NICE_TICKS if log_min <= math.log10(t) <= log_max]
    step = 25 if n_max <= 200 else 50
    xtick_vals = list(range(step, n_max + 1, step))
    if not xtick_vals or xtick_vals[0] != 1:
        xtick_vals = [1] + xtick_vals

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])

    return {
        "width": LOSS_WIDTH,
        "height": LOSS_HEIGHT,
        "plot_left": LOSS_MARGIN["left"],
        "plot_top": LOSS_MARGIN["top"],
        "plot_right": LOSS_WIDTH - LOSS_MARGIN["right"],
        "plot_bottom": LOSS_HEIGHT - LOSS_MARGIN["bottom"],
        "train_path": path_for(losses),
        "val_path": path_for(val_losses),
        "yticks": [{"pos": round(sy(t), 1), "label": f"{t:,}"} for t in yticks],
        "xticks": [{"pos": round(sx(e), 1), "label": str(e)} for e in xtick_vals],
        "best_point": {
            "x": round(sx(epochs[best_idx]), 1),
            "y": round(sy(val_losses[best_idx]), 1),
            "epoch": epochs[best_idx],
            "val_loss": round(val_losses[best_idx], 4),
        },
        "epochs_run": n_max,
    }
