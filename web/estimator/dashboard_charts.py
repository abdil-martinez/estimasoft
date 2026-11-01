"""Geometria de los widgets de 'Indicadores': scatter KLOC vs esfuerzo y dona
de modos de desarrollo. El ranking y las barras de error local se resuelven
con el mismo patron .bar-track/.bar-fill que ya usa panel.html, calculado en
la vista. Mismo enfoque que charts.py: SVG calculado en Python, sin JS ni
librerias de graficos."""

import math

DONUT_SIZE = 180
DONUT_R = 62
DONUT_STROKE = 26

SCATTER_W, SCATTER_H = 560, 280
SCATTER_MARGIN = {"left": 48, "right": 16, "top": 16, "bottom": 32}
SCATTER_PLOT_W = SCATTER_W - SCATTER_MARGIN["left"] - SCATTER_MARGIN["right"]
SCATTER_PLOT_H = SCATTER_H - SCATTER_MARGIN["top"] - SCATTER_MARGIN["bottom"]


def build_effort_scatter(estimations):
    """estimations: lista con .kloc, .effort_mlp_pm, .effort_cocomo_pm,
    .project_name. Dispersion KLOC (eje X) vs esfuerzo en PM (eje Y), con una
    serie por metodo (MLP y COCOMO), cada una conectada por una linea al
    ordenar por KLOC -- reemplaza el 'costo en el tiempo' (las estimaciones de
    prueba se guardan todas el mismo dia, ese eje no aporta). Util tambien
    para detectar casos donde el esfuerzo escala muy por encima del resto
    (factores de costo altos o posible extrapolacion)."""
    rows = list(estimations)
    if not rows:
        return None

    klocs = [r.kloc for r in rows]
    efforts = [r.effort_mlp_pm for r in rows] + [r.effort_cocomo_pm for r in rows]
    kmin, kmax = min(klocs), max(klocs)
    emin, emax = min(efforts), max(efforts)
    if kmax == kmin:
        kmin = 0
    if emax == emin:
        emin = 0
    kpad = (kmax - kmin) * 0.08 or 1
    epad = (emax - emin) * 0.08 or 1

    def sx(v):
        return SCATTER_MARGIN["left"] + (v - (kmin - kpad)) / ((kmax + kpad) - (kmin - kpad)) * SCATTER_PLOT_W

    def sy(v):
        return SCATTER_MARGIN["top"] + SCATTER_PLOT_H - (v - (emin - epad)) / ((emax + epad) - (emin - epad)) * SCATTER_PLOT_H

    ordered = sorted(rows, key=lambda r: r.kloc)

    def series(attr):
        pts = [
            {
                "cx": round(sx(r.kloc), 1),
                "cy": round(sy(getattr(r, attr)), 1),
                "kloc": round(r.kloc, 1),
                "effort": round(getattr(r, attr), 1),
                "name": r.project_name,
            }
            for r in ordered
        ]
        path = "M " + " L ".join(f"{p['cx']},{p['cy']}" for p in pts)
        return pts, path

    points_mlp, path_mlp = series("effort_mlp_pm")
    points_cocomo, path_cocomo = series("effort_cocomo_pm")

    def nice_ticks(lo, hi, count=4):
        span = hi - lo
        if span <= 0:
            return [lo]
        step = span / count
        return [round(lo + step * i, 1) for i in range(count + 1)]

    xticks = [{"pos": round(sx(v), 1), "label": f"{v:,.0f}"} for v in nice_ticks(kmin, kmax)]
    yticks = [{"pos": round(sy(v), 1), "label": f"{v:,.0f}"} for v in nice_ticks(emin, emax)]

    return {
        "width": SCATTER_W,
        "height": SCATTER_H,
        "plot_left": SCATTER_MARGIN["left"],
        "plot_top": SCATTER_MARGIN["top"],
        "plot_right": SCATTER_W - SCATTER_MARGIN["right"],
        "plot_bottom": SCATTER_H - SCATTER_MARGIN["bottom"],
        "points_mlp": points_mlp,
        "path_mlp": path_mlp,
        "points_cocomo": points_cocomo,
        "path_cocomo": path_cocomo,
        "xticks": xticks,
        "yticks": yticks,
    }


def build_mode_donut(rows):
    """rows: [(label, value, css_class), ...] en el orden categorico fijo
    (organic, semidetached, embedded). None si el total es cero."""
    total = sum(v for _, v, _ in rows)
    if not total:
        return None

    r = DONUT_R
    circumference = 2 * math.pi * r
    segments = []
    offset = 0.0
    for label, value, css_class in rows:
        frac = value / total
        length = frac * circumference
        gap = 2.0 if value else 0.0
        segments.append({
            "label": label,
            "value": value,
            "css_class": css_class,
            "pct": round(frac * 100),
            "dasharray": f"{max(length - gap, 0):.1f} {max(circumference - length + gap, 0):.1f}",
            "dashoffset": round(-offset, 1),
        })
        offset += length

    return {
        "size": DONUT_SIZE,
        "cx": DONUT_SIZE / 2,
        "cy": DONUT_SIZE / 2,
        "r": r,
        "stroke": DONUT_STROKE,
        "segments": segments,
        "total": total,
    }


def build_local_error_bars(rows):
    """rows: lista de dicts {"name", "error_mlp", "error_cocomo"} (% de
    error absoluto frente al esfuerzo real, ya calculados en la vista).
    Barra doble por proyecto (MLP y COCOMO) para comparar que metodo se
    adapta mejor al contexto local -- un histograma de conteo (0/0/5) no
    distingue si el problema es del MLP o de ambos metodos frente a los
    datos de entrenamiento. Incluye una marca de referencia en 25%
    (umbral de PRED25). None si no hay proyectos con ambas estimaciones."""
    if not rows:
        return None

    vmax = max(max(r["error_mlp"], r["error_cocomo"]) for r in rows)
    vmax = max(vmax, 25) * 1.05

    def es_label(value):
        # Formato "61,7%" calculado en Python: evita mezclar {% localize off %}
        # (necesario para que el width/left en el style="" use punto decimal,
        # ver nota en style.css/README sobre LANGUAGE_CODE='es') con texto que
        # si debe mostrar coma, dentro del mismo bloque de un {% for %}.
        return f"{value:.1f}".replace(".", ",")

    bars = [
        {
            "name": r["name"],
            "error_mlp": round(r["error_mlp"], 1),
            "error_cocomo": round(r["error_cocomo"], 1),
            "error_mlp_label": es_label(r["error_mlp"]),
            "error_cocomo_label": es_label(r["error_cocomo"]),
            "pct_mlp": round(r["error_mlp"] / vmax * 100, 1),
            "pct_cocomo": round(r["error_cocomo"] / vmax * 100, 1),
        }
        for r in rows
    ]
    return {"bars": bars, "threshold_pct": round(25 / vmax * 100, 1)}
