"""Geometria de los widgets del Dashboard: linea de costo en el tiempo y dona
de modos de desarrollo. El ranking y el histograma de error se resuelven con
el mismo patron .bar-track/.bar-fill que ya usa panel.html, calculado en la
vista. Mismo enfoque que charts.py: SVG calculado en Python, sin JS ni
librerias de graficos."""

import math

LINE_W, LINE_H = 560, 200
LINE_MARGIN = {"left": 8, "right": 8, "top": 12, "bottom": 24}

DONUT_SIZE = 180
DONUT_R = 62
DONUT_STROKE = 26


def build_timeline(estimations):
    """estimations: lista ordenada ascendente por created_at, con .cost_bs,
    .project_name y .created_at. None si no hay datos."""
    rows = list(estimations)
    if not rows:
        return None

    values = [r.cost_bs for r in rows]
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmin = 0

    plot_w = LINE_W - LINE_MARGIN["left"] - LINE_MARGIN["right"]
    plot_h = LINE_H - LINE_MARGIN["top"] - LINE_MARGIN["bottom"]
    n = len(rows)

    def sx(i):
        return LINE_MARGIN["left"] + (i / (n - 1) if n > 1 else 0.5) * plot_w

    def sy(v):
        span = vmax - vmin or 1
        return LINE_MARGIN["top"] + plot_h - (v - vmin) / span * plot_h

    points = [
        {
            "x": round(sx(i), 1),
            "y": round(sy(r.cost_bs), 1),
            "label": r.project_name,
            "value": r.cost_bs,
            "date": r.created_at,
        }
        for i, r in enumerate(rows)
    ]
    baseline = LINE_MARGIN["top"] + plot_h
    path = "M " + " L ".join(f"{p['x']},{p['y']}" for p in points)
    area = path + f" L {points[-1]['x']},{baseline} L {points[0]['x']},{baseline} Z"

    return {
        "width": LINE_W,
        "height": LINE_H,
        "plot_bottom": baseline,
        "points": points,
        "path": path,
        "area": area,
        "first": points[0],
        "last": points[-1],
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


ERROR_BUCKETS = [
    ("≤ 10 %", 0, 10, "good"),
    ("10 – 25 %", 10, 25, "warning"),
    ("> 25 %", 25, None, "serious"),
]


def build_error_histogram(local_projects):
    """local_projects: iterable con .error_pct (puede ser None). None si
    ningun proyecto tiene estimacion del MLP todavia."""
    errors = [p.error_pct for p in local_projects if p.error_pct is not None]
    if not errors:
        return None

    buckets = []
    for label, lo, hi, status in ERROR_BUCKETS:
        count = sum(1 for e in errors if e >= lo and (hi is None or e < hi))
        buckets.append({"label": label, "count": count, "status": status})
    vmax = max(b["count"] for b in buckets) or 1
    for b in buckets:
        b["pct"] = round(b["count"] / vmax * 100)

    return {"buckets": buckets, "total": len(errors)}
