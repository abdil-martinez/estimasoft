"""Carga del modelo entrenado (scripts/train_final_model.py) y calculo de
estimaciones: MLP, COCOMO intermedio y las magnitudes derivadas (costo,
duracion, personal) que pide la pantalla 'Nueva estimacion' del prototipo.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .models import COST_DRIVER_FIELDS

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

MODE_ORDER = {"organic": 0, "semidetached": 1, "embedded": 2}
COCOMO_PARAMS = {
    "organic": (2.4, 1.05, 2.5, 0.38),
    "semidetached": (3.0, 1.12, 2.5, 0.35),
    "embedded": (3.6, 1.20, 2.5, 0.32),
}

# Tabla estandar de multiplicadores de COCOMO intermedio (Boehm, 1981), la
# misma que usa scripts/build_dataset.py para convertir NASA93 a escala
# numerica. El formulario del prototipo pide categoria (Muy bajo..Muy alto);
# aqui se traduce a multiplicador antes de alimentar el modelo.
RATING_TABLE = {
    "rely": {"vl": 0.75, "l": 0.88, "n": 1.00, "h": 1.15, "vh": 1.40},
    "data": {"l": 0.94, "n": 1.00, "h": 1.08, "vh": 1.16},
    "cplx": {"vl": 0.70, "l": 0.85, "n": 1.00, "h": 1.15, "vh": 1.30, "xh": 1.65},
    "time": {"n": 1.00, "h": 1.11, "vh": 1.30, "xh": 1.66},
    "stor": {"n": 1.00, "h": 1.06, "vh": 1.21, "xh": 1.56},
    "virt": {"l": 0.87, "n": 1.00, "h": 1.15, "vh": 1.30},
    "turn": {"l": 0.87, "n": 1.00, "h": 1.07, "vh": 1.15},
    "acap": {"vl": 1.46, "l": 1.19, "n": 1.00, "h": 0.86, "vh": 0.71},
    "aexp": {"vl": 1.29, "l": 1.13, "n": 1.00, "h": 0.91, "vh": 0.82},
    "pcap": {"vl": 1.42, "l": 1.17, "n": 1.00, "h": 0.86, "vh": 0.70},
    "vexp": {"vl": 1.21, "l": 1.10, "n": 1.00, "h": 0.90},
    "lexp": {"vl": 1.14, "l": 1.07, "n": 1.00, "h": 0.95},
    "modp": {"vl": 1.24, "l": 1.10, "n": 1.00, "h": 0.91, "vh": 0.82},
    "tool": {"vl": 1.24, "l": 1.10, "n": 1.00, "h": 0.91, "vh": 0.83},
    "sced": {"vl": 1.23, "l": 1.08, "n": 1.00, "h": 1.04, "vh": 1.10},
}


def rating_to_multiplier(driver, label):
    table = RATING_TABLE[driver]
    if label not in table:
        # Si la categoria no esta definida para este factor, usar el nivel
        # valido mas cercano (algunos factores no tienen 'vl' ni 'xh').
        order = ["vl", "l", "n", "h", "vh", "xh"]
        available = [lvl for lvl in order if lvl in table]
        idx = min(range(len(available)), key=lambda i: abs(order.index(available[i]) - order.index(label)))
        label = available[idx]
    return table[label]


def eaf_from_ratings(ratings: dict) -> float:
    eaf = 1.0
    for driver in COST_DRIVER_FIELDS:
        eaf *= rating_to_multiplier(driver, ratings[driver])
    return eaf


@lru_cache(maxsize=1)
def _load_artifacts():
    model_path = MODELS_DIR / "mlp_final.keras"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    scaler = json.loads((MODELS_DIR / "scaler.json").read_text())
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return model, scaler, metrics


def eval_predictions():
    """Predicciones out-of-fold (CV de scripts/train_eval.py) para el scatter
    'esfuerzo real vs. estimado' de la pantalla 'Modelo y evaluacion'."""
    import pandas as pd

    path = PROCESSED_DIR / "eval_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def model_metrics():
    """Metadata de evaluacion (dataset, arquitectura, metricas de CV) para la
    pantalla 'Modelo y evaluacion'. Lanza FileNotFoundError si el modelo
    todavia no fue entrenado (scripts/train_final_model.py)."""
    _, _, metrics = _load_artifacts()
    return metrics


def cocomo_effort(kloc: float, mode: str, eaf: float) -> float:
    a, b, _, _ = COCOMO_PARAMS[mode]
    return a * (kloc ** b) * eaf


def cocomo_duration(effort_pm: float, mode: str) -> float:
    _, _, c, d = COCOMO_PARAMS[mode]
    return c * (effort_pm ** d)


def mlp_effort(kloc: float, mode: str, eaf: float) -> float:
    model, scaler, _ = _load_artifacts()
    features = {
        "log_kloc": np.log(kloc),
        "mode_ord": MODE_ORDER[mode],
        "log_eaf": np.log(eaf),
    }
    x = np.array([[features[name] for name in scaler["feature_order"]]], dtype=float)
    x_scaled = (x - np.array(scaler["mean"])) / np.array(scaler["scale"])
    log_pred = model.predict(x_scaled, verbose=0).ravel()[0]
    return float(np.exp(log_pred))


def estimate(kloc: float, mode: str, ratings: dict, rate_per_pm: float):
    """Devuelve el diccionario completo que necesita la pantalla 'Nueva
    estimacion': esfuerzo (MLP y COCOMO), costo, duracion y personal."""
    eaf = eaf_from_ratings(ratings)

    effort_mlp = mlp_effort(kloc, mode, eaf)
    effort_cocomo = cocomo_effort(kloc, mode, eaf)

    duration = cocomo_duration(effort_mlp, mode)
    staff = effort_mlp / duration if duration else 0.0
    cost = effort_mlp * rate_per_pm

    # Rango probable a partir del MdMRE del MLP en la validacion cruzada (no es
    # un intervalo de confianza formal, es una cota simple basada en el error
    # relativo mediano observado out-of-fold).
    _, _, metrics = _load_artifacts()
    mdmre = metrics["cv_metrics"]["mlp"]["MdMRE"]

    return {
        "eaf": eaf,
        "effort_mlp_pm": effort_mlp,
        "effort_cocomo_pm": effort_cocomo,
        "cost_bs": cost,
        "duration_months": duration,
        "staff_count": staff,
        "range_low_pm": effort_mlp * (1 - mdmre),
        "range_high_pm": effort_mlp * (1 + mdmre),
    }
