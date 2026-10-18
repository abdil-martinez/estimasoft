"""
Fase CRISP-DM: Despliegue (parte 1).

Entrena el MLP final (arquitectura y features fijadas en train_eval.py, ver ese
script para la justificacion y la comparacion contra COCOMO/regresion lineal)
sobre el 100% del dataset combinado, y guarda los artefactos que el prototipo
Django necesita para hacer inferencia en linea:

  models/mlp_final.keras   -- red entrenada
  models/scaler.json       -- media/desvio de StandardScaler (para reproducir
                               la normalizacion sin depender de sklearn en runtime)
  models/metrics.json      -- metricas de validacion cruzada (de eval_metrics.csv)
                               + metadata de arquitectura, para mostrar en la
                               pantalla "Modelo y evaluacion"

Este script asume que scripts/train_eval.py ya corrio y dejo
data/processed/eval_metrics.csv, eval_arch_selection.csv.
"""

import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dataset_combined.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

COST_DRIVERS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]
MODE_ORDER = {"organic": 0, "semidetached": 1, "embedded": 2}

# Debe coincidir con CANDIDATES en train_eval.py: se usa para traducir el
# nombre ganador de eval_arch_selection.csv a sus hiperparametros.
CANDIDATES = {
    "mlp_4": {"hidden": (4,), "dropout": 0.0, "l2": 1e-3},
    "mlp_8": {"hidden": (8,), "dropout": 0.1, "l2": 1e-3},
    "mlp_8_reg": {"hidden": (8,), "dropout": 0.2, "l2": 1e-2},
    "mlp_8_4": {"hidden": (8, 4), "dropout": 0.1, "l2": 1e-3},
}
RANDOM_STATE = 42


def build_mlp(n_inputs, hidden, dropout, l2):
    reg = keras.regularizers.l2(l2) if l2 > 0 else None
    layers = [keras.layers.Input(shape=(n_inputs,))]
    for units in hidden:
        layers.append(keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if dropout > 0:
            layers.append(keras.layers.Dropout(dropout))
    layers.append(keras.layers.Dense(1))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def pick_final_architecture():
    arch_counts = pd.read_csv(PROCESSED / "eval_arch_selection.csv", index_col=0)
    winner = arch_counts["veces_elegida"].idxmax()
    return winner, CANDIDATES[winner]


def main():
    arch_name, FINAL_ARCH = pick_final_architecture()
    print(f"Arquitectura final elegida (mas votada en la CV anidada): {arch_name} -> {FINAL_ARCH}")

    df = pd.read_csv(DATA)
    df["mode_ord"] = df["mode"].map(MODE_ORDER)
    df["eaf"] = df[COST_DRIVERS].prod(axis=1)
    df["log_kloc"] = np.log(df["kloc"])
    df["log_eaf"] = np.log(df["eaf"])

    feature_cols = ["log_kloc", "mode_ord", "log_eaf"]
    X = df[feature_cols].to_numpy(dtype=float)
    y_log = np.log(df["effort_pm"].to_numpy(dtype=float))

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = build_mlp(X_scaled.shape[1], **FINAL_ARCH)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True,
    )
    model.fit(
        X_scaled, y_log, validation_split=0.15,
        epochs=200, batch_size=8, verbose=0, callbacks=[early_stop],
    )
    model.save(MODELS_DIR / "mlp_final.keras")

    scaler_payload = {
        "feature_order": feature_cols,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    (MODELS_DIR / "scaler.json").write_text(json.dumps(scaler_payload, indent=2))

    eval_metrics = pd.read_csv(PROCESSED / "eval_metrics.csv", index_col="metodo").to_dict(orient="index")
    arch_selection = pd.read_csv(PROCESSED / "eval_arch_selection.csv", index_col=0).to_dict()["veces_elegida"]
    hypothesis_test = json.loads((PROCESSED / "eval_hypothesis_test.json").read_text())

    metrics_payload = {
        "dataset": {
            "cocomo81": int((df["source"] == "COCOMO81").sum()),
            "nasa93": int((df["source"] == "NASA93").sum()),
            "total": int(len(df)),
        },
        "architecture": {
            "hidden_layers": list(FINAL_ARCH["hidden"]),
            "inputs": feature_cols,
            "dropout": FINAL_ARCH["dropout"],
            "l2": FINAL_ARCH["l2"],
            "optimizer": "Adam (lr=0.001)",
            "output": "log(esfuerzo en persona-mes)",
            "selection_counts": arch_selection,
        },
        "cv_metrics": eval_metrics,
        "hypothesis_test": hypothesis_test,
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    print(f"Modelo final guardado en {MODELS_DIR}")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
