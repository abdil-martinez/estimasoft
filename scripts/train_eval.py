"""
Fase CRISP-DM: Modelado y Evaluacion.

Entrena y evalua, con validacion cruzada k-fold anidada, tres metodos sobre el
dataset combinado (COCOMO81 + NASA93, scripts/build_dataset.py):

  1. COCOMO intermedio: formula parametrica E = a * KLOC^b * EAF (no se entrena,
     solo usa las constantes de Boehm 1981 segun el modo de desarrollo).
  2. Regresion lineal: baseline entrenado sobre log(esfuerzo).
  3. Perceptron multicapa (Keras): red elegida por validacion cruzada interna
     entre varias arquitecturas candidatas (objetivo especifico 3 del perfil:
     "disenar, entrenar y validar distintas arquitecturas... seleccionando la de
     mejor desempeno"), en vez de fijar una sola arquitectura a mano.

Diseno de la validacion (anidada, para no filtrar el fold de prueba hacia la
seleccion de arquitectura):
  - CV externa de 10 particiones -> metricas finales reportadas (out-of-fold).
  - Dentro de cada entrenamiento externo, un split train/val 80/20 elige la
    mejor arquitectura candidata por MAE de validacion; esa arquitectura se
    reentrena sobre todo el conjunto de entrenamiento externo (con su propio
    early stopping) antes de predecir el fold de prueba.

Features de los 15 factores de costo: se usa log(EAF) (producto de los 15
multiplicadores, en logaritmo) como una sola entrada en vez de los 15 factores
por separado. Se probo primero con los 15 por separado (en logaritmo) y
empeoraba incluso a la regresion lineal (MAE 319 -> 508 en el mismo split de
validacion cruzada): con solo ~140 proyectos de entrenamiento, 15 coeficientes
individuales son demasiados grados de libertad y el modelo sobreajusta ruido
en vez de senal real. Comprimir los 15 factores en log(EAF) replica la propia
estructura multiplicativa de COCOMO (E = a*KLOC^b*EAF) y resulto ser mucho mas
eficiente en datos; es el motivo de que la regresion lineal "a mano" (con
log(EAF) como feature unica) ya superaba al MLP original que usaba los 17
campos crudos.

Reporta, por metodo: MAE (persona-mes), MMRE, MdMRE, PRED(25); y la prueba de
Wilcoxon (errores absolutos pareados) entre MLP y COCOMO intermedio, tal como
pide la hipotesis del perfil de tesis.

Entrada:  data/processed/dataset_combined.csv
Salida:   data/processed/eval_metrics.csv, data/processed/eval_predictions.csv,
          data/processed/eval_arch_selection.csv
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "dataset_combined.csv"
OUT_DIR = ROOT / "data" / "processed"

COST_DRIVERS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]
MODE_ORDER = {"organic": 0, "semidetached": 1, "embedded": 2}
COCOMO_PARAMS = {
    "organic": (2.4, 1.05),
    "semidetached": (3.0, 1.12),
    "embedded": (3.6, 1.20),
}

N_SPLITS = 10
RANDOM_STATE = 42

# Arquitecturas candidatas a comparar por validacion interna, todas sobre el
# input compacto de 3 features (log_kloc, modo, log_eaf) -- ver nota arriba
# sobre por que no se usan los 15 factores por separado. Todas son chicas
# porque con ~140 proyectos de entrenamiento una red grande (ej. el 32,16 del
# mockup original) sobreajusta de inmediato.
CANDIDATES = [
    {"name": "mlp_4", "hidden": (4,), "dropout": 0.0, "l2": 1e-3},
    {"name": "mlp_8", "hidden": (8,), "dropout": 0.1, "l2": 1e-3},
    {"name": "mlp_8_reg", "hidden": (8,), "dropout": 0.2, "l2": 1e-2},
    {"name": "mlp_8_4", "hidden": (8, 4), "dropout": 0.1, "l2": 1e-3},
]


def load_data():
    df = pd.read_csv(DATA)
    df["mode_ord"] = df["mode"].map(MODE_ORDER)
    df["eaf"] = df[COST_DRIVERS].prod(axis=1)
    df["log_kloc"] = np.log(df["kloc"])
    df["log_eaf"] = np.log(df["eaf"])
    return df


def cocomo_predict(df):
    a = df["mode"].map(lambda m: COCOMO_PARAMS[m][0])
    b = df["mode"].map(lambda m: COCOMO_PARAMS[m][1])
    return a * (df["kloc"] ** b) * df["eaf"]


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


def fit_mlp(cand, X_train, y_train, X_val=None, y_val=None, seed=0, epochs=180, patience=20):
    tf.keras.utils.set_random_seed(seed)
    model = build_mlp(X_train.shape[1], cand["hidden"], cand["dropout"], cand["l2"])
    if X_val is not None:
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
        )
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=8, verbose=0, callbacks=[early_stop],
        )
    else:
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True,
        )
        model.fit(
            X_train, y_train, validation_split=0.15,
            epochs=epochs, batch_size=8, verbose=0, callbacks=[early_stop],
        )
    return model


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    abs_err = np.abs(actual - pred)
    rel_err = abs_err / actual
    return {
        "MAE": abs_err.mean(),
        "MMRE": rel_err.mean(),
        "MdMRE": np.median(rel_err),
        "PRED25": float((rel_err <= 0.25).mean()),
    }


def select_architecture(X_train, y_log_train, seed):
    """Elige la arquitectura candidata con menor MAE (en escala real) sobre un
    split de validacion interno 80/20 del conjunto de entrenamiento externo."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_log_train, test_size=0.2, random_state=seed,
    )
    scaler = StandardScaler().fit(X_fit)
    X_fit_s, X_val_s = scaler.transform(X_fit), scaler.transform(X_val)

    best_name, best_mae = None, np.inf
    for cand in CANDIDATES:
        model = fit_mlp(cand, X_fit_s, y_fit, X_val_s, y_val, seed=seed, epochs=150, patience=15)
        pred = np.exp(model.predict(X_val_s, verbose=0).ravel())
        mae = np.abs(np.exp(y_val) - pred).mean()
        if mae < best_mae:
            best_name, best_mae = cand["name"], mae
    return best_name


def main():
    df = load_data()
    feature_cols = ["log_kloc", "mode_ord", "log_eaf"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["effort_pm"].to_numpy(dtype=float)
    y_log = np.log(y)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = {"cocomo": np.zeros_like(y), "linreg": np.zeros_like(y), "mlp": np.zeros_like(y)}
    chosen_archs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_log_train = y_log[train_idx]

        # 1. COCOMO intermedio: formula fija, no requiere entrenamiento.
        oof["cocomo"][test_idx] = cocomo_predict(df_test)

        # 2. Regresion lineal sobre log(KLOC), log(EAF) y modo.
        X_lin_train = np.column_stack([
            np.log(df_train["kloc"]), np.log(df_train["eaf"]), df_train["mode_ord"],
        ])
        X_lin_test = np.column_stack([
            np.log(df_test["kloc"]), np.log(df_test["eaf"]), df_test["mode_ord"],
        ])
        linreg = LinearRegression().fit(X_lin_train, y_log_train)
        oof["linreg"][test_idx] = np.exp(linreg.predict(X_lin_test))

        # 3. Perceptron multicapa: seleccion de arquitectura + reentrenamiento final.
        chosen = select_architecture(X_train, y_log_train, seed=RANDOM_STATE + fold)
        cand = next(c for c in CANDIDATES if c["name"] == chosen)
        chosen_archs.append(chosen)

        scaler = StandardScaler().fit(X_train)
        X_mlp_train = scaler.transform(X_train)
        X_mlp_test = scaler.transform(X_test)
        model = fit_mlp(cand, X_mlp_train, y_log_train, seed=RANDOM_STATE + fold, epochs=180, patience=20)

        log_pred = model.predict(X_mlp_test, verbose=0).ravel()
        oof["mlp"][test_idx] = np.exp(np.clip(log_pred, y_log.min() - 1, y_log.max() + 1))

        print(f"fold {fold}/{N_SPLITS} listo (train={len(train_idx)}, test={len(test_idx)}, arquitectura={chosen})")

    results = {name: metrics(y, pred) for name, pred in oof.items()}
    stat, pvalue = wilcoxon(np.abs(y - oof["mlp"]), np.abs(y - oof["cocomo"]))

    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "metodo"
    metrics_df.to_csv(OUT_DIR / "eval_metrics.csv")

    preds_df = df[["project_id", "source", "effort_pm"]].copy()
    preds_df["pred_cocomo"] = oof["cocomo"]
    preds_df["pred_linreg"] = oof["linreg"]
    preds_df["pred_mlp"] = oof["mlp"]
    preds_df.to_csv(OUT_DIR / "eval_predictions.csv", index=False)

    arch_counts = pd.Series(chosen_archs).value_counts()
    arch_counts.to_csv(OUT_DIR / "eval_arch_selection.csv", header=["veces_elegida"])

    import json
    (OUT_DIR / "eval_hypothesis_test.json").write_text(json.dumps({
        "test": "wilcoxon",
        "comparison": "MLP vs COCOMO intermedio (errores absolutos pareados)",
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "alpha": 0.05,
        "reject_h0": bool(pvalue < 0.05),
        "mlp_better": bool(results["mlp"]["MAE"] < results["cocomo"]["MAE"]),
    }, indent=2))

    print("\n=== Arquitectura elegida por fold (CV interna) ===")
    print(arch_counts.to_string())

    print("\n=== Metricas (out-of-fold, 10-fold CV externa) ===")
    print(metrics_df.round(3).to_string())
    print(f"\nWilcoxon MLP vs COCOMO intermedio (errores absolutos pareados): stat={stat:.3f}, p={pvalue:.4f}")
    mlp_better = results["mlp"]["MAE"] < results["cocomo"]["MAE"]
    if pvalue < 0.05 and mlp_better:
        print("p < 0.05 y MAE(MLP) < MAE(COCOMO) -> se rechaza H0: el MLP comete un error significativamente menor")
    elif pvalue < 0.05 and not mlp_better:
        print("p < 0.05 pero MAE(MLP) >= MAE(COCOMO) -> la diferencia es significativa en contra del MLP, no se cumple Hi")
    else:
        print("p >= 0.05 -> no hay evidencia suficiente para rechazar H0")


if __name__ == "__main__":
    main()
