"""
Fase CRISP-DM: Preparacion de datos.

Unifica COCOMO81 (63 proyectos, PROMISE) y NASA93 (93 proyectos, PROMISE) en un
solo dataset con esquema comun: modo de desarrollo, tamano (KLOC), 15 factores de
costo de COCOMO intermedio como multiplicadores numericos y esfuerzo real
(persona-mes).

COCOMO81 ya trae los factores como multiplicadores numericos continuos (ej. 0.88,
1.15, pero tambien valores intermedios como 0.94 o 1.08 que no caen en los 6
niveles estandar de Boehm — el dataset original de 63 proyectos usa calificaciones
interpoladas). NASA93 trae los factores como categorias (vl,l,n,h,vh,xh); se
convierten a su multiplicador estandar (tabla de COCOMO intermedio, Boehm 1981)
para quedar en la misma escala numerica que COCOMO81. Esta direccion de mapeo es
exacta (categoria -> multiplicador tabulado), a diferencia de la inversa.

El prototipo web sigue pidiendo al usuario una categoria (Muy bajo..Muy alto) por
factor; esa categoria se traduce al multiplicador estandar antes de alimentar el
modelo, usando la misma tabla.

Entrada:  data/raw/cocomo81.csv, data/raw/nasa93.arff
Salida:   data/processed/dataset_combined.csv
"""

import csv
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

COST_DRIVERS = [
    "rely", "data", "cplx", "time", "stor", "virt", "turn",
    "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
]

# Tabla estandar de multiplicadores de COCOMO intermedio (Boehm, 1981, cap. 8).
# "-" = nivel no definido para ese factor.
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

MODE_MAP = {"embedded": "embedded", "organic": "organic", "semidetached": "semidetached"}


def rating_to_multiplier(driver: str, label: str) -> float:
    return RATING_TABLE[driver][label]


def load_cocomo81():
    rows = []
    with open(RAW / "cocomo81.csv", newline="") as f:
        for row in csv.DictReader(f):
            rec = {
                "source": "COCOMO81",
                "project_id": f"COCOMO81-{row['num']}",
                "mode": MODE_MAP[row["dev_mode"]],
                "kloc": float(row["loc"]),
                "effort_pm": float(row["actual"]),
            }
            for d in COST_DRIVERS:
                rec[d] = float(row[d])
            rows.append(rec)
    return rows


def load_nasa93():
    text = (RAW / "nasa93.arff").read_text()
    data_section = text.split("@data")[1]
    attr_names = re.findall(r"@attribute\s+(\S+)", text.split("@data")[0])

    rows = []
    for line in data_section.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        values = line.split(",")
        rec_raw = dict(zip(attr_names, values))
        rec = {
            "source": "NASA93",
            "project_id": f"NASA93-{rec_raw['recordnumber']}",
            "mode": MODE_MAP[rec_raw["mode"]],
            "kloc": float(rec_raw["equivphyskloc"]),
            "effort_pm": float(rec_raw["act_effort"]),
        }
        for d in COST_DRIVERS:
            rec[d] = rating_to_multiplier(d, rec_raw[d])
        rows.append(rec)
    return rows


def main():
    cocomo = load_cocomo81()
    nasa = load_nasa93()
    combined = cocomo + nasa

    fieldnames = ["project_id", "source", "mode", "kloc"] + COST_DRIVERS + ["effort_pm"]
    out_path = OUT / "dataset_combined.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"COCOMO81: {len(cocomo)} proyectos")
    print(f"NASA93:   {len(nasa)} proyectos")
    print(f"Total:    {len(combined)} proyectos -> {out_path}")


if __name__ == "__main__":
    main()
