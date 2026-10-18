"""Carga datos de ejemplo en Postgres (via Django ORM) para poder mostrar
consultas SQL interesantes en DBeaver: varias estimaciones y proyectos locales
con nombres/instituciones plausibles. Los esfuerzos/costos se calculan con el
modelo real (estimator.ml.estimate), no son numeros inventados sueltos.

Uso: desde web/, con el venv activado:
    python manage.py shell < ../scripts/seed_demo_data.py
"""

import random

from estimator import ml
from estimator.models import Estimation, LocalProject

random.seed(7)

# Limpia los 2 registros de prueba de ayer ("Sistema de trámites en línea" / "dd")
LocalProject.objects.filter(institution="dd").delete()

DEFAULT_RATINGS = {
    "rely": "n", "data": "n", "cplx": "n", "time": "n", "stor": "n",
    "virt": "n", "turn": "n", "acap": "n", "aexp": "n", "pcap": "n",
    "vexp": "n", "lexp": "n", "modp": "n", "tool": "n", "sced": "n",
}

ESTIMATIONS = [
    {"name": "Sistema de gestión de matrículas", "kloc": 22.0, "mode": "semidetached", "rate": 10500,
     "ratings": {**DEFAULT_RATINGS, "cplx": "h", "acap": "h"}},
    {"name": "Portal de trámites municipales", "kloc": 35.0, "mode": "semidetached", "rate": 11000,
     "ratings": {**DEFAULT_RATINGS, "cplx": "h", "rely": "h", "sced": "h"}},
    {"name": "App de pedidos para restaurantes", "kloc": 9.5, "mode": "organic", "rate": 9500,
     "ratings": {**DEFAULT_RATINGS, "tool": "h"}},
    {"name": "Sistema de facturación electrónica", "kloc": 28.0, "mode": "semidetached", "rate": 12000,
     "ratings": {**DEFAULT_RATINGS, "rely": "vh", "cplx": "h"}},
    {"name": "Plataforma de historias clínicas", "kloc": 48.0, "mode": "embedded", "rate": 13500,
     "ratings": {**DEFAULT_RATINGS, "rely": "vh", "cplx": "vh", "sced": "h"}},
    {"name": "Sistema de control de inventarios", "kloc": 14.0, "mode": "organic", "rate": 9000,
     "ratings": DEFAULT_RATINGS},
]

LOCAL_PROJECTS = [
    {"name": "Sistema de correspondencia", "inst": "Alcaldía de El Alto", "kloc": 9.2, "mode": "organic",
     "real": 17.0, "cost_real": 170000, "ratings": DEFAULT_RATINGS},
    {"name": "Portal de pagos en línea", "inst": "Cooperativa La Paz Ltda.", "kloc": 21.5, "mode": "semidetached",
     "real": 52.0, "cost_real": 598000, "ratings": {**DEFAULT_RATINGS, "rely": "h", "cplx": "h"}},
    {"name": "App de seguimiento de pedidos", "inst": "Distribuidora Andina", "kloc": 6.8, "mode": "organic",
     "real": 11.5, "cost_real": 103500, "ratings": DEFAULT_RATINGS},
    {"name": "Sistema académico", "inst": "Universidad Privada de Santa Cruz", "kloc": 30.1, "mode": "semidetached",
     "real": 71.0, "cost_real": 781000, "ratings": {**DEFAULT_RATINGS, "cplx": "h", "sced": "h"}},
    {"name": "Módulo de facturación", "inst": "Comercial Cochabamba S.R.L.", "kloc": 4.4, "mode": "organic",
     "real": 8.0, "cost_real": 72000, "ratings": DEFAULT_RATINGS},
]

for e in ESTIMATIONS:
    r = ml.estimate(kloc=e["kloc"], mode=e["mode"], ratings=e["ratings"], rate_per_pm=e["rate"])
    Estimation.objects.create(
        project_name=e["name"], kloc=e["kloc"], mode=e["mode"], rate_per_pm=e["rate"],
        effort_mlp_pm=r["effort_mlp_pm"], effort_cocomo_pm=r["effort_cocomo_pm"],
        cost_bs=r["cost_bs"], duration_months=r["duration_months"], staff_count=r["staff_count"],
        **e["ratings"],
    )

for p in LOCAL_PROJECTS:
    eaf = ml.eaf_from_ratings(p["ratings"])
    mlp_pred = ml.mlp_effort(p["kloc"], p["mode"], eaf)
    LocalProject.objects.create(
        project_name=p["name"], institution=p["inst"], kloc=p["kloc"], mode=p["mode"],
        effort_real_pm=p["real"], cost_real_bs=p["cost_real"], effort_mlp_pm=mlp_pred,
        **p["ratings"],
    )

print(f"Estimation: {Estimation.objects.count()} filas")
print(f"LocalProject: {LocalProject.objects.count()} filas")
