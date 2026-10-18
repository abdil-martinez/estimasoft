# EstimaSoft

Prototipo web para estimar el esfuerzo y costo de desarrollo de software
mediante un perceptrón multicapa (MLP), comparado contra COCOMO intermedio.
Proyecto de tesis de grado — Carrera de Informática, UMSA.

- **Backend**: Django
- **Modelo**: TensorFlow / Keras (ya entrenado, incluido en `models/`)
- **Base de datos**: PostgreSQL

## Requisitos

- Python 3.12 (otras 3.11/3.12 probablemente funcionan)
- Docker (para levantar PostgreSQL rápido, sin instalarlo en el sistema)

## Pasos para probarlo

1. Cloná el repositorio y entrá a la carpeta:

   ```bash
   git clone <URL_DEL_REPO>
   cd tesis
   ```

2. Creá el entorno virtual e instalá dependencias:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # en Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Levantá PostgreSQL con Docker (una sola vez; las próximas veces alcanza
   con `docker start estimasoft-postgres`):

   ```bash
   docker run -d --name estimasoft-postgres \
     -e POSTGRES_DB=estimasoft \
     -e POSTGRES_USER=estimasoft \
     -e POSTGRES_PASSWORD=estimasoft \
     -p 5433:5432 \
     postgres:15-alpine
   ```

4. (Opcional) Copiá `web/.env.example` a `web/.env`. Los valores por
   defecto en `settings.py` ya coinciden con el comando de arriba, así que
   este paso no es obligatorio para probarlo local.

5. Migrá la base de datos y levantá el servidor:

   ```bash
   cd web
   python manage.py migrate
   python manage.py runserver 8001
   ```

6. Abrí `http://localhost:8001/` en el navegador.

El modelo ya viene entrenado (`models/mlp_final.keras`, `scaler.json`,
`metrics.json`), así que no hace falta reentrenar nada para probar el
prototipo. Si querés reproducir el entrenamiento desde cero, los scripts
están en `scripts/` (`build_dataset.py` → `train_eval.py` →
`train_final_model.py`), documentados en el Capítulo III de la tesis.

## Estructura del proyecto

```
data/        datasets públicos (COCOMO81, NASA93) y procesados
models/      modelo MLP entrenado + métricas de validación
scripts/     pipeline de preparación de datos y entrenamiento
sql/         scripts SQL auxiliares
web/         proyecto Django (app "estimator")
```
