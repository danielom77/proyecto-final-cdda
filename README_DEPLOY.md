# Turbipredict — Ciencia de Datos Ambiental

Dashboard de predicción de eventos extremos de turbidez en el sistema San Antonio–Guadalupe, Texas.

## Arquitectura final

- **Render** ejecuta el backend Python/Flask, carga el dataset, entrena/evalúa Gradient Boosting y atiende `/api/dashboard`, `/api/predict` y `/api/health`.
- **Vercel** se mantiene como URL pública y redirige internamente todas las rutas hacia Render mediante `vercel.json`.

## Archivos necesarios para producción

```text
app.py
requirements.txt
vercel.json
.vercelignore
.gitignore
dataset_modelo_2018_2025.csv
Proyecto_Final_CdDA.ipynb      # opcional, para trazabilidad académica
requirements-dev.txt           # opcional, para ejecutar notebook localmente
```

## Archivos/carpetas que puedes eliminar del repo si ya no usarás Plan B

```text
artifacts/
build_dashboard_artifacts.py
README_PLAN_B.md
placeholder
__pycache__/
```

`Data Original/` puede quedarse localmente, pero `.gitignore` evita subirla a GitHub.

## Render

Configura el Web Service así:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --timeout 120
```

Luego prueba:

```text
https://TU-BACKEND.onrender.com/api/health
https://TU-BACKEND.onrender.com/api/dashboard
```

## Vercel

Cuando Render funcione, edita `vercel.json` y reemplaza:

```text
https://REEMPLAZA-CON-TU-BACKEND-RENDER.onrender.com
```

por tu URL real de Render.

Después:

```bash
git add .
git commit -m "Configuro arquitectura Vercel frontend y Render backend"
git push
```
