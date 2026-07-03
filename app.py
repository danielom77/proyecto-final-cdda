"""Turbipredict dashboard/backend.

Arquitectura final:
- Render ejecuta este Flask como backend Python persistente.
- Vercel actúa como URL pública/proxy mediante rewrites.
- El modelo Gradient Boosting se entrena una vez por proceso y queda en caché con lru_cache.
- /api/dashboard devuelve métricas y datos para Plotly/Leaflet.
- /api/predict calcula predicciones nuevas con los valores enviados por el usuario.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_modelo_2018_2025.csv"
TARGET_COLUMN = "high_turbidity_t1"
TEST_START_DATE = pd.Timestamp("2025-01-01")
MODEL_NAME = "Gradient Boosting"
THRESHOLD = 0.50

AUTHORS = [
    "Jackeline Esther Ortega Palomino",
    "Jhoryi Joel Iman Juarez",
    "Judith Susana Carlos Rivera",
    "Daniel Abraham Mercado Valdez",
]

STATION_COORDS = {
    "08180800": {"lat": 29.262183, "lon": -98.581407, "name": "San Antonio River, tramo aguas arriba"},
    "08181500": {"lat": 29.264128, "lon": -98.490849, "name": "San Antonio River, San Antonio"},
    "08181800": {"lat": 29.222185, "lon": -98.355846, "name": "Medina River, afluente del sistema San Antonio"},
    "08188060": {"lat": 28.848694, "lon": -97.737139, "name": "Tramo asociado al sistema Guadalupe"},
    "08188500": {"lat": 28.649286, "lon": -97.384858, "name": "Guadalupe River at Cuero"},
}

PREFERRED_FEATURES = [
    "Turbidez",
    "Caudal",
    "Nivel_Agua",
    "Precipitacion",
    "Temperatura",
    "Conductividad",
    "Oxigeno_Disuelto",
    "pH",
    "Turbidez_lag1",
    "Caudal_lag1",
    "Nivel_Agua_lag1",
    "Precipitacion_lag1",
    "Precipitacion_acum_3d",
    "Precipitacion_acum_7d",
    "Caudal_delta_1d",
    "Nivel_delta_1d",
    "mes",
    "dia_anio_sin",
    "dia_anio_cos",
    "P95_Turbidez",
]

DISPLAY_COLUMNS = [
    "Fecha",
    "site_id",
    "Turbidez",
    "Turbidez_t1_obs",
    "P95_Turbidez",
    "Caudal",
    "Nivel_Agua",
    "Precipitacion",
    "Precipitacion_acum_3d",
    "Precipitacion_acum_7d",
    TARGET_COLUMN,
    "periodo",
]


def as_float(value: Any, decimals: int = 3) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), decimals)
    except Exception:
        return None


def as_int(value: Any) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def as_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def normalize_site_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 8:
        text = text.zfill(8)
    return text


def clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in records:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                item[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, np.generic):
                item[key] = value.item()
            elif value is None:
                item[key] = None
            else:
                try:
                    item[key] = None if pd.isna(value) else value
                except Exception:
                    item[key] = value
        cleaned.append(item)
    return cleaned


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATA_PATH.name} en {BASE_DIR}")

    df = pd.read_csv(DATA_PATH)
    if "Fecha" not in df.columns:
        raise ValueError("El dataset debe contener una columna 'Fecha'.")

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).copy()

    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].apply(normalize_site_id)
    else:
        df["site_id"] = "sin_estacion"

    numeric_candidates = [col for col in df.columns if col not in {"Fecha", "site_id", "periodo"}]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    elif {"Turbidez_t1_obs", "P95_Turbidez"}.issubset(df.columns):
        df[TARGET_COLUMN] = (df["Turbidez_t1_obs"] > df["P95_Turbidez"]).astype(int)
    else:
        df[TARGET_COLUMN] = 0

    return df.sort_values(["Fecha", "site_id"]).reset_index(drop=False).rename(columns={"index": "_source_index"})


def get_features(df: pd.DataFrame) -> list[str]:
    features = [col for col in PREFERRED_FEATURES if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if features:
        return features
    excluded = {TARGET_COLUMN, "Fecha", "site_id", "periodo", "_source_index"}
    return [col for col in df.select_dtypes(include=[np.number]).columns if col not in excluded]


@lru_cache(maxsize=1)
def model_artifacts() -> dict[str, Any]:
    df = load_data().copy()

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
    except Exception as exc:
        return {"available": False, "reason": f"No se pudo importar scikit-learn: {exc}"}

    features = get_features(df)
    if not features:
        return {"available": False, "reason": "No hay variables predictoras numéricas disponibles."}

    required_cols = ["_source_index", "Fecha", "site_id"] + features + [TARGET_COLUMN]
    model_df = df[required_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=features + [TARGET_COLUMN]).copy()
    if model_df.empty or model_df[TARGET_COLUMN].nunique() < 2:
        return {"available": False, "reason": "No hay suficientes clases para entrenar y evaluar el modelo."}

    train_df = model_df[model_df["Fecha"] < TEST_START_DATE].copy()
    test_df = model_df[model_df["Fecha"] >= TEST_START_DATE].copy()
    split_strategy = "Validación temporal 2018–2024 → 2025"

    if train_df.empty or test_df.empty or train_df[TARGET_COLUMN].nunique() < 2 or test_df[TARGET_COLUMN].nunique() < 2:
        model_df = model_df.sort_values("Fecha")
        cut = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:cut].copy()
        test_df = model_df.iloc[cut:].copy()
        split_strategy = "Fallback: corte temporal 80/20 por fecha"

    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[features]
    y_test = test_df[TARGET_COLUMN].astype(int)

    medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_test, probabilities)
    except Exception:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_test, probabilities)
    except Exception:
        pr_auc = None

    importance = (
        pd.DataFrame({"feature": features, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(14)
    )

    eval_df = test_df[["_source_index", "Fecha", "site_id", TARGET_COLUMN] + features].copy()
    optional_cols = [col for col in ["Turbidez_t1_obs", "P95_Turbidez", "periodo"] if col in df.columns]
    if optional_cols:
        extra = df.set_index("_source_index").loc[eval_df["_source_index"], optional_cols].reset_index(drop=True)
        for col in optional_cols:
            eval_df[col] = extra[col].values
    eval_df["probability"] = probabilities
    eval_df["y_pred"] = y_pred
    eval_df["y_true"] = y_test.values

    try:
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_curve_points = [{"fpr": as_float(x, 4), "tpr": as_float(y, 4)} for x, y in zip(fpr, tpr)]
    except Exception:
        roc_curve_points = []

    try:
        prec, rec, _ = precision_recall_curve(y_test, probabilities)
        pr_curve_points = [{"recall": as_float(r, 4), "precision": as_float(p, 4)} for p, r in zip(prec, rec)]
    except Exception:
        pr_curve_points = []

    event_rate = float(y_test.mean() * 100) if len(y_test) else 0
    metrics = {
        "records": int(len(y_test)),
        "real_events": int(y_test.sum()),
        "detected_events": int(tp),
        "missed_events": int(fn),
        "alerts": int(tp + fp),
        "false_alarms": int(fp),
        "precision": as_float(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": as_float(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "accuracy": as_float(accuracy_score(y_test, y_pred) * 100, 2),
        "f1": as_float(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc": as_float(roc_auc, 3),
        "pr_auc": as_float(pr_auc, 3),
        "threshold": THRESHOLD,
        "event_rate": as_float(event_rate, 2),
        "train_start": as_date(train_df["Fecha"].min()),
        "train_end": as_date(train_df["Fecha"].max()),
        "test_start": as_date(test_df["Fecha"].min()),
        "test_end": as_date(test_df["Fecha"].max()),
        "split_strategy": split_strategy,
        "feature_count": len(features),
        "model_name": MODEL_NAME,
        "backend_mode": "Modelo entrenado y cacheado en Render; predicciones calculadas en vivo por /api/predict.",
    }

    return {
        "available": True,
        "features": features,
        "medians": medians,
        "model": model,
        "eval_df": eval_df,
        "metrics": metrics,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": clean_records(importance.to_dict(orient="records")),
        "roc_curve": roc_curve_points,
        "pr_curve": pr_curve_points,
    }


def filtered_eval(selected_site: str) -> pd.DataFrame:
    artifacts = model_artifacts()
    if not artifacts.get("available"):
        return pd.DataFrame()
    df = artifacts["eval_df"].copy()
    if selected_site != "all":
        df = df[df["site_id"].astype(str) == selected_site].copy()
    return df


def series_data(df: pd.DataFrame, variable: str) -> list[dict[str, Any]]:
    if df.empty or variable not in df.columns:
        return []
    data = (
        df.dropna(subset=["Fecha", variable])
        .groupby("Fecha", as_index=False)[variable]
        .mean()
        .sort_values("Fecha")
    )
    return [{"date": as_date(row["Fecha"]), "value": as_float(row[variable], 3)} for _, row in data.iterrows()]


def histogram_data(df: pd.DataFrame, variable: str) -> list[dict[str, Any]]:
    if df.empty or variable not in df.columns:
        return []
    s = df[variable].dropna().astype(float)
    if s.empty:
        return []
    upper = s.quantile(0.99)
    s = s[s <= upper]
    if len(s) > 1500:
        s = s.sample(1500, random_state=42)
    return [{"value": as_float(v, 3)} for v in s.tolist()]


def scatter_data(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or not {"Caudal", "Turbidez"}.issubset(df.columns):
        return []
    data = df.dropna(subset=["Caudal", "Turbidez"]).copy()
    if data.empty:
        return []
    q_flow = data["Caudal"].quantile(0.99)
    q_turb = data["Turbidez"].quantile(0.99)
    data = data[(data["Caudal"] <= q_flow) & (data["Turbidez"] <= q_turb)]
    if len(data) > 1000:
        data = data.sample(1000, random_state=42)
    return [
        {
            "flow": as_float(row["Caudal"], 3),
            "turbidity": as_float(row["Turbidez"], 3),
            "date": as_date(row["Fecha"]),
            "site_id": row["site_id"],
            "event": int(row.get("y_true", 0)),
        }
        for _, row in data.iterrows()
    ]


def monthly_events(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    data = df.copy()
    data["month"] = data["Fecha"].dt.to_period("M").astype(str)
    grouped = data.groupby("month", as_index=False).agg(real_events=("y_true", "sum"), alerts=("y_pred", "sum"))
    return [
        {"month": row["month"], "real_events": int(row["real_events"]), "alerts": int(row["alerts"])}
        for _, row in grouped.iterrows()
    ]


def station_summary(eval_df: pd.DataFrame) -> list[dict[str, Any]]:
    if eval_df.empty:
        return []
    rows = []
    for site, group in eval_df.groupby("site_id"):
        real = int(group["y_true"].sum())
        alerts = int(group["y_pred"].sum())
        tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
        fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
        max_col = "Turbidez_t1_obs" if "Turbidez_t1_obs" in group.columns else "Turbidez"
        rows.append({
            "site_id": site,
            "records": int(len(group)),
            "real_events": real,
            "alerts": alerts,
            "detected": tp,
            "missed": fn,
            "false_alarms": fp,
            "prob_mean": as_float(group["probability"].mean() * 100, 2),
            "prob_max": as_float(group["probability"].max() * 100, 2),
            "turbidity_max_t1": as_float(group[max_col].max(), 2) if max_col in group.columns else None,
            "lat": STATION_COORDS.get(site, {}).get("lat"),
            "lon": STATION_COORDS.get(site, {}).get("lon"),
            "description": STATION_COORDS.get(site, {}).get("name", "Estación USGS"),
        })
    return sorted(rows, key=lambda item: item["site_id"])


@app.route("/")
def index() -> str:
    html = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Turbipredict | Ciencia de Datos Ambiental</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    :root { --navy:#08344a; --teal:#0b8a8f; --ink:#0f2a44; --muted:#52657c; --bg:#f4f7fb; --panel:#ffffff; --line:#dce6f1; --red:#d62828; --orange:#f77f00; --green:#087e8b; --shadow:0 18px 42px rgba(15,42,68,.10); }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter, Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }
    .hero { padding:42px 5vw 78px; color:white; background:linear-gradient(135deg,#073047,#087e8b); }
    .hero-top { display:flex; justify-content:space-between; gap:16px; align-items:center; flex-wrap:wrap; }
    .pill { display:inline-flex; align-items:center; padding:9px 15px; border-radius:999px; border:1px solid rgba(255,255,255,.28); background:rgba(255,255,255,.12); font-weight:800; font-size:14px; }
    h1 { max-width:1160px; margin:30px 0 16px; font-size:clamp(38px,5vw,72px); line-height:.98; letter-spacing:-2px; }
    .subtitle { max-width:1260px; font-size:19px; line-height:1.55; color:#eef9fb; margin:0; }
    .authors { margin-top:20px; display:flex; flex-wrap:wrap; gap:10px; }
    .author { padding:8px 11px; border-radius:10px; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); font-size:13px; font-weight:700; }
    main { padding:0 5vw 60px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:24px; margin:26px 0; box-shadow:var(--shadow); overflow:hidden; }
    .first { margin-top:-44px; }
    h2 { margin:0 0 14px; font-size:26px; color:var(--navy); }
    h3 { margin:0 0 8px; color:var(--navy); font-size:17px; }
    p { color:var(--muted); line-height:1.62; }
    .controls { display:grid; grid-template-columns:1fr 1fr 220px; gap:16px; align-items:end; margin-top:16px; }
    label { display:block; font-weight:800; margin:0 0 7px; color:#2d4057; }
    select,input,button { width:100%; border-radius:13px; padding:13px 14px; border:1px solid var(--line); font:inherit; background:white; }
    button { border:none; background:var(--teal); color:white; font-weight:900; cursor:pointer; }
    button:hover { filter:brightness(.95); }
    .status { margin-top:12px; color:var(--muted); }
    .method-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:18px; }
    .method { background:#f8fafc; border:1px solid var(--line); border-radius:14px; padding:14px; }
    .method small { display:block; color:var(--muted); font-weight:800; margin-bottom:6px; }
    .method strong { color:var(--navy); font-size:16px; }
    .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(160px,1fr)); gap:14px; margin-top:20px; }
    .metric { border:1px solid var(--line); border-radius:14px; padding:18px; text-align:center; background:#f8fafc; }
    .value { font-size:30px; font-weight:900; color:var(--navy); }
    .metric.alert .value { color:var(--red); } .metric.ok .value { color:var(--green); } .metric.warn .value { color:var(--orange); }
    .label { margin-top:7px; color:var(--muted); font-size:13px; }
    .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
    .chart { width:100%; height:390px; }
    #map { width:100%; height:520px; border-radius:14px; border:1px solid var(--line); overflow:hidden; }
    .map-legend { background:white; padding:12px 14px; border:1px solid #cbd5e1; border-radius:12px; box-shadow:0 8px 24px rgba(15,42,68,.18); color:#0f2a44; font-size:13px; line-height:1.65; }
    .map-legend .dot { display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:7px; vertical-align:middle; }
    .map-legend .line { display:inline-block; width:28px; height:0; border-top:4px solid #2a9d8f; margin-right:7px; vertical-align:middle; }
    .map-legend .box { display:inline-block; width:18px; height:12px; border:2px solid #1f78b4; background:rgba(31,120,180,.08); margin-right:7px; vertical-align:middle; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:14px; }
    table { width:100%; border-collapse:collapse; min-width:1050px; font-size:13px; }
    th { background:var(--navy); color:white; padding:12px; text-align:center; }
    td { border-bottom:1px solid var(--line); border-right:1px solid var(--line); padding:10px; text-align:center; }
    tr:nth-child(even) td { background:#f8fafc; }
    .prediction { display:grid; grid-template-columns:1.45fr .75fr; gap:22px; align-items:stretch; }
    .form-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
    .risk-card { border:1px solid var(--line); border-radius:16px; background:#f8fafc; padding:22px; }
    .risk-label { display:inline-flex; border-radius:999px; padding:8px 12px; font-weight:900; font-size:13px; background:#dcfce7; color:#166534; }
    .risk-medium { background:#ffedd5; color:#9a3412; } .risk-high { background:#fee2e2; color:#991b1b; }
    .risk-value { margin:20px 0 10px; font-size:58px; font-weight:950; color:var(--navy); line-height:1; }

    .cm-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:16px; }
    .cm-cell { background:#f8fafc; border:1px solid var(--line); border-radius:14px; padding:18px; text-align:center; }
    .cm-cell .cm-value { font-size:30px; font-weight:950; color:var(--navy); }
    .cm-cell .cm-label { margin-top:6px; color:var(--muted); font-size:13px; }
    .audit-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .audit { background:#f8fafc; border:1px solid var(--line); border-radius:14px; padding:16px; }
    .audit b { color:var(--navy); }
    .foot-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; }
    .mini { background:#f8fafc; border:1px solid var(--line); border-radius:14px; padding:17px; }
    .note { font-size:13px; color:var(--muted); margin-top:8px; }
    footer { text-align:center; color:var(--muted); padding:28px 0 6px; }
    @media(max-width:1100px){ .grid-2,.prediction{grid-template-columns:1fr;} .metric-grid{grid-template-columns:repeat(2,1fr);} .controls,.form-grid,.foot-grid,.method-strip{grid-template-columns:1fr;} }
  </style>
</head>
<body>
  <section class="hero">
    <div class="hero-top"><span class="pill">Ciencia de Datos Ambiental</span><span class="pill">Flask · GitHub · Vercel · Render</span></div>
    <h1>Dashboard de predicción de eventos extremos de turbidez</h1>
    <p class="subtitle">Sistema San Antonio–Guadalupe, Texas · Modelo seleccionado: <b>Gradient Boosting</b>. La herramienta estima la probabilidad de superar el P95 local de turbidez al día siguiente, con validación temporal y predicción interactiva en vivo.</p>
    <div class="authors">{% for author in authors %}<span class="author">{{ author }}</span>{% endfor %}</div>
  </section>

  <main>
    <section class="card first">
      <h2>Resumen ejecutivo del modelo</h2>
      <p>El objetivo del proyecto es clasificar si al día siguiente ocurrirá un evento extremo de turbidez. La evaluación se plantea como una validación temporal: entrenar con pasado, evaluar en futuro y analizar el equilibrio entre eventos detectados y falsas alarmas.</p>
      <div class="controls">
        <div><label>Estación USGS</label><select id="station"><option value="all">Todas las estaciones</option></select></div>
        <div><label>Variable exploratoria</label><select id="variable"></select></div>
        <div><button id="refresh">Actualizar dashboard</button></div>
      </div>
      <div id="status" class="status">Cargando dashboard...</div>
      <div class="method-strip">
        <div class="method"><small>Modelo</small><strong id="s-model">—</strong></div>
        <div class="method"><small>Validación</small><strong id="s-split">—</strong></div>
        <div class="method"><small>Variable objetivo</small><strong>Turbidez t+1 &gt; P95 local</strong></div>
        <div class="method"><small>Predicción</small><strong>Backend en vivo + caché</strong><div class="note">El modelo se entrena una vez al iniciar/cargar el backend y cada simulación llama a <code>/api/predict</code>.</div></div>
      </div>
      <div class="metric-grid">
        <div class="metric"><div id="m-records" class="value">—</div><div class="label">Registros evaluados</div></div>
        <div class="metric alert"><div id="m-real" class="value">—</div><div class="label">Eventos reales</div></div>
        <div class="metric ok"><div id="m-detected" class="value">—</div><div class="label">Eventos detectados</div></div>
        <div class="metric alert"><div id="m-missed" class="value">—</div><div class="label">Eventos no detectados</div></div>
        <div class="metric"><div id="m-alerts" class="value">—</div><div class="label">Alertas emitidas</div></div>
        <div class="metric warn"><div id="m-fp" class="value">—</div><div class="label">Falsas alarmas</div></div>
        <div class="metric"><div id="m-precision" class="value">—</div><div class="label">Precisión</div></div>
        <div class="metric"><div id="m-recall" class="value">—</div><div class="label">Recall</div></div>
        <div class="metric"><div id="m-accuracy" class="value">—</div><div class="label">Accuracy</div></div>
        <div class="metric"><div id="m-f1" class="value">—</div><div class="label">F1-score</div></div>
        <div class="metric"><div id="m-roc" class="value">—</div><div class="label">ROC-AUC</div></div>
        <div class="metric"><div id="m-pr" class="value">—</div><div class="label">PR-AUC</div></div>
      </div>
    </section>

    <section class="card"><h2>Mapa interactivo de estaciones USGS</h2><p>Ubicación de estaciones del sistema San Antonio–Guadalupe. El tamaño del marcador resume la cantidad de eventos reales y el color diferencia estaciones con y sin eventos extremos en el periodo de prueba.</p><div id="map"></div></section>

    <section class="card"><h2>Exploración temporal y probabilidad de alerta</h2><div class="grid-2"><div><h3>Probabilidad diaria de evento extremo</h3><p>La línea punteada representa el umbral de alerta utilizado para clasificar evento/no evento.</p><div id="chart-prob" class="chart"></div></div><div><h3>Serie temporal de la variable seleccionada</h3><p>Promedio diario para la estación o conjunto seleccionado.</p><div id="chart-line" class="chart"></div></div></div></section>
    <section class="card"><h2>Distribución y relaciones ambientales</h2><div class="grid-2"><div><h3>Distribución de la variable principal</h3><p>Se recorta el percentil 99 solo para legibilidad gráfica; los valores extremos se conservan en tablas y métricas.</p><div id="chart-hist" class="chart"></div></div><div><h3>Relación caudal–turbidez</h3><p>Los puntos rojos corresponden a eventos extremos reales.</p><div id="chart-scatter" class="chart"></div></div></div></section>
    <section class="card"><h2>Evaluación del modelo e importancia de variables</h2><div class="grid-2"><div><h3>Eventos reales vs alertas por mes</h3><p>Compara ocurrencias observadas y alertas emitidas.</p><div id="chart-month" class="chart"></div></div><div><h3>Importancia de variables</h3><p>Predictores con mayor peso relativo en el modelo Gradient Boosting.</p><div id="chart-importance" class="chart"></div></div></div></section>
    <section class="card"><h2>Matriz de confusión y lectura operacional</h2><p>La matriz separa los aciertos y errores del modelo en el periodo de prueba. En eventos extremos, los falsos negativos representan eventos no anticipados y los falsos positivos representan falsas alarmas.</p><div class="cm-grid"><div class="cm-cell"><div id="cm-tn" class="cm-value">—</div><div class="cm-label">Verdaderos negativos<br>No evento correctamente clasificado</div></div><div class="cm-cell"><div id="cm-fp" class="cm-value">—</div><div class="cm-label">Falsos positivos<br>Alertas sin evento real</div></div><div class="cm-cell"><div id="cm-fn" class="cm-value">—</div><div class="cm-label">Falsos negativos<br>Eventos no detectados</div></div><div class="cm-cell"><div id="cm-tp" class="cm-value">—</div><div class="cm-label">Verdaderos positivos<br>Eventos detectados</div></div></div><div class="audit-grid" style="margin-top:18px"><div class="audit"><b>Interpretación para gestión ambiental.</b><p>Un falso negativo es más delicado si el objetivo es prevención, porque implica no anticipar un evento de alta turbidez. Un falso positivo es una falsa alarma y debe mantenerse bajo para no saturar decisiones operativas.</p></div><div class="audit"><b>Rigor de evaluación.</b><p>El modelo se evalúa con separación temporal, no con mezcla aleatoria de fechas. Esto aproxima el uso real: entrenar con pasado y estimar riesgo en un periodo futuro.</p></div></div></section>
    <section class="card"><h2>Curvas de desempeño</h2><div class="grid-2"><div><h3>Curva ROC</h3><p>Capacidad general para separar eventos y no eventos.</p><div id="chart-roc" class="chart"></div></div><div><h3>Curva Precision–Recall</h3><p>Más informativa cuando los eventos extremos son minoritarios.</p><div id="chart-pr" class="chart"></div></div></div></section>
    <section class="card"><h2>Resumen por estación</h2><p>Comparación territorial de registros evaluados, eventos reales, alertas y probabilidades estimadas.</p><div class="table-wrap"><table id="station-table"></table></div></section>

    <section class="card"><h2>Predicción interactiva de riesgo</h2><p>Ingresa condiciones actuales para estimar la probabilidad de un evento extremo de turbidez al día siguiente. Los campos vacíos se completan con medianas históricas del conjunto de entrenamiento.</p><div class="prediction"><div><div class="form-grid"><div><label>Turbidez actual</label><input id="p-turbidez" type="number" step="0.01" /></div><div><label>Caudal</label><input id="p-caudal" type="number" step="0.01" /></div><div><label>Nivel de agua</label><input id="p-nivel" type="number" step="0.01" /></div><div><label>Precipitación diaria</label><input id="p-prec" type="number" step="0.01" /></div><div><label>Precip. acum. 3 días</label><input id="p-prec3" type="number" step="0.01" /></div><div><label>Precip. acum. 7 días</label><input id="p-prec7" type="number" step="0.01" /></div><div><label>Temperatura</label><input id="p-temp" type="number" step="0.01" placeholder="Opcional" /></div><div><label>pH</label><input id="p-ph" type="number" step="0.01" placeholder="Opcional" /></div><div><label>Conductividad</label><input id="p-cond" type="number" step="0.01" placeholder="Opcional" /></div></div><div style="margin-top:14px"><button id="predict">Predecir riesgo</button></div></div><div class="risk-card"><span id="risk-label" class="risk-label">Sin predicción</span><div id="risk-value" class="risk-value">—</div><p id="risk-text">Completa los campos y ejecuta una predicción.</p><p class="note">La simulación no usa valores estáticos: envía los datos al backend y calcula la probabilidad con el modelo entrenado.</p></div></div></section>
    <section class="card"><h2>Últimos registros evaluados</h2><div class="table-wrap"><table id="records-table"></table></div></section>
    <section class="card"><h2>Fuentes de datos, autoría y alcance</h2><div class="foot-grid"><div class="mini"><h3>Fuentes</h3><p>USGS Water Data para variables hidrológicas y fisicoquímicas; gridMET para precipitación diaria; procesamiento reproducible en Python.</p></div><div class="mini"><h3>Modelo</h3><p>Gradient Boosting con validación temporal. La predicción estima probabilidad de superar el P95 local de turbidez al día siguiente.</p></div><div class="mini"><h3>Alcance</h3><p>Es una herramienta analítica de apoyo; no reemplaza sistemas oficiales de alerta ni validación hidrológica operativa.</p></div></div></section>
    <footer>Proyecto Final · Ciencia de Datos Ambiental · Grupo 4</footer>
  </main>
<script>
let map=null, markersLayer=null, optionsLoaded=false;
const stationSelect=document.getElementById('station'), variableSelect=document.getElementById('variable'), statusBox=document.getElementById('status');
const plotConfig={responsive:true, displaylogo:false}; const grid='#dbe5ef'; const baseLayout={margin:{t:18,r:24,b:58,l:64}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{family:'Arial', color:'#0f2a44'}, hovermode:'closest'};
function fmt(v,digits=2){ if(v===null||v===undefined||Number.isNaN(Number(v))) return '—'; return Number(v).toLocaleString('es-PE',{maximumFractionDigits:digits}); }
function setText(id,value){ document.getElementById(id).textContent=value; } function pct(v){ return v===null||v===undefined?'—':`${fmt(v,2)}%`; } function layout(extra={}){ return Object.assign({}, baseLayout, extra); }
function initMap(){ if(map) return; map=L.map('map').setView([29.05,-98.1],9); L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{maxZoom:20, attribution:'&copy; OpenStreetMap &copy; CARTO'}).addTo(map); markersLayer=L.layerGroup().addTo(map); L.polygon([[29.364128,-98.731407],[29.364128,-97.264858],[28.529286,-97.264858],[28.529286,-98.731407]],{color:'#1f78b4', fillColor:'#1f78b4', fillOpacity:.08, weight:2}).addTo(map); L.polyline([[29.262183,-98.581407],[29.264128,-98.490849],[29.222185,-98.355846],[28.848694,-97.737139],[28.649286,-97.384858]],{color:'#2a9d8f', weight:4, opacity:.85}).addTo(map); const legend=L.control({position:'bottomleft'}); legend.onAdd=function(){ const div=L.DomUtil.create('div','map-legend'); div.innerHTML='<b>Leyenda</b><br><span class="dot" style="background:#d62828"></span>Estación con eventos extremos<br><span class="dot" style="background:#2b6cb0"></span>Estación sin eventos extremos<br><span class="line"></span>Eje referencial San Antonio–Guadalupe<br><span class="box"></span>Área aproximada de análisis<br><small>Mayor tamaño = más eventos reales</small>'; return div; }; legend.addTo(map); }
function loadOptions(data){ if(optionsLoaded) return; data.stations.forEach(s=>{let o=document.createElement('option'); o.value=s; o.textContent=s; stationSelect.appendChild(o);}); data.variables.forEach(v=>{let o=document.createElement('option'); o.value=v; o.textContent=v; variableSelect.appendChild(o);}); variableSelect.value=data.selected.variable||'Turbidez'; optionsLoaded=true; }
function updateMetrics(m){ setText('m-records',fmt(m.records,0)); setText('m-real',fmt(m.real_events,0)); setText('m-detected',fmt(m.detected_events,0)); setText('m-missed',fmt(m.missed_events,0)); setText('m-alerts',fmt(m.alerts,0)); setText('m-fp',fmt(m.false_alarms,0)); setText('m-precision',pct(m.precision)); setText('m-recall',pct(m.recall)); setText('m-accuracy',pct(m.accuracy)); setText('m-f1',pct(m.f1)); setText('m-roc',fmt(m.roc_auc,3)); setText('m-pr',fmt(m.pr_auc,3)); setText('s-model',m.model_name||'Gradient Boosting'); setText('s-split',`${m.train_start}–${m.train_end} / ${m.test_start}–${m.test_end}`); const cm=m.confusion_matrix||{}; setText('cm-tn',fmt(cm.tn,0)); setText('cm-fp',fmt(cm.fp,0)); setText('cm-fn',fmt(cm.fn,0)); setText('cm-tp',fmt(cm.tp,0)); }
function drawMap(stations){ initMap(); markersLayer.clearLayers(); const bounds=[]; stations.forEach(s=>{ if(!s.lat||!s.lon) return; const radius=7+Math.min(18,(s.real_events||0)*0.7); const color=(s.real_events||0)>0?'#d62828':'#2b6cb0'; const marker=L.circleMarker([s.lat,s.lon],{radius,color,fillColor:color,fillOpacity:.82,weight:2}); marker.bindPopup(`<b>Estación USGS ${s.site_id}</b><br>${s.description||''}<br><br><b>Registros:</b> ${s.records}<br><b>Eventos reales:</b> ${s.real_events}<br><b>Eventos detectados:</b> ${s.detected}<br><b>Falsas alarmas:</b> ${s.false_alarms}<br><b>Prob. máxima:</b> ${s.prob_max??'—'}%<br><b>Turbidez máxima t+1:</b> ${s.turbidity_max_t1??'—'}`); marker.bindTooltip(`USGS ${s.site_id}`); marker.addTo(markersLayer); bounds.push([s.lat,s.lon]); }); if(bounds.length) map.fitBounds(bounds,{padding:[40,40]}); }
function drawProbability(data){ const threshold=(data.metrics.threshold||.5)*100; const trace={x:data.probability_series.map(d=>d.date), y:data.probability_series.map(d=>d.probability*100), type:'scatter', mode:'lines', name:'Probabilidad', line:{color:'#087e8b',width:3}, hovertemplate:'%{x}<br>Probabilidad: %{y:.1f}%<extra></extra>'}; const th={x:data.probability_series.map(d=>d.date), y:data.probability_series.map(_=>threshold), type:'scatter', mode:'lines', name:'Umbral', line:{color:'#d62828',width:2,dash:'dash'}, hoverinfo:'skip'}; Plotly.newPlot('chart-prob',[trace,th],layout({yaxis:{title:'Probabilidad',ticksuffix:'%',range:[0,100],gridcolor:grid},xaxis:{type:'date',gridcolor:grid}}),plotConfig); }
function drawLine(data){ const variable=data.selected.variable; const trace={x:data.line.map(d=>d.date),y:data.line.map(d=>d.value),type:'scatter',mode:'lines',name:variable,line:{color:'#08344a',width:2.8},hovertemplate:'%{x}<br>'+variable+': %{y:.2f}<extra></extra>'}; Plotly.newPlot('chart-line',[trace],layout({xaxis:{type:'date',gridcolor:grid},yaxis:{title:variable,gridcolor:grid}}),plotConfig); }
function drawHist(data){ const trace={x:data.histogram.map(d=>d.value),type:'histogram',nbinsx:35,marker:{color:'#08344a'},hovertemplate:'Valor: %{x}<br>Frecuencia: %{y}<extra></extra>'}; Plotly.newPlot('chart-hist',[trace],layout({xaxis:{title:data.selected.variable,type:'linear',gridcolor:grid},yaxis:{title:'Frecuencia',gridcolor:grid},bargap:.04}),plotConfig); }
function drawScatter(data){ const normal=data.scatter.filter(d=>!d.event), extreme=data.scatter.filter(d=>d.event); const traces=[{x:normal.map(d=>d.flow),y:normal.map(d=>d.turbidity),type:'scatter',mode:'markers',name:'No extremo',marker:{color:'#087e8b',size:7,opacity:.65},hovertemplate:'Caudal: %{x:.2f}<br>Turbidez: %{y:.2f}<extra></extra>'},{x:extreme.map(d=>d.flow),y:extreme.map(d=>d.turbidity),type:'scatter',mode:'markers',name:'Evento extremo',marker:{color:'#d62828',size:9,opacity:.8},hovertemplate:'Caudal: %{x:.2f}<br>Turbidez: %{y:.2f}<extra></extra>'}]; Plotly.newPlot('chart-scatter',traces,layout({xaxis:{title:'Caudal',type:'linear',gridcolor:grid},yaxis:{title:'Turbidez',type:'linear',gridcolor:grid}}),plotConfig); }
function drawMonth(data){ const months=data.monthly_events.map(d=>d.month); const a={x:months,y:data.monthly_events.map(d=>d.real_events),type:'bar',name:'Eventos reales',marker:{color:'#08344a'}}; const b={x:months,y:data.monthly_events.map(d=>d.alerts),type:'bar',name:'Alertas',marker:{color:'#f77f00'}}; Plotly.newPlot('chart-month',[a,b],layout({barmode:'group',xaxis:{type:'category',gridcolor:grid},yaxis:{title:'Conteo',gridcolor:grid}}),plotConfig); }
function drawImportance(data){ const imp=(data.feature_importance||[]).slice().reverse(); const trace={x:imp.map(d=>d.importance),y:imp.map(d=>d.feature),type:'bar',orientation:'h',marker:{color:'#087e8b'},hovertemplate:'%{y}<br>Importancia: %{x:.3f}<extra></extra>'}; Plotly.newPlot('chart-importance',[trace],layout({xaxis:{title:'Importancia relativa',type:'linear',gridcolor:grid},yaxis:{type:'category',automargin:true,gridcolor:grid}}),plotConfig); }
function drawCurves(data){ const roc=data.roc_curve||[], pr=data.pr_curve||[]; Plotly.newPlot('chart-roc',[{x:roc.map(d=>d.fpr),y:roc.map(d=>d.tpr),type:'scatter',mode:'lines',name:'ROC',line:{color:'#087e8b',width:3}},{x:[0,1],y:[0,1],type:'scatter',mode:'lines',name:'Azar',line:{color:'#94a3b8',dash:'dash'}}],layout({xaxis:{title:'False Positive Rate',range:[0,1],gridcolor:grid},yaxis:{title:'True Positive Rate',range:[0,1],gridcolor:grid}}),plotConfig); Plotly.newPlot('chart-pr',[{x:pr.map(d=>d.recall),y:pr.map(d=>d.precision),type:'scatter',mode:'lines',name:'PR',line:{color:'#f77f00',width:3}}],layout({xaxis:{title:'Recall',range:[0,1],gridcolor:grid},yaxis:{title:'Precision',range:[0,1],gridcolor:grid}}),plotConfig); }
function makeTable(id,cols,rows){ const table=document.getElementById(id); if(!rows||!rows.length){table.innerHTML='<tbody><tr><td>No hay registros.</td></tr></tbody>';return;} table.innerHTML='<thead><tr>'+cols.map(c=>`<th>${c.label}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${r[c.key]??''}</td>`).join('')+'</tr>').join('')+'</tbody>'; }
function updateTables(data){ makeTable('station-table',[{key:'site_id',label:'Estación USGS'},{key:'records',label:'Registros evaluados'},{key:'real_events',label:'Eventos reales'},{key:'detected',label:'Detectados'},{key:'alerts',label:'Alertas emitidas'},{key:'missed',label:'No detectados'},{key:'false_alarms',label:'Falsas alarmas'},{key:'prob_mean',label:'Prob. media (%)'},{key:'prob_max',label:'Prob. máxima (%)'},{key:'turbidity_max_t1',label:'Turbidez máxima t+1'}],data.station_summary); makeTable('records-table',data.table_columns.map(c=>({key:c,label:c})),data.table_rows); }
function fillPredictionDefaults(data){ const d=data.prediction_defaults||{}; const map={Turbidez:'p-turbidez',Caudal:'p-caudal',Nivel_Agua:'p-nivel',Precipitacion:'p-prec',Precipitacion_acum_3d:'p-prec3',Precipitacion_acum_7d:'p-prec7'}; Object.entries(map).forEach(([k,id])=>{const el=document.getElementById(id); if(el&&!el.value&&d[k]!==undefined&&d[k]!==null) el.value=d[k];}); }
async function loadDashboard(){ try{ statusBox.textContent='Actualizando dashboard...'; const params=new URLSearchParams({site_id:stationSelect.value,variable:variableSelect.value||'Turbidez'}); const res=await fetch(`/api/dashboard?${params.toString()}`); const data=await res.json(); if(!res.ok||data.error) throw new Error(data.error||'Error de API'); loadOptions(data); updateMetrics(data.metrics); drawMap(data.station_summary); drawProbability(data); drawLine(data); drawHist(data); drawScatter(data); drawMonth(data); drawImportance(data); drawCurves(data); updateTables(data); fillPredictionDefaults(data); statusBox.textContent=`Filtro activo: ${data.selected.site_id==='all'?'Todas las estaciones':data.selected.site_id} · Variable: ${data.selected.variable} · ${data.metrics.backend_mode}`; }catch(e){ console.error(e); statusBox.textContent='No se pudo cargar el dashboard. Revisa /api/health y los logs del backend.'; } }
function readVal(id){ const value=document.getElementById(id).value; return value===''?null:Number(value); }
async function predict(){ const payload={Turbidez:readVal('p-turbidez'),Caudal:readVal('p-caudal'),Nivel_Agua:readVal('p-nivel'),Precipitacion:readVal('p-prec'),Precipitacion_acum_3d:readVal('p-prec3'),Precipitacion_acum_7d:readVal('p-prec7'),Temperatura:readVal('p-temp'),pH:readVal('p-ph'),Conductividad:readVal('p-cond')}; const label=document.getElementById('risk-label'), value=document.getElementById('risk-value'), text=document.getElementById('risk-text'); try{ const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const out=await res.json(); if(!res.ok||!out.available) throw new Error(out.reason||'No disponible'); label.textContent=`Riesgo ${out.risk_level}`; label.className='risk-label'; if(out.risk_level==='Medio') label.classList.add('risk-medium'); if(out.risk_level==='Alto') label.classList.add('risk-high'); value.textContent=`${fmt(out.probability*100,1)}%`; text.textContent=out.message; }catch(e){ label.textContent='Error'; value.textContent='—'; text.textContent=e.message; } }
document.getElementById('refresh').addEventListener('click',loadDashboard); stationSelect.addEventListener('change',loadDashboard); variableSelect.addEventListener('change',loadDashboard); document.getElementById('predict').addEventListener('click',predict); loadDashboard();
</script>
</body>
</html>
    """
    return render_template_string(html, authors=AUTHORS)


@app.route("/api/health")
def health() -> Any:
    status = {
        "status": "ok",
        "base_dir": str(BASE_DIR),
        "data_path": str(DATA_PATH),
        "data_exists": DATA_PATH.exists(),
        "model_mode": "live_cached",
        "message": "El backend entrena el modelo con el dataset al primer request y conserva el objeto en caché de memoria. /api/predict calcula predicciones en vivo.",
    }
    try:
        df = load_data()
        status.update({"rows": int(df.shape[0]), "columns": int(df.shape[1])})
        artifacts = model_artifacts()
        status.update({
            "model_available": bool(artifacts.get("available")),
            "model_name": artifacts.get("metrics", {}).get("model_name"),
            "test_records": artifacts.get("metrics", {}).get("records"),
        })
        if not artifacts.get("available"):
            status["model_error"] = artifacts.get("reason")
    except Exception as exc:
        status.update({"status": "error", "error": str(exc)})
    return jsonify(status)


@app.route("/api/dashboard")
def api_dashboard() -> Any:
    try:
        df_all = load_data()
        artifacts = model_artifacts()
        if not artifacts.get("available"):
            return jsonify({"error": artifacts.get("reason", "Modelo no disponible")}), 500

        selected_site = request.args.get("site_id", "all")
        selected_variable = request.args.get("variable", "Turbidez")
        variables = [col for col in get_features(df_all) if col in df_all.columns]
        if not variables:
            return jsonify({"error": "No hay variables disponibles para graficar."}), 500
        if selected_variable not in variables:
            selected_variable = "Turbidez" if "Turbidez" in variables else variables[0]

        eval_all = artifacts["eval_df"].copy()
        eval_filtered = filtered_eval(selected_site)
        if eval_filtered.empty:
            eval_filtered = eval_all.copy()
            selected_site = "all"

        stations = sorted(eval_all["site_id"].dropna().astype(str).unique().tolist())

        y_true = eval_filtered["y_true"].astype(int)
        y_pred = eval_filtered["y_pred"].astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        precision = 100 * tp / (tp + fp) if (tp + fp) else 0
        recall = 100 * tp / (tp + fn) if (tp + fn) else 0
        accuracy = 100 * (tp + tn) / len(eval_filtered) if len(eval_filtered) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        metrics = dict(artifacts["metrics"])
        metrics.update({
            "records": int(len(eval_filtered)),
            "real_events": int(y_true.sum()),
            "detected_events": tp,
            "missed_events": fn,
            "alerts": int(y_pred.sum()),
            "false_alarms": fp,
            "precision": as_float(precision, 2),
            "recall": as_float(recall, 2),
            "accuracy": as_float(accuracy, 2),
            "f1": as_float(f1, 2),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        })

        table_cols = [c for c in DISPLAY_COLUMNS if c in eval_filtered.columns]
        table_cols += ["probability", "y_true", "y_pred"]
        table_df = eval_filtered.copy().sort_values("Fecha", ascending=False).head(12)
        table_df["Fecha"] = table_df["Fecha"].dt.strftime("%Y-%m-%d")
        table_df["probability"] = (table_df["probability"] * 100).round(2).astype(str) + "%"
        table_rows = clean_records(table_df[table_cols].to_dict(orient="records"))

        prob_series = [
            {"date": as_date(row["Fecha"]), "probability": as_float(row["probability"], 4)}
            for _, row in eval_filtered.sort_values("Fecha").iterrows()
        ]

        return jsonify({
            "selected": {"site_id": selected_site, "variable": selected_variable},
            "stations": stations,
            "variables": variables,
            "metrics": metrics,
            "station_summary": station_summary(eval_all),
            "probability_series": prob_series,
            "line": series_data(eval_filtered, selected_variable),
            "histogram": histogram_data(eval_filtered, selected_variable),
            "scatter": scatter_data(eval_filtered),
            "monthly_events": monthly_events(eval_filtered),
            "feature_importance": artifacts["feature_importance"],
            "roc_curve": artifacts.get("roc_curve", []),
            "pr_curve": artifacts.get("pr_curve", []),
            "prediction_defaults": {k: as_float(v, 2) for k, v in artifacts["medians"].items()},
            "table_columns": table_cols,
            "table_rows": table_rows,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict() -> Any:
    try:
        artifacts = model_artifacts()
        if not artifacts.get("available"):
            return jsonify({"available": False, "reason": artifacts.get("reason", "Modelo no disponible")}), 500

        data = request.get_json(silent=True) or {}
        features = artifacts["features"]
        medians = artifacts["medians"]
        row: dict[str, float] = {}
        for feature in features:
            value = data.get(feature)
            if value is None or value == "":
                row[feature] = float(medians.get(feature, 0.0))
            else:
                try:
                    row[feature] = float(value)
                except Exception:
                    row[feature] = float(medians.get(feature, 0.0))

        X_new = pd.DataFrame([row], columns=features)
        probability = float(artifacts["model"].predict_proba(X_new)[0][1])

        if probability >= 0.70:
            level = "Alto"
            message = "La probabilidad estimada es alta. Se recomienda priorizar monitoreo y revisión operativa para las próximas 24 horas."
        elif probability >= THRESHOLD:
            level = "Medio"
            message = "La probabilidad supera el umbral de alerta del modelo. Conviene mantener seguimiento cercano."
        else:
            level = "Bajo"
            message = "Las condiciones ingresadas sugieren bajo riesgo relativo de turbidez extrema al día siguiente."

        return jsonify({
            "available": True,
            "probability": as_float(probability, 4),
            "risk_level": level,
            "threshold": THRESHOLD,
            "message": message,
            "used_features": row,
            "model_mode": "live_cached_prediction",
        })
    except Exception as exc:
        return jsonify({"available": False, "reason": str(exc)}), 500


@app.route("/api/resumen")
def api_resumen() -> Any:
    df = load_data()
    artifacts = model_artifacts()
    return jsonify({
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nombres_columnas": df.columns.tolist(),
        "objetivo": TARGET_COLUMN,
        "modelo_disponible": bool(artifacts.get("available")),
        "modo_modelo": "entrenamiento en backend con caché + predicción en vivo",
    })


if __name__ == "__main__":
    app.run(debug=True)
