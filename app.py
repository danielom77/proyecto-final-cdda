from flask import Flask, jsonify, render_template_string, request
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "dataset_modelo_2018_2025.csv"
TARGET_COLUMN = "high_turbidity_t1"
DATE_COLUMN = "Fecha"
SITE_COLUMN = "site_id"
TEST_YEAR = 2025
COURSE_NAME = "Ciencia de Datos Ambiental"
MODEL_NAME = "Gradient Boosting"

# Coordenadas aproximadas de estaciones USGS usadas para el mapa del sistema San Antonio-Guadalupe.
# Si agregas nuevas estaciones, añade sus coordenadas aquí.
STATION_COORDS = {
    "08180800": {"lat": 29.262183, "lon": -98.581407, "name": "USGS 08180800"},
    "08181500": {"lat": 29.264128, "lon": -98.490849, "name": "USGS 08181500"},
    "08181800": {"lat": 29.222185, "lon": -98.355846, "name": "USGS 08181800"},
    "08188060": {"lat": 28.884000, "lon": -97.737139, "name": "USGS 08188060"},
    "08188500": {"lat": 28.649286, "lon": -97.384858, "name": "USGS 08188500"},
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
    "P95_Turbidez",
    "mes",
    "dia_anio_sin",
    "dia_anio_cos",
]

DISPLAY_COLUMNS = [
    "Fecha",
    "site_id",
    "Turbidez",
    "Caudal",
    "Nivel_Agua",
    "Precipitacion",
    "Temperatura",
    "Conductividad",
    "Oxigeno_Disuelto",
    "pH",
    "Turbidez_t1_obs",
    TARGET_COLUMN,
    "periodo",
]


def is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def safe_float(value, decimals=3):
    if is_missing(value):
        return None
    try:
        return round(float(value), decimals)
    except Exception:
        return None


def safe_int(value):
    if is_missing(value):
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def safe_date(value):
    if is_missing(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def safe_percent(value, decimals=2):
    if is_missing(value):
        return None
    return round(float(value) * 100, decimals)


def clean_records(records):
    cleaned = []
    for row in records:
        item = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                item[key] = value.strftime("%Y-%m-%d")
            elif isinstance(value, np.generic):
                item[key] = value.item()
            elif is_missing(value):
                item[key] = None
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned


@lru_cache(maxsize=1)
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró {DATA_PATH.name} en {DATA_PATH.parent}")

    df = pd.read_csv(DATA_PATH)

    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    if SITE_COLUMN in df.columns:
        df[SITE_COLUMN] = df[SITE_COLUMN].astype(str)

    for col in df.columns:
        if col not in [DATE_COLUMN, SITE_COLUMN, "periodo"]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    return df


def available_features(df):
    features = []
    for col in PREFERRED_FEATURES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    if features:
        return features

    excluded = {TARGET_COLUMN, DATE_COLUMN, SITE_COLUMN, "periodo"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in excluded]


def temporal_train_test_split(model_df):
    """Devuelve índices de entrenamiento, validación y prueba usando 2025 como prueba cuando existe."""
    if DATE_COLUMN in model_df.columns and model_df[DATE_COLUMN].notna().any():
        ordered = model_df.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)

        test_mask = ordered[DATE_COLUMN].dt.year == TEST_YEAR
        if test_mask.sum() >= 30 and ordered.loc[test_mask, TARGET_COLUMN].nunique() >= 2:
            train_valid = ordered.loc[~test_mask]
            test = ordered.loc[test_mask]
        else:
            cut = int(len(ordered) * 0.80)
            train_valid = ordered.iloc[:cut]
            test = ordered.iloc[cut:]

        if len(train_valid) < 50 or len(test) < 20:
            cut = int(len(ordered) * 0.80)
            train_valid = ordered.iloc[:cut]
            test = ordered.iloc[cut:]

        valid_cut = max(int(len(train_valid) * 0.80), 1)
        train = train_valid.iloc[:valid_cut]
        valid = train_valid.iloc[valid_cut:]

        if len(valid) < 20:
            valid = train_valid.tail(max(20, int(len(train_valid) * 0.20)))
            train = train_valid.drop(valid.index)

        return train.index, valid.index, test.index

    from sklearn.model_selection import train_test_split

    y = model_df[TARGET_COLUMN]
    idx_train_valid, idx_test = train_test_split(
        model_df.index,
        test_size=0.20,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )
    y_train_valid = model_df.loc[idx_train_valid, TARGET_COLUMN]
    idx_train, idx_valid = train_test_split(
        idx_train_valid,
        test_size=0.20,
        random_state=42,
        stratify=y_train_valid if y_train_valid.nunique() == 2 else None,
    )
    return idx_train, idx_valid, idx_test


def balanced_sample_weight(y):
    y = pd.Series(y)
    counts = y.value_counts().to_dict()
    total = len(y)
    weights = []
    for value in y:
        class_count = counts.get(value, 1)
        weights.append(total / (len(counts) * class_count))
    return np.array(weights)


@lru_cache(maxsize=1)
def model_artifacts():
    df = load_data().copy()

    if TARGET_COLUMN not in df.columns:
        return {"available": False, "reason": f"No existe la columna objetivo '{TARGET_COLUMN}'."}

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            fbeta_score,
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
        )
    except Exception as exc:
        return {"available": False, "reason": f"No se pudo importar scikit-learn: {exc}"}

    features = available_features(df)
    if not features:
        return {"available": False, "reason": "No hay variables numéricas suficientes para entrenar el modelo."}

    keep_cols = features + [TARGET_COLUMN]
    if DATE_COLUMN in df.columns:
        keep_cols.append(DATE_COLUMN)
    if SITE_COLUMN in df.columns:
        keep_cols.append(SITE_COLUMN)
    if "Turbidez_t1_obs" in df.columns:
        keep_cols.append("Turbidez_t1_obs")

    model_df = df[keep_cols].replace([np.inf, -np.inf], np.nan).copy()
    model_df = model_df.dropna(subset=[TARGET_COLUMN])

    # Imputación simple por mediana para variables predictoras.
    medians = model_df[features].median(numeric_only=True).to_dict()
    model_df[features] = model_df[features].fillna(medians)

    if model_df.empty or model_df[TARGET_COLUMN].nunique() < 2:
        return {"available": False, "reason": "La variable objetivo no tiene ambas clases suficientes."}

    try:
        idx_train, idx_valid, idx_test = temporal_train_test_split(model_df)
    except Exception as exc:
        return {"available": False, "reason": f"No se pudo crear la partición temporal: {exc}"}

    if len(idx_train) == 0 or len(idx_valid) == 0 or len(idx_test) == 0:
        return {"available": False, "reason": "La partición entrenamiento-validación-prueba quedó vacía."}

    X_train = model_df.loc[idx_train, features]
    y_train = model_df.loc[idx_train, TARGET_COLUMN]
    X_valid = model_df.loc[idx_valid, features]
    y_valid = model_df.loc[idx_valid, TARGET_COLUMN]
    X_test = model_df.loc[idx_test, features]
    y_test = model_df.loc[idx_test, TARGET_COLUMN]

    if y_train.nunique() < 2:
        return {"available": False, "reason": "El conjunto de entrenamiento tiene una sola clase."}

    clf = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.045,
        max_depth=3,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=42,
    )

    clf.fit(X_train, y_train, sample_weight=balanced_sample_weight(y_train))

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    candidate_thresholds = np.linspace(0.05, 0.85, 81)

    best_threshold = 0.50
    best_score = -1
    for threshold in candidate_thresholds:
        pred = (valid_prob >= threshold).astype(int)
        score = fbeta_score(y_valid, pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    # Reentrenamos con entrenamiento + validación y evaluamos solo en prueba temporal.
    idx_train_full = list(idx_train) + list(idx_valid)
    X_train_full = model_df.loc[idx_train_full, features]
    y_train_full = model_df.loc[idx_train_full, TARGET_COLUMN]

    final_model = GradientBoostingClassifier(
        n_estimators=180,
        learning_rate=0.045,
        max_depth=3,
        min_samples_leaf=8,
        subsample=0.85,
        random_state=42,
    )
    final_model.fit(X_train_full, y_train_full, sample_weight=balanced_sample_weight(y_train_full))

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = roc_auc_score(y_test, test_prob)
    except Exception:
        roc_auc = None

    try:
        pr_auc = average_precision_score(y_test, test_prob)
    except Exception:
        pr_auc = None

    test_meta = model_df.loc[idx_test].copy()
    test_meta["y_true"] = y_test.astype(int)
    test_meta["y_pred"] = test_pred.astype(int)
    test_meta["probability"] = test_prob

    feature_importance = (
        pd.DataFrame({"feature": features, "importance": final_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(14)
    )

    if DATE_COLUMN in model_df.columns:
        train_start = safe_date(model_df.loc[idx_train_full, DATE_COLUMN].min())
        train_end = safe_date(model_df.loc[idx_train_full, DATE_COLUMN].max())
        test_start = safe_date(model_df.loc[idx_test, DATE_COLUMN].min())
        test_end = safe_date(model_df.loc[idx_test, DATE_COLUMN].max())
    else:
        train_start = train_end = test_start = test_end = ""

    return {
        "available": True,
        "model": final_model,
        "features": features,
        "medians": medians,
        "threshold": safe_float(best_threshold, 3),
        "metrics": {
            "accuracy": safe_float(accuracy_score(y_test, test_pred), 4),
            "precision": safe_float(precision_score(y_test, test_pred, zero_division=0), 4),
            "recall": safe_float(recall_score(y_test, test_pred, zero_division=0), 4),
            "f1": safe_float(f1_score(y_test, test_pred, zero_division=0), 4),
            "roc_auc": safe_float(roc_auc, 4),
            "pr_auc": safe_float(pr_auc, 4),
            "test_size": int(len(y_test)),
            "event_rate": safe_percent(y_test.mean(), 2),
            "train_period": f"{train_start} a {train_end}",
            "test_period": f"{test_start} a {test_end}",
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": clean_records(feature_importance.to_dict(orient="records")),
        "test_predictions": test_meta,
    }


def filtered_df(df, selected_site):
    if selected_site != "all" and SITE_COLUMN in df.columns:
        return df[df[SITE_COLUMN].astype(str) == str(selected_site)].copy()
    return df.copy()


def station_summary_from_predictions(pred_df):
    if pred_df is None or pred_df.empty or SITE_COLUMN not in pred_df.columns:
        return []

    rows = []
    for site, group in pred_df.groupby(SITE_COLUMN):
        events_real = int(group["y_true"].sum()) if "y_true" in group.columns else 0
        events_pred = int(group["y_pred"].sum()) if "y_pred" in group.columns else 0
        true_pos = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum()) if {"y_true", "y_pred"}.issubset(group.columns) else 0
        missed = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum()) if {"y_true", "y_pred"}.issubset(group.columns) else 0
        false_alarms = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum()) if {"y_true", "y_pred"}.issubset(group.columns) else 0
        max_t1 = safe_float(group["Turbidez_t1_obs"].max(), 2) if "Turbidez_t1_obs" in group.columns else None

        rows.append({
            "site_id": str(site),
            "records": int(len(group)),
            "real_events": events_real,
            "predicted_alerts": events_pred,
            "detected_events": true_pos,
            "missed_events": missed,
            "false_alarms": false_alarms,
            "mean_probability": safe_percent(group["probability"].mean(), 2) if "probability" in group.columns else None,
            "max_probability": safe_percent(group["probability"].max(), 2) if "probability" in group.columns else None,
            "max_turbidity_t1": max_t1,
        })

    return sorted(rows, key=lambda x: x["site_id"])


def station_map_payload(summary):
    payload = []
    summary_by_site = {str(row["site_id"]): row for row in summary}

    for site, coords in STATION_COORDS.items():
        row = summary_by_site.get(site, {})
        max_prob = row.get("max_probability") or 0
        real_events = row.get("real_events") or 0
        predicted_alerts = row.get("predicted_alerts") or 0
        radius = 7 + min(9, real_events * 0.7 + max_prob / 18)
        color = "#d62828" if predicted_alerts > 0 else "#2b6cb0"

        payload.append({
            "site_id": site,
            "name": coords["name"],
            "lat": coords["lat"],
            "lon": coords["lon"],
            "radius": safe_float(radius, 2),
            "color": color,
            **row,
        })

    return payload


@app.route("/")
def home():
    html = r'''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard de eventos extremos de turbidez</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
        <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>

        <style>
            :root {
                --azul: #0b3954;
                --azul-claro: #087e8b;
                --rojo: #d62828;
                --naranja: #f77f00;
                --verde: #2a9d8f;
                --gris-fondo: #f4f7fb;
                --gris-texto: #4a5568;
                --borde: #e2e8f0;
                --card: #ffffff;
                --texto: #1a202c;
            }

            * { box-sizing: border-box; }

            body {
                margin: 0;
                background-color: var(--gris-fondo);
                color: var(--texto);
                font-family: Arial, Helvetica, sans-serif;
            }

            .hero {
                background: linear-gradient(135deg, #0b3954, #087e8b);
                color: white;
                padding: 46px 62px;
            }

            .hero-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 18px;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }

            .badge {
                display: inline-flex;
                align-items: center;
                border: 1px solid rgba(255,255,255,0.28);
                background: rgba(255,255,255,0.10);
                padding: 8px 12px;
                border-radius: 999px;
                font-weight: 700;
                font-size: 12px;
            }

            .hero h1 {
                margin: 0;
                font-size: clamp(32px, 4vw, 48px);
                font-weight: 700;
                max-width: 1120px;
                line-height: 1.05;
            }

            .hero p {
                margin-top: 14px;
                font-size: 16px;
                line-height: 1.58;
                max-width: 1140px;
                color: #e8f6f8;
            }

            .container {
                padding: 32px 62px 52px 62px;
            }

            .card {
                background: var(--card);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 28px;
                box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
                border: 1px solid var(--borde);
                overflow-x: auto;
            }

            h2 {
                margin-top: 0;
                color: var(--azul);
                font-size: 23px;
            }

            h3 {
                margin: 0 0 6px;
                color: var(--azul);
                font-size: 17px;
            }

            .descripcion {
                color: var(--gris-texto);
                font-size: 15px;
                line-height: 1.6;
                margin-bottom: 18px;
            }

            .controls {
                display: grid;
                grid-template-columns: 1fr 1fr auto;
                gap: 16px;
                align-items: end;
                margin-top: 18px;
            }

            label {
                display: block;
                font-size: 13px;
                color: var(--gris-texto);
                font-weight: 700;
                margin-bottom: 7px;
            }

            select, input, button {
                width: 100%;
                padding: 11px 12px;
                border-radius: 12px;
                border: 1px solid var(--borde);
                font-size: 14px;
                background: white;
            }

            button {
                cursor: pointer;
                border: none;
                background: var(--azul-claro);
                color: white;
                font-weight: 700;
                min-width: 190px;
            }

            button:hover { filter: brightness(0.95); }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 18px;
                margin-top: 20px;
            }

            .metric {
                background: #f8fafc;
                border: 1px solid var(--borde);
                border-radius: 14px;
                padding: 18px;
                text-align: center;
            }

            .metric .value {
                font-size: 29px;
                font-weight: 700;
                color: var(--azul);
            }

            .metric .label {
                margin-top: 6px;
                color: var(--gris-texto);
                font-size: 14px;
            }

            .metric.alerta .value { color: var(--rojo); }
            .metric.ok .value { color: var(--azul-claro); }
            .metric.warning .value { color: var(--naranja); }

            .grid-2 {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 22px;
            }

            .plot-card {
                background: white;
                border: 1px solid var(--borde);
                border-radius: 15px;
                padding: 18px;
                min-height: 410px;
            }

            .chart {
                width: 100%;
                height: 360px;
            }

            #station-map {
                width: 100%;
                height: 560px;
                border-radius: 14px;
                border: 1px solid var(--borde);
                overflow: hidden;
            }

            .legend {
                background: white;
                border: 2px solid #cbd5e0;
                border-radius: 10px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.55;
                box-shadow: 0 3px 10px rgba(0,0,0,0.20);
            }

            .tabla {
                border-collapse: collapse;
                width: 100%;
                font-size: 13.5px;
            }

            .tabla th {
                background-color: var(--azul);
                color: white;
                padding: 10px;
                text-align: center;
                border: 1px solid #dbe4ee;
                white-space: nowrap;
            }

            .tabla td {
                padding: 9px;
                text-align: center;
                border: 1px solid #dbe4ee;
                white-space: nowrap;
            }

            .tabla tr:nth-child(even) { background-color: #f8fafc; }

            .cm-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-top: 18px;
            }

            .cm-cell {
                border: 1px solid var(--borde);
                border-radius: 14px;
                background: #f8fafc;
                padding: 18px;
                text-align: center;
            }

            .cm-cell .value {
                font-size: 30px;
                font-weight: 700;
                color: var(--azul);
            }

            .cm-cell .label {
                font-size: 13px;
                color: var(--gris-texto);
                margin-top: 5px;
            }

            .prediction-layout {
                display: grid;
                grid-template-columns: 1.25fr 0.75fr;
                gap: 22px;
                align-items: stretch;
            }

            .form-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 14px;
            }

            .risk-box {
                background: #f8fafc;
                border: 1px solid var(--borde);
                border-radius: 15px;
                padding: 22px;
            }

            .risk-pill {
                display: inline-flex;
                padding: 8px 12px;
                border-radius: 999px;
                font-weight: 700;
                font-size: 13px;
                background: #edf2f7;
                color: var(--azul);
            }

            .risk-pill.low { background: #e6fffa; color: #047857; }
            .risk-pill.medium { background: #fff7ed; color: #c2410c; }
            .risk-pill.high { background: #fee2e2; color: #b91c1c; }

            .risk-prob {
                font-size: 56px;
                font-weight: 700;
                color: var(--azul);
                margin: 18px 0 8px;
            }

            .note {
                color: var(--gris-texto);
                font-size: 13px;
                line-height: 1.5;
            }

            .callouts {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 18px;
            }

            .callout {
                background: #f8fafc;
                border: 1px solid var(--borde);
                border-radius: 14px;
                padding: 18px;
            }

            .callout b { color: var(--azul); }

            .footer {
                color: var(--gris-texto);
                font-size: 13px;
                line-height: 1.5;
                padding: 10px 0 30px 0;
            }

            @media (max-width: 1100px) {
                .metric-grid { grid-template-columns: repeat(2, 1fr); }
                .grid-2, .prediction-layout, .callouts { grid-template-columns: 1fr; }
                .form-grid { grid-template-columns: repeat(2, 1fr); }
                .hero { padding: 32px 28px; }
                .container { padding: 24px 28px; }
            }

            @media (max-width: 720px) {
                .controls, .form-grid { grid-template-columns: 1fr; }
                .metric-grid { grid-template-columns: 1fr; }
                .chart { height: 320px; }
            }
        </style>
    </head>

    <body>
        <section class="hero">
            <div class="hero-top">
                <span class="badge">Ciencia de Datos Ambiental</span>
                <span class="badge">Flask · GitHub · Vercel</span>
            </div>
            <h1>Dashboard de predicción de eventos extremos de turbidez</h1>
            <p>
                Sistema San Antonio–Guadalupe, Texas · Modelo seleccionado: <b>Gradient Boosting</b>.
                El modelo se entrena con información histórica y se evalúa en un periodo temporal posterior,
                priorizando la detección temprana de eventos críticos de turbidez con 24 horas de anticipación.
            </p>
        </section>

        <main class="container">
            <section class="card">
                <h2>Resumen ejecutivo del modelo</h2>
                <p class="descripcion">
                    El objetivo del proyecto es clasificar si al día siguiente ocurrirá un evento extremo de turbidez.
                    La evaluación se plantea como una validación temporal para aproximar un escenario operativo real:
                    entrenar con pasado, evaluar en futuro y analizar el equilibrio entre eventos detectados y falsas alarmas.
                </p>

                <div class="controls">
                    <div>
                        <label for="station-select">Estación USGS</label>
                        <select id="station-select"><option value="all">Todas las estaciones</option></select>
                    </div>
                    <div>
                        <label for="variable-select">Variable exploratoria</label>
                        <select id="variable-select"></select>
                    </div>
                    <div>
                        <button id="update-button" type="button">Actualizar dashboard</button>
                    </div>
                </div>
                <p id="status" class="note">Cargando información...</p>

                <div class="metric-grid">
                    <div class="metric"><div id="kpi-records" class="value">—</div><div class="label">Registros evaluados</div></div>
                    <div class="metric alerta"><div id="kpi-events" class="value">—</div><div class="label">Eventos reales</div></div>
                    <div class="metric ok"><div id="kpi-detected" class="value">—</div><div class="label">Eventos detectados</div></div>
                    <div class="metric alerta"><div id="kpi-missed" class="value">—</div><div class="label">Eventos no detectados</div></div>
                    <div class="metric"><div id="kpi-alerts" class="value">—</div><div class="label">Alertas emitidas</div></div>
                    <div class="metric warning"><div id="kpi-false" class="value">—</div><div class="label">Falsas alarmas</div></div>
                    <div class="metric"><div id="kpi-precision" class="value">—</div><div class="label">Precisión</div></div>
                    <div class="metric"><div id="kpi-recall" class="value">—</div><div class="label">Recall</div></div>
                    <div class="metric"><div id="kpi-accuracy" class="value">—</div><div class="label">Accuracy</div></div>
                    <div class="metric"><div id="kpi-f1" class="value">—</div><div class="label">F1-score</div></div>
                    <div class="metric"><div id="kpi-roc" class="value">—</div><div class="label">ROC-AUC</div></div>
                    <div class="metric"><div id="kpi-pr" class="value">—</div><div class="label">PR-AUC</div></div>
                </div>
            </section>

            <section class="card">
                <h2>Mapa interactivo de estaciones USGS</h2>
                <p class="descripcion">
                    El mapa muestra la ubicación aproximada de las estaciones USGS, el eje referencial del sistema monitoreado
                    y el área aproximada de análisis. El tamaño del marcador resume la intensidad de eventos y probabilidades máximas.
                </p>
                <div id="station-map"></div>
            </section>

            <section class="card">
                <h2>Resumen por estación</h2>
                <p class="descripcion">
                    Comparación de frecuencia de eventos, alertas emitidas, probabilidad estimada y turbidez máxima observada al día siguiente.
                </p>
                <div style="overflow-x:auto;">
                    <table id="station-table" class="tabla"></table>
                </div>
            </section>

            <section class="card">
                <h2>Probabilidad diaria de evento extremo</h2>
                <p class="descripcion">
                    La línea punteada representa el umbral calibrado en validación. Valores por encima del umbral generan una alerta.
                </p>
                <div id="probability-chart" class="chart" style="height: 500px;"></div>
            </section>

            <section class="card">
                <h2>Exploración temporal y eventos</h2>
                <div class="grid-2">
                    <div class="plot-card">
                        <h3>Serie temporal de la variable seleccionada</h3>
                        <p class="note">Promedio diario por filtro activo; para turbidez se aplica escala logarítmica cuando mejora la lectura.</p>
                        <div id="series-chart" class="chart"></div>
                    </div>
                    <div class="plot-card">
                        <h3>Eventos extremos por mes</h3>
                        <p class="note">Suma mensual de eventos reales de alta turbidez.</p>
                        <div id="events-chart" class="chart"></div>
                    </div>
                    <div class="plot-card">
                        <h3>Relación caudal–turbidez</h3>
                        <p class="note">Dispersión con recorte robusto de outliers para evitar que pocos extremos oculten el patrón general.</p>
                        <div id="scatter-chart" class="chart"></div>
                    </div>
                    <div class="plot-card">
                        <h3>Distribución de turbidez</h3>
                        <p class="note">Histograma con recorte al percentil 99.5 para hacer visible la masa principal y conservar lectura de extremos.</p>
                        <div id="hist-chart" class="chart"></div>
                    </div>
                </div>
            </section>

            <section class="card">
                <h2>Evaluación rigurosa del modelo</h2>
                <p id="model-status" class="descripcion">Evaluando partición temporal...</p>
                <div class="grid-2">
                    <div>
                        <h3>Matriz de confusión</h3>
                        <div class="cm-grid">
                            <div class="cm-cell"><div id="cm-tn" class="value">—</div><div class="label">Verdaderos negativos</div></div>
                            <div class="cm-cell"><div id="cm-fp" class="value">—</div><div class="label">Falsos positivos</div></div>
                            <div class="cm-cell"><div id="cm-fn" class="value">—</div><div class="label">Falsos negativos</div></div>
                            <div class="cm-cell"><div id="cm-tp" class="value">—</div><div class="label">Verdaderos positivos</div></div>
                        </div>
                    </div>
                    <div>
                        <h3>Importancia de variables</h3>
                        <div id="importance-chart" class="chart"></div>
                    </div>
                </div>
            </section>

            <section class="card">
                <h2>Predicción interactiva de riesgo</h2>
                <p class="descripcion">
                    Ingresa condiciones actuales para estimar la probabilidad de un evento extremo de turbidez al día siguiente.
                    Los campos vacíos se completan automáticamente con medianas históricas del conjunto de entrenamiento.
                </p>
                <div class="prediction-layout">
                    <div>
                        <div class="form-grid">
                            <div><label for="p-turbidez">Turbidez actual</label><input id="p-turbidez" type="number" step="0.01" placeholder="Ej. 15.4"></div>
                            <div><label for="p-caudal">Caudal</label><input id="p-caudal" type="number" step="0.01" placeholder="Ej. 120"></div>
                            <div><label for="p-nivel">Nivel de agua</label><input id="p-nivel" type="number" step="0.01" placeholder="Ej. 4.5"></div>
                            <div><label for="p-prec">Precipitación diaria</label><input id="p-prec" type="number" step="0.01" placeholder="Ej. 8"></div>
                            <div><label for="p-prec3">Precip. acum. 3 días</label><input id="p-prec3" type="number" step="0.01" placeholder="Ej. 22"></div>
                            <div><label for="p-prec7">Precip. acum. 7 días</label><input id="p-prec7" type="number" step="0.01" placeholder="Ej. 45"></div>
                            <div><label for="p-temp">Temperatura</label><input id="p-temp" type="number" step="0.01" placeholder="Opcional"></div>
                            <div><label for="p-ph">pH</label><input id="p-ph" type="number" step="0.01" placeholder="Opcional"></div>
                            <div><label for="p-cond">Conductividad</label><input id="p-cond" type="number" step="0.01" placeholder="Opcional"></div>
                        </div>
                        <div style="margin-top:16px;"><button id="predict-button" type="button">Predecir riesgo</button></div>
                    </div>
                    <div class="risk-box">
                        <span id="risk-pill" class="risk-pill">Sin predicción</span>
                        <div id="risk-prob" class="risk-prob">—</div>
                        <p id="risk-text" class="note">Completa los campos y ejecuta la predicción.</p>
                    </div>
                </div>
            </section>

            <section class="card">
                <h2>Últimos registros del filtro activo</h2>
                <p class="descripcion">Observaciones recientes usadas para revisión rápida del dataset operativo.</p>
                <div style="overflow-x:auto;">
                    <table id="data-table" class="tabla"></table>
                </div>
            </section>

            <section class="card">
                <h2>Conclusiones analíticas</h2>
                <div class="callouts">
                    <div class="callout"><b>1. Detección anticipada</b><p class="note">El enfoque permite transformar datos diarios en una alerta anticipada para eventos críticos de turbidez.</p></div>
                    <div class="callout"><b>2. Métrica crítica</b><p class="note">En monitoreo ambiental, el recall es central porque un falso negativo implica no detectar una condición potencialmente crítica.</p></div>
                    <div class="callout"><b>3. Trazabilidad</b><p class="note">El dashboard separa exploración, validación temporal, matriz de confusión, importancia de variables y simulación de riesgo.</p></div>
                </div>
            </section>

            <p class="footer">
                Proyecto Final · Ciencia de Datos Ambiental · Dashboard desplegable en Vercel desde GitHub.
            </p>
        </main>

        <script>
            const stationSelect = document.getElementById("station-select");
            const variableSelect = document.getElementById("variable-select");
            const statusBox = document.getElementById("status");
            let optionsLoaded = false;
            let stationMap = null;
            let markerLayer = null;

            const colors = {
                azul: "#0b3954",
                teal: "#087e8b",
                rojo: "#d62828",
                naranja: "#f77f00",
                verde: "#2a9d8f",
                gris: "#4a5568",
                grid: "#e2e8f0",
            };

            const plotConfig = { responsive: true, displaylogo: false };

            function baseLayout(extra = {}) {
                return {
                    margin: { t: 26, r: 24, b: 56, l: 64 },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    font: { family: "Arial, Helvetica, sans-serif", color: "#1a202c" },
                    hovermode: "closest",
                    xaxis: { gridcolor: colors.grid, zerolinecolor: colors.grid, automargin: true },
                    yaxis: { gridcolor: colors.grid, zerolinecolor: colors.grid, automargin: true },
                    legend: { orientation: "h", y: -0.22 },
                    ...extra,
                };
            }

            function fmt(value, digits = 2) {
                if (value === null || value === undefined || Number.isNaN(value)) return "—";
                return Number(value).toLocaleString("es-PE", { maximumFractionDigits: digits });
            }

            function pct(value, digits = 2) {
                if (value === null || value === undefined || Number.isNaN(value)) return "—";
                return `${fmt(value * 100, digits)}%`;
            }

            function pctFrom100(value, digits = 2) {
                if (value === null || value === undefined || Number.isNaN(value)) return "—";
                return `${fmt(value, digits)}%`;
            }

            function setText(id, value) { document.getElementById(id).textContent = value; }

            function loadOptions(data) {
                if (optionsLoaded) return;

                data.stations.forEach(site => {
                    const opt = document.createElement("option");
                    opt.value = site;
                    opt.textContent = site;
                    stationSelect.appendChild(opt);
                });

                data.variables.forEach(variable => {
                    const opt = document.createElement("option");
                    opt.value = variable;
                    opt.textContent = variable;
                    variableSelect.appendChild(opt);
                });

                if (data.variables.includes("Turbidez")) variableSelect.value = "Turbidez";
                optionsLoaded = true;
            }

            function updateKpis(data) {
                const k = data.kpis;
                setText("kpi-records", fmt(k.records, 0));
                setText("kpi-events", fmt(k.real_events, 0));
                setText("kpi-detected", fmt(k.detected_events, 0));
                setText("kpi-missed", fmt(k.missed_events, 0));
                setText("kpi-alerts", fmt(k.predicted_alerts, 0));
                setText("kpi-false", fmt(k.false_alarms, 0));

                const m = data.model.metrics || {};
                setText("kpi-precision", pct(m.precision, 2));
                setText("kpi-recall", pct(m.recall, 2));
                setText("kpi-accuracy", pct(m.accuracy, 2));
                setText("kpi-f1", pct(m.f1, 2));
                setText("kpi-roc", fmt(m.roc_auc, 3));
                setText("kpi-pr", fmt(m.pr_auc, 3));
            }

            function renderStationTable(rows) {
                const table = document.getElementById("station-table");
                const columns = [
                    ["site_id", "Estación USGS"],
                    ["records", "Registros evaluados"],
                    ["real_events", "Eventos reales"],
                    ["detected_events", "Eventos detectados"],
                    ["missed_events", "No detectados"],
                    ["false_alarms", "Falsas alarmas"],
                    ["mean_probability", "Prob. media (%)"],
                    ["max_probability", "Prob. máxima (%)"],
                    ["max_turbidity_t1", "Turbidez máxima t+1"],
                ];
                table.innerHTML = `
                    <thead><tr>${columns.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>
                    <tbody>
                        ${rows.map(row => `<tr>${columns.map(c => `<td>${row[c[0]] ?? ""}</td>`).join("")}</tr>`).join("")}
                    </tbody>
                `;
            }

            function renderDataTable(data) {
                const table = document.getElementById("data-table");
                const cols = data.table_columns;
                const rows = data.table_rows;
                if (!rows.length) {
                    table.innerHTML = "<tbody><tr><td>No hay registros para mostrar.</td></tr></tbody>";
                    return;
                }
                table.innerHTML = `
                    <thead><tr>${cols.map(col => `<th>${col}</th>`).join("")}</tr></thead>
                    <tbody>${rows.map(row => `<tr>${cols.map(col => `<td>${row[col] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>
                `;
            }

            function initMap() {
                if (stationMap) return;
                stationMap = L.map("station-map", { scrollWheelZoom: false });
                L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
                    maxZoom: 20,
                    attribution: "&copy; OpenStreetMap contributors &copy; CARTO"
                }).addTo(stationMap);

                L.polygon([
                    [29.364128, -98.731407], [29.364128, -97.264858],
                    [28.529286, -97.264858], [28.529286, -98.731407]
                ], { color: "#1f78b4", fillColor: "#1f78b4", fillOpacity: 0.08, weight: 2 }).addTo(stationMap)
                  .bindTooltip("Área aproximada de estudio");

                L.polyline([
                    [29.262183, -98.581407], [29.264128, -98.490849],
                    [28.884, -97.737139], [28.649286, -97.384858]
                ], { color: colors.verde, weight: 4, opacity: 0.85 }).addTo(stationMap)
                  .bindTooltip("Eje principal referencial del sistema monitoreado");

                L.polyline([
                    [29.222185, -98.355846], [29.264128, -98.490849]
                ], { color: "#457b9d", weight: 3, opacity: 0.85, dashArray: "8, 6" }).addTo(stationMap)
                  .bindTooltip("Afluente monitoreado referencial");

                const legend = L.control({ position: "bottomleft" });
                legend.onAdd = function() {
                    const div = L.DomUtil.create("div", "legend");
                    div.innerHTML = `<b>Mapa del área de estudio</b><br>
                        <span style="color:#d62828;">●</span> Estación con alertas predichas<br>
                        <span style="color:#2b6cb0;">●</span> Estación sin alertas predichas<br>
                        <span style="color:#2a9d8f;">━━</span> Eje principal referencial<br>
                        <span style="color:#457b9d;">- - -</span> Afluente referencial<br>
                        <span style="color:#1f78b4;">▧</span> Área aproximada de análisis`;
                    return div;
                };
                legend.addTo(stationMap);
                markerLayer = L.layerGroup().addTo(stationMap);
            }

            function updateMap(stations) {
                initMap();
                markerLayer.clearLayers();
                const bounds = [];
                stations.forEach(s => {
                    bounds.push([s.lat, s.lon]);
                    const popup = `<div style="font-family:Arial; font-size:13px; min-width:220px;">
                        <h4 style="margin:0 0 8px;">${s.name}</h4>
                        <b>Registros evaluados:</b> ${s.records ?? 0}<br>
                        <b>Eventos reales:</b> ${s.real_events ?? 0}<br>
                        <b>Eventos detectados:</b> ${s.detected_events ?? 0}<br>
                        <b>Alertas predichas:</b> ${s.predicted_alerts ?? 0}<br>
                        <b>Probabilidad media:</b> ${s.mean_probability ?? "—"}%<br>
                        <b>Probabilidad máxima:</b> ${s.max_probability ?? "—"}%<br>
                        <b>Turbidez máxima t+1:</b> ${s.max_turbidity_t1 ?? "—"}
                    </div>`;
                    L.circleMarker([s.lat, s.lon], {
                        radius: s.radius || 9,
                        color: s.color || "#2b6cb0",
                        fillColor: s.color || "#2b6cb0",
                        fillOpacity: 0.88,
                        weight: 2
                    }).bindPopup(popup).bindTooltip(`Estación USGS ${s.site_id}`).addTo(markerLayer);
                });
                if (bounds.length) stationMap.fitBounds(bounds, { padding: [34, 34] });
                setTimeout(() => stationMap.invalidateSize(), 200);
            }

            function drawProbability(data) {
                const tracesBySite = {};
                data.probability_series.forEach(row => {
                    if (!tracesBySite[row.site_id]) tracesBySite[row.site_id] = { x: [], y: [], customdata: [], name: row.site_id };
                    tracesBySite[row.site_id].x.push(row.date);
                    tracesBySite[row.site_id].y.push(row.probability);
                    tracesBySite[row.site_id].customdata.push([row.real_event, row.predicted_alert]);
                });
                const traces = Object.values(tracesBySite).map(obj => ({
                    x: obj.x,
                    y: obj.y,
                    customdata: obj.customdata,
                    name: obj.name,
                    type: "scatter",
                    mode: "lines",
                    line: { width: 2 },
                    hovertemplate: "Estación %{fullData.name}<br>Fecha: %{x}<br>Probabilidad: %{y:.1%}<br>Real: %{customdata[0]}<br>Alerta: %{customdata[1]}<extra></extra>",
                }));
                traces.push({
                    x: data.threshold_line.map(d => d.date),
                    y: data.threshold_line.map(d => d.threshold),
                    name: "Umbral calibrado",
                    type: "scatter",
                    mode: "lines",
                    line: { color: colors.rojo, width: 2, dash: "dash" },
                    hovertemplate: "Umbral: %{y:.1%}<extra></extra>",
                });
                Plotly.newPlot("probability-chart", traces, baseLayout({
                    yaxis: { title: "Probabilidad estimada", tickformat: ".0%", gridcolor: colors.grid, rangemode: "tozero" },
                    xaxis: { title: "Fecha evaluada", gridcolor: colors.grid },
                }), plotConfig);
            }

            function drawSeries(data) {
                const variable = data.selected.variable;
                const yValues = data.series.map(d => d.value).filter(v => v !== null && v !== undefined);
                const useLog = variable.toLowerCase().includes("turbidez") && Math.max(...yValues, 1) / Math.max(Math.min(...yValues.filter(v => v > 0), 1), 0.1) > 50;
                const trace = {
                    x: data.series.map(d => d.date),
                    y: data.series.map(d => d.value),
                    type: "scatter",
                    mode: "lines",
                    line: { color: colors.azul, width: 2.5 },
                    hovertemplate: `%{x}<br>${variable}: %{y:.2f}<extra></extra>`,
                    name: variable,
                };
                Plotly.newPlot("series-chart", [trace], baseLayout({
                    yaxis: { title: variable, type: useLog ? "log" : "linear", gridcolor: colors.grid },
                    xaxis: { title: "Fecha", gridcolor: colors.grid },
                    showlegend: false,
                }), plotConfig);
            }

            function drawEvents(data) {
                const trace = {
                    x: data.monthly_events.map(d => d.month),
                    y: data.monthly_events.map(d => d.events),
                    type: "bar",
                    marker: { color: colors.naranja },
                    hovertemplate: "%{x}<br>Eventos: %{y}<extra></extra>",
                };
                Plotly.newPlot("events-chart", [trace], baseLayout({
                    yaxis: { title: "Eventos reales", gridcolor: colors.grid },
                    xaxis: { title: "Mes", gridcolor: colors.grid },
                    showlegend: false,
                }), plotConfig);
            }

            function drawScatter(data) {
                const trace = {
                    x: data.scatter.map(d => d.flow),
                    y: data.scatter.map(d => d.turbidity),
                    text: data.scatter.map(d => `Fecha: ${d.date}<br>Estación: ${d.site_id}`),
                    type: "scatter",
                    mode: "markers",
                    marker: { size: 7, color: colors.teal, opacity: 0.62, line: { color: "white", width: 0.5 } },
                    hovertemplate: "%{text}<br>Caudal: %{x:.2f}<br>Turbidez: %{y:.2f}<extra></extra>",
                };
                Plotly.newPlot("scatter-chart", [trace], baseLayout({
                    xaxis: { title: "Caudal", gridcolor: colors.grid },
                    yaxis: { title: "Turbidez", gridcolor: colors.grid },
                    showlegend: false,
                }), plotConfig);
            }

            function drawHistogram(data) {
                const trace = {
                    x: data.histogram.map(d => d.value),
                    type: "histogram",
                    nbinsx: 38,
                    marker: { color: colors.azul },
                    hovertemplate: "Turbidez: %{x}<br>Frecuencia: %{y}<extra></extra>",
                };
                Plotly.newPlot("hist-chart", [trace], baseLayout({
                    xaxis: { title: "Turbidez", gridcolor: colors.grid },
                    yaxis: { title: "Frecuencia", gridcolor: colors.grid },
                    bargap: 0.04,
                    showlegend: false,
                }), plotConfig);
            }

            function drawImportance(data) {
                const imp = [...(data.model.feature_importance || [])].reverse();
                const trace = {
                    x: imp.map(d => d.importance),
                    y: imp.map(d => d.feature),
                    type: "bar",
                    orientation: "h",
                    marker: { color: colors.teal },
                    hovertemplate: "%{y}<br>Importancia: %{x:.3f}<extra></extra>",
                };
                Plotly.newPlot("importance-chart", [trace], baseLayout({
                    xaxis: { title: "Importancia relativa", gridcolor: colors.grid },
                    yaxis: { automargin: true, gridcolor: colors.grid },
                    showlegend: false,
                }), plotConfig);
            }

            function updateModelPanel(data) {
                if (!data.model.available) {
                    document.getElementById("model-status").textContent = data.model.reason || "Modelo no disponible.";
                    return;
                }
                const m = data.model.metrics;
                document.getElementById("model-status").textContent =
                    `Modelo ${data.model.name}. Entrenamiento: ${m.train_period}. Evaluación: ${m.test_period}. ` +
                    `Umbral calibrado: ${fmt(data.model.threshold, 3)}. Tasa de evento en prueba: ${fmt(m.event_rate, 2)}%.`;

                setText("cm-tn", fmt(data.model.confusion_matrix.tn, 0));
                setText("cm-fp", fmt(data.model.confusion_matrix.fp, 0));
                setText("cm-fn", fmt(data.model.confusion_matrix.fn, 0));
                setText("cm-tp", fmt(data.model.confusion_matrix.tp, 0));
                drawImportance(data);
            }

            async function loadDashboard() {
                try {
                    statusBox.textContent = "Actualizando información...";
                    const params = new URLSearchParams({
                        site_id: stationSelect.value,
                        variable: variableSelect.value || "Turbidez"
                    });
                    const response = await fetch(`/api/dashboard?${params.toString()}`);
                    const data = await response.json();

                    loadOptions(data);
                    updateKpis(data);
                    renderStationTable(data.station_summary);
                    renderDataTable(data);
                    updateMap(data.map_stations);
                    drawProbability(data);
                    drawSeries(data);
                    drawEvents(data);
                    drawScatter(data);
                    drawHistogram(data);
                    updateModelPanel(data);

                    const site = data.selected.site_id === "all" ? "Todas las estaciones" : data.selected.site_id;
                    statusBox.textContent = `Filtro activo: ${site} · Variable: ${data.selected.variable}`;
                } catch (error) {
                    console.error(error);
                    statusBox.textContent = "No se pudo cargar el dashboard. Revisa la consola o el endpoint /api/dashboard.";
                }
            }

            function readInput(id) {
                const value = document.getElementById(id).value;
                return value === "" ? null : Number(value);
            }

            async function predictRisk() {
                const payload = {
                    Turbidez: readInput("p-turbidez"),
                    Caudal: readInput("p-caudal"),
                    Nivel_Agua: readInput("p-nivel"),
                    Precipitacion: readInput("p-prec"),
                    Precipitacion_acum_3d: readInput("p-prec3"),
                    Precipitacion_acum_7d: readInput("p-prec7"),
                    Temperatura: readInput("p-temp"),
                    pH: readInput("p-ph"),
                    Conductividad: readInput("p-cond"),
                };
                const response = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const result = await response.json();
                const pill = document.getElementById("risk-pill");
                const prob = document.getElementById("risk-prob");
                const text = document.getElementById("risk-text");
                pill.className = "risk-pill";
                if (!result.available) {
                    pill.textContent = "Modelo no disponible";
                    prob.textContent = "—";
                    text.textContent = result.reason || "No se pudo generar la predicción.";
                    return;
                }
                pill.textContent = `Riesgo ${result.risk_level}`;
                if (result.risk_level === "Alto") pill.classList.add("high");
                else if (result.risk_level === "Medio") pill.classList.add("medium");
                else pill.classList.add("low");
                prob.textContent = pct(result.probability, 1);
                text.textContent = result.message;
            }

            document.getElementById("update-button").addEventListener("click", loadDashboard);
            document.getElementById("predict-button").addEventListener("click", predictRisk);
            stationSelect.addEventListener("change", loadDashboard);
            variableSelect.addEventListener("change", loadDashboard);
            loadDashboard();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)


@app.route("/api/dashboard")
def api_dashboard():
    df_all = load_data().copy()
    artifacts = model_artifacts()

    selected_site = request.args.get("site_id", "all")
    selected_variable = request.args.get("variable", "Turbidez")

    variables = [col for col in PREFERRED_FEATURES if col in df_all.columns and pd.api.types.is_numeric_dtype(df_all[col])]
    if not variables:
        variables = [col for col in df_all.select_dtypes(include=[np.number]).columns.tolist() if col != TARGET_COLUMN]
    if selected_variable not in df_all.columns:
        selected_variable = variables[0] if variables else df_all.columns[0]

    df = filtered_df(df_all, selected_site)
    stations = sorted(df_all[SITE_COLUMN].dropna().astype(str).unique().tolist()) if SITE_COLUMN in df_all.columns else []

    if artifacts.get("available"):
        pred_all = artifacts["test_predictions"].copy()
        pred_filtered = filtered_df(pred_all, selected_site)
        station_summary = station_summary_from_predictions(pred_all)
        kpi_base = pred_filtered
        metrics = artifacts["metrics"]
        cm = artifacts["confusion_matrix"]
        model_payload = {
            "available": True,
            "name": MODEL_NAME,
            "threshold": artifacts["threshold"],
            "metrics": metrics,
            "confusion_matrix": cm,
            "feature_importance": artifacts["feature_importance"],
        }
    else:
        pred_all = pd.DataFrame()
        pred_filtered = pd.DataFrame()
        station_summary = []
        kpi_base = df
        model_payload = {"available": False, "reason": artifacts.get("reason", "Modelo no disponible."), "metrics": {}, "confusion_matrix": {}, "feature_importance": []}

    if not pred_filtered.empty and {"y_true", "y_pred"}.issubset(pred_filtered.columns):
        real_events = int(pred_filtered["y_true"].sum())
        predicted_alerts = int(pred_filtered["y_pred"].sum())
        detected_events = int(((pred_filtered["y_true"] == 1) & (pred_filtered["y_pred"] == 1)).sum())
        missed_events = int(((pred_filtered["y_true"] == 1) & (pred_filtered["y_pred"] == 0)).sum())
        false_alarms = int(((pred_filtered["y_true"] == 0) & (pred_filtered["y_pred"] == 1)).sum())
        records = int(len(pred_filtered))
    else:
        records = int(len(df))
        real_events = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df.columns else 0
        predicted_alerts = detected_events = missed_events = false_alarms = 0

    kpis = {
        "records": records,
        "real_events": real_events,
        "predicted_alerts": predicted_alerts,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "false_alarms": false_alarms,
    }

    # Serie temporal suavizada diaria/semanal según cantidad de puntos.
    series = []
    if DATE_COLUMN in df.columns and selected_variable in df.columns and not df.empty:
        series_df = df.dropna(subset=[DATE_COLUMN, selected_variable]).copy()
        series_df = series_df.sort_values(DATE_COLUMN)
        if len(series_df) > 1800:
            series_df["period"] = series_df[DATE_COLUMN].dt.to_period("W").dt.start_time
            grouped = series_df.groupby("period", as_index=False)[selected_variable].mean().rename(columns={"period": DATE_COLUMN})
        else:
            grouped = series_df.groupby(DATE_COLUMN, as_index=False)[selected_variable].mean()
        series = [{"date": safe_date(r[DATE_COLUMN]), "value": safe_float(r[selected_variable], 3)} for _, r in grouped.iterrows()]

    monthly_events = []
    if DATE_COLUMN in df.columns and not df.empty:
        mdf = df.dropna(subset=[DATE_COLUMN]).copy()
        mdf["month"] = mdf[DATE_COLUMN].dt.to_period("M").astype(str)
        if TARGET_COLUMN in mdf.columns:
            grouped = mdf.groupby("month", as_index=False)[TARGET_COLUMN].sum().rename(columns={TARGET_COLUMN: "events"})
        else:
            grouped = mdf.groupby("month", as_index=False).size().rename(columns={"size": "events"})
        monthly_events = [{"month": r["month"], "events": int(r["events"])} for _, r in grouped.iterrows()]

    probability_series = []
    threshold_line = []
    if artifacts.get("available") and not pred_filtered.empty and DATE_COLUMN in pred_filtered.columns:
        p = pred_filtered.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN).copy()
        if len(p) > 2500:
            p = p.sample(2500, random_state=42).sort_values(DATE_COLUMN)
        probability_series = [{
            "date": safe_date(r[DATE_COLUMN]),
            "site_id": str(r[SITE_COLUMN]) if SITE_COLUMN in p.columns else "all",
            "probability": safe_float(r["probability"], 5),
            "real_event": int(r["y_true"]),
            "predicted_alert": int(r["y_pred"]),
        } for _, r in p.iterrows()]
        if len(p) > 0:
            threshold_line = [
                {"date": safe_date(p[DATE_COLUMN].min()), "threshold": artifacts["threshold"]},
                {"date": safe_date(p[DATE_COLUMN].max()), "threshold": artifacts["threshold"]},
            ]

    scatter = []
    if {"Caudal", "Turbidez"}.issubset(df.columns) and not df.empty:
        sdf = df.dropna(subset=["Caudal", "Turbidez"]).copy()
        if len(sdf) > 0:
            q = sdf[["Caudal", "Turbidez"]].quantile([0.01, 0.995])
            sdf = sdf[
                (sdf["Caudal"] >= q.loc[0.01, "Caudal"]) &
                (sdf["Caudal"] <= q.loc[0.995, "Caudal"]) &
                (sdf["Turbidez"] >= q.loc[0.01, "Turbidez"]) &
                (sdf["Turbidez"] <= q.loc[0.995, "Turbidez"])
            ]
        if len(sdf) > 1200:
            sdf = sdf.sample(1200, random_state=42)
        scatter = [{
            "flow": safe_float(r["Caudal"], 3),
            "turbidity": safe_float(r["Turbidez"], 3),
            "date": safe_date(r[DATE_COLUMN]) if DATE_COLUMN in sdf.columns else "",
            "site_id": str(r[SITE_COLUMN]) if SITE_COLUMN in sdf.columns else "",
        } for _, r in sdf.iterrows()]

    histogram = []
    if "Turbidez" in df.columns and not df.empty:
        hdf = df.dropna(subset=["Turbidez"]).copy()
        if len(hdf) > 0:
            cap = hdf["Turbidez"].quantile(0.995)
            hdf = hdf[hdf["Turbidez"] <= cap]
        if len(hdf) > 2000:
            hdf = hdf.sample(2000, random_state=42)
        histogram = [{"value": safe_float(v, 3)} for v in hdf["Turbidez"].tolist()]

    table_columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
    table_rows = []
    if table_columns and not df.empty:
        tdf = df.copy()
        if DATE_COLUMN in tdf.columns:
            tdf = tdf.sort_values(DATE_COLUMN, ascending=False)
            tdf[DATE_COLUMN] = tdf[DATE_COLUMN].dt.strftime("%Y-%m-%d")
        table_rows = clean_records(tdf[table_columns].head(14).to_dict(orient="records"))

    return jsonify({
        "course": COURSE_NAME,
        "selected": {"site_id": selected_site, "variable": selected_variable},
        "stations": stations,
        "variables": variables,
        "kpis": kpis,
        "model": model_payload,
        "station_summary": station_summary,
        "map_stations": station_map_payload(station_summary),
        "series": series,
        "monthly_events": monthly_events,
        "probability_series": probability_series,
        "threshold_line": threshold_line,
        "scatter": scatter,
        "histogram": histogram,
        "table_columns": table_columns,
        "table_rows": table_rows,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    artifacts = model_artifacts()
    if not artifacts.get("available"):
        return jsonify({"available": False, "reason": artifacts.get("reason", "Modelo no disponible.")})

    data = request.get_json(silent=True) or {}
    features = artifacts["features"]
    medians = artifacts["medians"]
    row = {}
    for feature in features:
        value = data.get(feature, None)
        if value is None or value == "":
            row[feature] = medians.get(feature, 0)
        else:
            try:
                row[feature] = float(value)
            except Exception:
                row[feature] = medians.get(feature, 0)

    X_new = pd.DataFrame([row], columns=features)
    probability = float(artifacts["model"].predict_proba(X_new)[0][1])
    threshold = artifacts.get("threshold", 0.5)

    if probability >= max(0.70, threshold):
        risk_level = "Alto"
        message = "La probabilidad supera un nivel de alerta alto. Se recomienda priorizar monitoreo y revisión operativa."
    elif probability >= threshold:
        risk_level = "Medio"
        message = "La probabilidad supera el umbral calibrado del modelo. Conviene mantener seguimiento cercano."
    else:
        risk_level = "Bajo"
        message = "Las condiciones ingresadas no superan el umbral calibrado para alerta temprana."

    return jsonify({
        "available": True,
        "probability": safe_float(probability, 5),
        "threshold": safe_float(threshold, 3),
        "risk_level": risk_level,
        "message": message,
        "used_features": row,
    })


@app.route("/api/resumen")
def api_resumen():
    df = load_data()
    artifacts = model_artifacts()
    return jsonify({
        "curso": COURSE_NAME,
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nombres_columnas": df.columns.tolist(),
        "objetivo": TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
        "modelo_disponible": bool(artifacts.get("available")),
    })


if __name__ == "__main__":
    app.run(debug=True)
