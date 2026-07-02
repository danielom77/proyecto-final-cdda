from flask import Flask, jsonify, render_template_string
import pandas as pd
from pathlib import Path

app = Flask(__name__)

DATA_PATH = Path(__file__).parent / "dataset_modelo_2018_2025.csv"


def cargar_datos():
    df = pd.read_csv(DATA_PATH)

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

    return df


@app.route("/")
def home():
    df = cargar_datos()

    filas = df.shape[0]
    columnas = df.shape[1]

    estaciones = df["site_id"].nunique() if "site_id" in df.columns else "N/D"

    fecha_min = df["Fecha"].min().strftime("%d/%m/%Y") if "Fecha" in df.columns else "N/D"
    fecha_max = df["Fecha"].max().strftime("%d/%m/%Y") if "Fecha" in df.columns else "N/D"

    turbidez_promedio = round(df["Turbidez"].mean(), 2) if "Turbidez" in df.columns else "N/D"
    turbidez_max = round(df["Turbidez"].max(), 2) if "Turbidez" in df.columns else "N/D"

    if "high_turbidity_t1" in df.columns:
        eventos_altos = int(df["high_turbidity_t1"].sum())
    else:
        eventos_altos = "N/D"

    columnas_lista = df.columns.tolist()

    vista_previa = df.head(12).to_html(
        index=False,
        classes="tabla-datos",
        border=0
    )

    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Proyecto Final CdDA</title>
        <style>
            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: #f3f6fb;
                color: #1f2937;
            }

            header {
                background: linear-gradient(135deg, #0f172a, #1e3a8a);
                color: white;
                padding: 42px 60px;
            }

            header h1 {
                margin: 0;
                font-size: 42px;
                letter-spacing: -1px;
            }

            header p {
                margin-top: 12px;
                font-size: 18px;
                color: #dbeafe;
            }

            main {
                padding: 36px 60px;
            }

            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                gap: 18px;
                margin-bottom: 32px;
            }

            .card {
                background: white;
                border-radius: 16px;
                padding: 22px;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
                border: 1px solid #e5e7eb;
            }

            .card span {
                display: block;
                color: #64748b;
                font-size: 14px;
                margin-bottom: 8px;
            }

            .card strong {
                font-size: 30px;
                color: #0f172a;
            }

            .section {
                background: white;
                border-radius: 16px;
                padding: 26px;
                margin-bottom: 28px;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
                border: 1px solid #e5e7eb;
            }

            .section h2 {
                margin-top: 0;
                color: #0f172a;
            }

            .columnas {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }

            .chip {
                background: #e0f2fe;
                color: #075985;
                padding: 8px 12px;
                border-radius: 999px;
                font-size: 14px;
                font-weight: 600;
            }

            .tabla-contenedor {
                overflow-x: auto;
            }

            .tabla-datos {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }

            .tabla-datos th {
                background: #0f172a;
                color: white;
                padding: 12px;
                text-align: left;
                white-space: nowrap;
            }

            .tabla-datos td {
                padding: 10px 12px;
                border-bottom: 1px solid #e5e7eb;
                white-space: nowrap;
            }

            .tabla-datos tr:hover {
                background: #f8fafc;
            }

            .footer {
                color: #64748b;
                font-size: 14px;
                margin-top: 24px;
            }

            a {
                color: #2563eb;
                font-weight: bold;
                text-decoration: none;
            }

            @media (max-width: 700px) {
                header, main {
                    padding: 28px 22px;
                }

                header h1 {
                    font-size: 32px;
                }
            }
        </style>
    </head>

    <body>
        <header>
            <h1>Proyecto Final CdDA</h1>
            <p>Dashboard de análisis y predicción de turbidez del agua</p>
        </header>

        <main>
            <div class="cards">
                <div class="card">
                    <span>Filas del dataset</span>
                    <strong>{{ filas }}</strong>
                </div>

                <div class="card">
                    <span>Columnas</span>
                    <strong>{{ columnas }}</strong>
                </div>

                <div class="card">
                    <span>Estaciones</span>
                    <strong>{{ estaciones }}</strong>
                </div>

                <div class="card">
                    <span>Periodo</span>
                    <strong style="font-size: 20px;">{{ fecha_min }} - {{ fecha_max }}</strong>
                </div>

                <div class="card">
                    <span>Turbidez promedio</span>
                    <strong>{{ turbidez_promedio }}</strong>
                </div>

                <div class="card">
                    <span>Turbidez máxima</span>
                    <strong>{{ turbidez_max }}</strong>
                </div>

                <div class="card">
                    <span>Eventos de alta turbidez</span>
                    <strong>{{ eventos_altos }}</strong>
                </div>
            </div>

            <div class="section">
                <h2>Variables del modelo</h2>
                <div class="columnas">
                    {% for col in columnas_lista %}
                        <span class="chip">{{ col }}</span>
                    {% endfor %}
                </div>
            </div>

            <div class="section">
                <h2>Vista previa del dataset</h2>
                <div class="tabla-contenedor">
                    {{ vista_previa | safe }}
                </div>
            </div>

            <div class="section">
                <h2>API del proyecto</h2>
                <p>
                    Puedes consultar un resumen en formato JSON aquí:
                    <a href="/api/resumen">/api/resumen</a>
                </p>
            </div>

            <p class="footer">
                Proyecto desplegable en Vercel desde GitHub.
            </p>
        </main>
    </body>
    </html>
    """

    return render_template_string(
        html,
        filas=filas,
        columnas=columnas,
        estaciones=estaciones,
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        turbidez_promedio=turbidez_promedio,
        turbidez_max=turbidez_max,
        eventos_altos=eventos_altos,
        columnas_lista=columnas_lista,
        vista_previa=vista_previa
    )


@app.route("/api/resumen")
def resumen():
    df = cargar_datos()

    return jsonify({
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "nombres_columnas": df.columns.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)