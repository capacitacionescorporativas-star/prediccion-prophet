import os
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_prophet.json"


def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    with MODEL_PATH.open("r", encoding="utf-8") as archivo:
        return model_from_json(archivo.read())


modelo = cargar_modelo()

# El JSON de Prophet conserva el histórico usado durante el entrenamiento.
historico = modelo.history[["ds", "y"]].copy()
historico["ds"] = pd.to_datetime(historico["ds"])
historico["y"] = pd.to_numeric(historico["y"], errors="coerce")
historico = historico.dropna(subset=["ds", "y"]).sort_values("ds")

ultimo_anio_historico = int(historico["ds"].dt.year.max())
# Horizonte seleccionable: 25 años posteriores al último dato observado.
anios_disponibles = [str(anio) for anio in range(ultimo_anio_historico + 1, ultimo_anio_historico + 26)]


def calcular_metricas():
    pronostico_hist = modelo.predict(historico[["ds"]])
    evaluacion = historico[["ds", "y"]].merge(
        pronostico_hist[["ds", "yhat"]], on="ds", how="inner"
    )

    real = evaluacion["y"].to_numpy(dtype=float)
    pred = evaluacion["yhat"].to_numpy(dtype=float)
    error = real - pred

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    mascara = real != 0
    mape = float(np.mean(np.abs(error[mascara] / real[mascara])) * 100) if mascara.any() else np.nan

    return pd.DataFrame(
        {
            "Métrica": ["MAE", "RMSE", "MAPE"],
            "Valor": [round(mae, 2), round(rmse, 2), round(mape, 4)],
            "Unidad": ["personas", "personas", "%"],
        }
    )


tabla_metricas = calcular_metricas()


def estimar_poblacion(anios_seleccionados):
    if not anios_seleccionados:
        raise gr.Error("Seleccione al menos un año para realizar la predicción.")

    anios = sorted({int(anio) for anio in anios_seleccionados})
    fechas = pd.DataFrame({"ds": pd.to_datetime([f"{anio}-01-01" for anio in anios])})
    pred = modelo.predict(fechas)

    tabla = pd.DataFrame(
        {
            "Año": pred["ds"].dt.year.astype(int),
            "Población estimada": np.rint(pred["yhat"]).astype("int64"),
            "Límite inferior": np.rint(pred["yhat_lower"]).astype("int64"),
            "Límite superior": np.rint(pred["yhat_upper"]).astype("int64"),
        }
    )

    # Para la gráfica se genera una trayectoria anual completa desde el histórico
    # hasta el mayor año solicitado.
    anio_max = max(anios)
    fechas_grafico = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [f"{anio}-01-01" for anio in range(ultimo_anio_historico + 1, anio_max + 1)]
            )
        }
    )
    pred_grafico = modelo.predict(fechas_grafico)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(historico["ds"], historico["y"], marker="o", markersize=3, linewidth=1.5, label="Histórico")
    ax.plot(pred_grafico["ds"], pred_grafico["yhat"], marker="o", markersize=3, linewidth=1.8, label="Predicción")
    ax.fill_between(
        pred_grafico["ds"],
        pred_grafico["yhat_lower"],
        pred_grafico["yhat_upper"],
        alpha=0.18,
        label="Intervalo de incertidumbre",
    )

    seleccion = pred_grafico[pred_grafico["ds"].dt.year.isin(anios)]
    ax.scatter(seleccion["ds"], seleccion["yhat"], s=60, zorder=5, label="Años seleccionados")

    ax.set_title("Población histórica y predicción de Nicaragua")
    ax.set_xlabel("Año")
    ax.set_ylabel("Población")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return tabla, fig


def exportar_excel(tabla):
    if tabla is None:
        raise gr.Error("Primero genere una predicción.")

    if isinstance(tabla, pd.DataFrame):
        df = tabla.copy()
    else:
        df = pd.DataFrame(tabla)

    if df.empty:
        raise gr.Error("Primero genere una predicción.")

    archivo = tempfile.NamedTemporaryFile(
        prefix="prediccion_poblacion_nicaragua_",
        suffix=".xlsx",
        delete=False,
    )
    ruta = archivo.name
    archivo.close()

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Predicción")
        tabla_metricas.to_excel(writer, index=False, sheet_name="Métricas")

    return ruta


with gr.Blocks(title="Predicción de Población de Nicaragua") as demo:
    gr.Markdown("# 🇳🇮 Predicción de Población de Nicaragua")
    gr.Markdown("### Prophet · Datos del Banco Mundial")

    gr.Dataframe(
        value=tabla_metricas,
        label="Errores promedio del modelo",
        interactive=False,
    )

    anios_input = gr.CheckboxGroup(
        choices=anios_disponibles,
        label="Seleccione el año o los años",
    )

    btn_estimar = gr.Button("Estimar", variant="primary")

    tabla_output = gr.Dataframe(
        label="Predicción",
        interactive=False,
    )

    grafico_output = gr.Plot(label="Histórico y predicción")

    btn_estimar.click(
        fn=estimar_poblacion,
        inputs=anios_input,
        outputs=[tabla_output, grafico_output],
    )

    gr.Markdown("### 📥 Descargar predicción")

    btn_descargar = gr.Button("Descargar Excel")
    archivo_output = gr.File(label="Archivo Excel")

    btn_descargar.click(
        fn=exportar_excel,
        inputs=tabla_output,
        outputs=archivo_output,
    )


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", "7860"))
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=puerto,
        show_error=True,
    )
