"""Prediccion de poblacion de Nicaragua con Prophet y Streamlit.

El archivo modelo_prophet.json debe estar en la misma carpeta que app.py.
El modelo se carga tal como fue entrenado; no se vuelve a entrenar.
"""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from prophet.serialize import model_from_json


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_prophet.json"

st.set_page_config(
    page_title="Prediccion de Poblacion de Nicaragua",
    page_icon="🇳🇮",
    layout="wide",
)


@st.cache_resource(show_spinner="Cargando el modelo Prophet...")
def cargar_modelo():
    """Carga el modelo entrenado una sola vez por instancia de Streamlit."""
    with MODEL_PATH.open("r", encoding="utf-8") as archivo:
        return model_from_json(archivo.read())


def cargar_historico(modelo):
    """Recupera el historico almacenado en el modelo, sin agregar datos."""
    historico = modelo.history[["ds", "y"]].copy()
    historico["ds"] = pd.to_datetime(historico["ds"])
    historico["y"] = pd.to_numeric(historico["y"], errors="coerce")
    return historico.dropna(subset=["ds", "y"]).sort_values("ds")


@st.cache_data(show_spinner=False)
def calcular_metricas():
    """MAE, RMSE y MAPE calculados sobre el historico de entrenamiento."""
    modelo = cargar_modelo()
    historico = cargar_historico(modelo)
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
    mape = (
        float(np.mean(np.abs(error[mascara] / real[mascara])) * 100)
        if mascara.any()
        else np.nan
    )

    return pd.DataFrame(
        {
            "Metrica": ["MAE", "RMSE", "MAPE"],
            "Valor": [round(mae, 2), round(rmse, 2), round(mape, 4)],
            "Unidad": ["personas", "personas", "%"],
        }
    )


def estimar_poblacion(modelo, anios_seleccionados, ultimo_anio_historico):
    """Predice los anios elegidos y la trayectoria anual hasta el ultimo de ellos."""
    anios = sorted({int(anio) for anio in anios_seleccionados})
    if not anios:
        raise ValueError("Seleccione al menos un anio para realizar la prediccion.")

    fechas = pd.DataFrame({
        "ds": pd.to_datetime([f"{anio}-01-01" for anio in anios])
    })
    pred = modelo.predict(fechas)

    tabla = pd.DataFrame(
        {
            "Año": pred["ds"].dt.year.astype(int),
            "Población estimada": np.rint(pred["yhat"]).astype("int64"),
            "Límite inferior": np.rint(pred["yhat_lower"]).astype("int64"),
            "Límite superior": np.rint(pred["yhat_upper"]).astype("int64"),
        }
    )

    fechas_grafico = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    f"{anio}-01-01"
                    for anio in range(ultimo_anio_historico + 1, max(anios) + 1)
                ]
            )
        }
    )
    pronostico_grafico = modelo.predict(fechas_grafico)
    return tabla, pronostico_grafico


def construir_grafico(historico, pronostico_grafico, anios_seleccionados):
    """Muestra el historico, pronostico, bandas y anios seleccionados."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        historico["ds"], historico["y"],
        marker="o", markersize=3, linewidth=1.5, label="Histórico",
    )
    ax.plot(
        pronostico_grafico["ds"], pronostico_grafico["yhat"],
        marker="o", markersize=3, linewidth=1.8, label="Predicción",
    )
    ax.fill_between(
        pronostico_grafico["ds"].to_numpy(),
        pronostico_grafico["yhat_lower"].to_numpy(dtype=float),
        pronostico_grafico["yhat_upper"].to_numpy(dtype=float),
        alpha=0.18,
        label="Intervalo de incertidumbre",
    )

    seleccion = pronostico_grafico[
        pronostico_grafico["ds"].dt.year.isin(anios_seleccionados)
    ]
    ax.scatter(
        seleccion["ds"], seleccion["yhat"],
        s=60, zorder=5, label="Años seleccionados",
    )
    ax.set_title("Población histórica y predicción de Nicaragua")
    ax.set_xlabel("Año")
    ax.set_ylabel("Población")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def exportar_excel(tabla, tabla_metricas):
    """Crea un Excel descargable con las hojas Prediccion y Metricas."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        tabla.to_excel(writer, index=False, sheet_name="Predicción")
        tabla_metricas.to_excel(writer, index=False, sheet_name="Métricas")
    return buffer.getvalue()


def main():
    st.title("🇳🇮 Predicción de Población de Nicaragua")
    st.caption("Prophet · Datos del Banco Mundial · Modelo previamente entrenado")

    if not MODEL_PATH.is_file():
        st.error(
            "No se encontró 'modelo_prophet.json'. Subilo a la misma carpeta "
            "que 'app.py' en tu repositorio de GitHub."
        )
        st.stop()

    try:
        modelo = cargar_modelo()
        historico = cargar_historico(modelo)
        if historico.empty:
            st.error("El modelo no contiene datos históricos válidos.")
            st.stop()
        tabla_metricas = calcular_metricas()
    except Exception as exc:
        st.error("No se pudo cargar o evaluar el modelo Prophet.")
        st.exception(exc)
        st.stop()

    ultimo_anio_historico = int(historico["ds"].dt.year.max())
    anios_disponibles = list(range(ultimo_anio_historico + 1, ultimo_anio_historico + 26))

    st.subheader("Calidad del ajuste del modelo")
    metricas = dict(zip(tabla_metricas["Metrica"], tabla_metricas["Valor"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE · personas", f"{metricas['MAE']:,.2f}")
    c2.metric("RMSE · personas", f"{metricas['RMSE']:,.2f}")
    c3.metric("MAPE", f"{metricas['MAPE']:,.2f}%")
    with st.expander("Ver tabla y significado de las métricas"):
        st.dataframe(
            tabla_metricas.rename(columns={"Metrica": "Métrica"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "MAE: error absoluto promedio; RMSE: penaliza más los errores grandes; "
            "MAPE: error porcentual medio. Se calculan sobre los datos usados para "
            "entrenar el modelo, no sobre un conjunto de prueba independiente."
        )

    st.divider()
    st.subheader("Generar predicción")
    st.write(f"Último año observado: **{ultimo_anio_historico}**. Podés proyectar los 25 años siguientes.")

    with st.form("formulario_estimacion"):
        anios_seleccionados = st.multiselect(
            "Seleccioná uno o varios años",
            options=anios_disponibles,
            default=anios_disponibles[:5],
        )
        estimar = st.form_submit_button("Estimar población", type="primary")

    if estimar:
        if not anios_seleccionados:
            st.warning("Seleccioná al menos un año para generar la predicción.")
        else:
            try:
                with st.spinner("Calculando pronósticos..."):
                    tabla, pronostico_grafico = estimar_poblacion(
                        modelo, anios_seleccionados, ultimo_anio_historico
                    )
                st.session_state["tabla_pronosticos"] = tabla
                st.session_state["pronostico_grafico"] = pronostico_grafico
                st.session_state["anios_pronostico"] = sorted(anios_seleccionados)
            except Exception as exc:
                st.error("No fue posible realizar la estimación.")
                st.exception(exc)

    if "tabla_pronosticos" not in st.session_state:
        st.info("Elegí los años y presioná **Estimar población** para ver los resultados.")
        return

    tabla = st.session_state["tabla_pronosticos"]
    pronostico_grafico = st.session_state["pronostico_grafico"]
    anios_resultados = st.session_state["anios_pronostico"]

    st.subheader("Resultados de la predicción")
    st.dataframe(tabla, hide_index=True, width="stretch")

    st.subheader("Histórico y proyección")
    fig = construir_grafico(historico, pronostico_grafico, anios_resultados)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "La banda sombreada representa el intervalo de incertidumbre de Prophet; "
        "las estimaciones son proyecciones, no datos observados."
    )

    st.download_button(
        "📥 Descargar predicción en Excel",
        data=exportar_excel(tabla, tabla_metricas),
        file_name="prediccion_poblacion_nicaragua.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


if __name__ == "__main__":
    main()
