# Predicción de Población de Nicaragua · Streamlit

Dashboard en **Streamlit** que utiliza un modelo **Prophet previamente entrenado** para proyectar la población de Nicaragua.

## Archivos que debe tener el repositorio

- `app.py`: aplicación web Streamlit.
- `requirements.txt`: dependencias Python (ya no utiliza Gradio).
- `modelo_prophet.json`: modelo entrenado. **Conservá el JSON original que ya subiste a GitHub**; no se incluye en este paquete.
- `README.md`: documentación opcional.

El archivo `modelo_prophet.json` debe estar en la misma carpeta que `app.py`.

## Ejecutar en tu PC

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Reemplazá `app.py`, `requirements.txt` y opcionalmente `README.md` en el repositorio `capacitacionescorporativas-star/prediccion-prophet` de GitHub.
2. **No borres `modelo_prophet.json`.**
3. En Streamlit Community Cloud, seleccioná el repositorio y la rama `main`.
4. Escribí `app.py` como archivo principal.
5. En **Advanced settings**, seleccioná **Python 3.11** (evitá 3.14 para este proyecto).
6. Presioná **Deploy**. Si ya publicaste la aplicación con otra versión de Python, la versión solo se cambia eliminando y desplegando de nuevo la aplicación.

No se necesita configurar `PORT` ni llamar a `demo.launch()`. Streamlit Cloud ejecuta `streamlit run app.py` automáticamente.

## Funciones

- Selección de hasta 25 años futuros desde el último dato observado en el modelo.
- Tabla con población estimada e intervalos inferior/superior.
- Gráfico del histórico, pronóstico e incertidumbre.
- Métricas MAE, RMSE y MAPE **calculadas sobre el histórico de entrenamiento**.
- Descarga de Excel con las hojas **Predicción** y **Métricas**.

**Importante:** El paquete no contiene el JSON entrenado; las predicciones reales requieren ese archivo. Estas métricas no representan validación fuera de muestra.
