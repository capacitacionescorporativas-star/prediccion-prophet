# Predicción de Población de Nicaragua

Aplicación Gradio para publicar en Render un modelo Prophet previamente entrenado.

## Archivos

- `app.py`: aplicación Gradio y lógica de predicción.
- `modelo_prophet.json`: modelo Prophet serializado.
- `requirements.txt`: dependencias de Python.
- `.python-version`: versión de Python usada por Render.

## Configuración en Render

Crear un **Web Service** conectado al repositorio.

- Language: `Python 3`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`

La aplicación toma automáticamente el puerto desde la variable de entorno `PORT` y escucha en `0.0.0.0`.

## Prueba local

```bash
pip install -r requirements.txt
python app.py
```

Luego abrir `http://127.0.0.1:7860`.

## Opción Blueprint

El archivo `render.yaml` permite crear el servicio como Render Blueprint. Si se crea manualmente desde **New > Web Service**, se pueden usar los mismos comandos indicados arriba.
