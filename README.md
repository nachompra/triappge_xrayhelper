# triappge_xrayhelper

![Logo](streamlit_utilities/TRiAPPGe%20Logo.png)

Este repositorio contiene un sistema de clasificación de radiografías con el objetivo de identificar y localizar fracturas óseas. Se emplean dos modelos de inteligencia artificial:

1. **Modelo de TensorFlow**: Este modelo predice la probabilidad de que una radiografía contenga una fractura.
2. **Modelo YOLO**: Localiza la ubicación exacta de la fractura, lo que aumenta la explicabilidad y precisión de las predicciones.

Además, se incluye una aplicación desarrollada con **Streamlit** para permitir la carga y visualización de radiografías, y proporcionar la predicción junto con la localización de la fractura (si la hay).

## Estructura del Repositorio

- `app_v2_clean.py`: Archivo principal de la aplicación Streamlit.
- `model/`: Carpeta con los modelos entrenados.
  - `best_binary_model_v3_augmented_reg.pkl`: Modelo de TensorFlow para predecir la probabilidad de fractura.
  - `best.pt`: Modelo YOLO para localizar la fractura.
- `data/`: Carpeta para almacenar las imágenes de radiografías.
- `requirements.txt`: Dependencias del proyecto.

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/nachompra/triappge_xrayhelper.git
    cd triappge_xrayhelper
    ```

2. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar la aplicación Streamlit, sigue estos pasos:

1. Ejecuta el siguiente comando para iniciar la aplicación:

    ```bash
    streamlit run app_v2_clean.py
    ```

2. Abre la aplicación en tu navegador, logate (de usa un correo y contraseña dummy) y sube una radiografía. El sistema te mostrará la probabilidad de fractura junto con la localización (si existe) sobre la imagen.

## Modelos

- **Modelo de TensorFlow**:
  - El modelo de TensorFlow está entrenado para clasificar radiografías y predecir la probabilidad de que contengan una fractura.
  - El modelo es una red neuronal convolucional (CNN) con varias capas de convolución y pooling.

- **Modelo YOLO**:
  - El modelo YOLO es utilizado para localizar la fractura en la radiografía. Marca la zona afectada con un cuadro delimitador, lo que permite una mayor interpretabilidad en las predicciones del modelo.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
