import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import pickle
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# CARGAR MODELO YOLO
def load_yolo_model():
    return YOLO("models/best.pt")  # Modelo YOLOv8


# CARGAR MODELO CLASIFICACION
def load_classification_model():
    with open("models/best_binary_model_v3_augmented_reg.pkl", "rb") as f:
        return pickle.load(f)  # Modelo de predicción de fractura

# DIBUJAR SOBRE XRAY LA PREDICCION DE TYOLO
def draw_detections(image, results):
    image = np.array(image, dtype=np.uint8)

    if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return image

    detections = results[0]  # Primera predicción de YOLO

    for box in detections.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        label = f"Fracture ({conf:.2f})"

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def preprocess_images_streamlit(uploaded_files, target_size=(64, 64)):  # Changed to 224x224
    images = []
    for uploaded_file in uploaded_files:
        # Convertir archivo a imagen de OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Aplicar CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # FIXED TYPO HERE
        enhanced_image = clahe.apply(image)

        # Aplicar reducción de ruido
        denoised_image = cv2.GaussianBlur(enhanced_image, (5,5), 0)

        # Redimensionar la imagen (update to 224x224)
        resized_image = cv2.resize(denoised_image, target_size)

        # Normalizar a rango [0,1] y convertir a float32
        normalized_image = (resized_image / 255.0).astype(np.float32)

        # Convertir imagen en escala de grises (1 canal) a RGB (3 canales)
        rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)

        images.append(rgb_image)
    
    return np.array(images)