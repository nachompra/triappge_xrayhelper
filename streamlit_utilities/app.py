import streamlit as st
import cv2 as cv
import numpy as np
import tensorflow as tf
from io import BytesIO
import pickle
from tensorflow.keras.models import load_model

from utils import *
st.image('streamlit_utilities\TRiAPPGe Logo.png', width=300)

# Cargar imágenes
uploaded_files = st.file_uploader(label='XRAY Loader', 
                                type=['png', 'jpg', 'jpeg'], 
                                accept_multiple_files=True)

if st.button('Predict'):
    if uploaded_files:
        target_array = preprocess_images_streamlit(uploaded_files)


        model_path = 'binary_model_v3_augmented_reg.pkl'
        model = pickle.load(open(model_path, 'rb')) 

        # Predicción
        predictions = model.predict(target_array)
        predicted_classes = (predictions > 0.5).astype(int)

        # Mapeamos clases a etiquetas
        labels = {0: 'Non Fractured', 1: 'Fractured'}
        
        # Mostrar los resultados con las imágenes
        for i, uploaded_file in enumerate(uploaded_files):
            # Mostrar imagen original y procesada
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_container_width=True)
            # with col2:
            #     processed_img = (target_array[i] * 255).astype(np.uint8)
            #     st.image(processed_img, caption="Processed Image", use_container_width=True)
            
            st.subheader(f"Prediction: {labels[predicted_classes[i][0]]}")
            st.progress(float(predictions[i][0]))
            st.write(f"Fracture Probability: {predictions[i][0]:.2%}")

    else:
        st.write("Please upload an X-ray image.")

import pickle

try:
    with open("binary_model_v3_augmented.pkl", "rb") as f:
        model = pickle.load(f)
    print("Modelo cargado correctamente:", type(model))
except Exception as e:
    print("Error al cargar el modelo:", e)