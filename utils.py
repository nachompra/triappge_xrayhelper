import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from PIL import Image 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import pickle
from sklearn.cluster import KMeans

def preprocess_images(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            # Aplicamos Gaussian blur para reducir Ruido
            denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
            
            # Ajustamos tamaño a entrada del modelo
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Creamos imagenes procesadas y labels
            images.append(rgb_image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_images_128(directory, target_size=(128, 128), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            # Aplicamos Gaussian blur para reducir Ruido
            denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
            
            # Ajustamos tamaño a entrada del modelo (Mayor definición)
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Creamos imagenes procesadas y labels
            images.append(rgb_image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_images_sobel(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            # Aplicamos Sobel (detección de límites de imagen)
            sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))  

            # Aplicamos Gaussian blur para reducir Ruido
            denoised_image = cv2.GaussianBlur(sobel_magnitude, (5, 5), 0)
            
            # Ajustamos tamaño a entrada del modelo
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Creamos imagenes procesadas y labels
            images.append(rgb_image)
            labels.append(label)

    return np.array(images), np.array(labels)


def preprocess_images_sobel_v2(directory, target_size=(64, 64), label=None, sigma=4.75, threshold=0.1):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Aplicamos Gaussian blur para reducir Ruido
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            
            # Aplicamos Sobel (detección de límites de imagen)
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Generamos un gradiante
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalizamos min/max
            gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
            
            # Establecemos límites binarios
            binary_edges = (gradient_magnitude > threshold).astype(np.uint8) * 255
            
            # Ajustamos tamaño a entrada del modelo
            resized_image = cv2.resize(binary_edges, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            images.append(rgb_image)
            labels.append(label)

    return np.array(images), np.array(labels)

def preprocess_images_negative(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Invertimos la escala de grises
            negative_image = cv2.bitwise_not(image)
            
            # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(negative_image)

            # Aplicamos Gaussian blur para reducir Ruido
            denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
            
            # Ajustamos tamaño a entrada del modelo
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Volvemos a convertir a tres canales RGB
            images.append(rgb_image)
            labels.append(label)
    return np.array(images), np.array(labels)


def load_dataset_split(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_128(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_128(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_128(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_inv(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_negative(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_negative(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_sobel(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_sobel(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_sobel_v2(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_sobel_v2(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel_v2(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_sobel_v3(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
   # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_sobel_v3(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel_v3(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_sobel_v4(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_sobel_v4(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel_v4(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels
def load_dataset_split_sobel_v5(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_sobel_v5(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel_v5(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_canny(split_dir):
    # Rutas a las carpetas con las clases
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Cargamos y procesamos imagenes
    fracture_images, fracture_labels = preprocess_images_canny(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_canny(non_fracture_dir, label=0)
    
    # Combinamos el procesamiento de las dos carpetas
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def preprocess_images_sobel_v3(directory, target_size=(64, 64), label=None, sigma=1.5, threshold=0.2):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # # Aplicamos Gaussian Blur para reducir ruido
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            
            # Aplicamos Sobel (detección de límites de imagen)
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Generamos un gradiante
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_direction = np.arctan2(sobel_y, sobel_x)
            
            #Aplicamos la función para quedarnos con los límites más finos
            thinned_edges = non_maximum_suppression(gradient_magnitude, gradient_direction)
            
            #Pasamos los límites a binario
            binary_edges = (thinned_edges > threshold).astype(np.uint8) * 255
            
            # Ajustamos tamaño a input del modelo
            resized_image = cv2.resize(binary_edges, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a 3 canales
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Guardamos la imagen procesada y su label
            images.append(rgb_image)
            labels.append(label)

    return np.array(images), np.array(labels)

def non_maximum_suppression(magnitude, direction):
 
    # Convertimos angulos a grados
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180
    
    # Creamos una matriz de ceros para ingestar los límites
    thinned = np.zeros_like(magnitude)
    
    # Iteramos sobre cada pixel
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            # Determinamos los vecinos en función el gradiante del eje
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            elif 112.5 <= angle[i, j] < 157.5:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            
            # Eliminamos los pixels que no lleguen a un valor máximo
            if magnitude[i, j] >= max(neighbors):
                thinned[i, j] = magnitude[i, j]
    
    return thinned

def preprocess_images_canny(directory, target_size=(64, 64), label=None, sigma=2, low_threshold_ratio=0.7, high_threshold_ratio=1.2):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # # Aplicamos Gaussian Blur para reducir ruido
            blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            
            # Aplicamos Sobel (detección de límites de imagen)
            sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Ajustamos treshold de forma dinámica
            median_gradient = np.median(gradient_magnitude)
            high_threshold = median_gradient * high_threshold_ratio
            low_threshold = high_threshold * low_threshold_ratio
            
            # Aplicamos la detección Canny de límites
            edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
            
            # Ajustamos tamaño al input del modelo
            resized_image = cv2.resize(edges, target_size)
            
            # Normalizamos min/max
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Volvemos a convertir en 3 Canales
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Guardamos la imagen procesada y su label
            images.append(rgb_image)
            labels.append(label)


def preprocess_images_sobel_v4(directory, target_size=(224, 224), label=None):
    images = []
    labels = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
           # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # # Aplicamos Gaussian Blur para reducir ruido
            denoised = cv2.medianBlur(image, 5)
            
            # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Aplicamos Sobel (detección de límtes)
            sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Usamos el metodo Otsu para ajustar el treshold de forma dinámica
            _, binary_edges = cv2.threshold(
                np.uint8(gradient_magnitude / np.max(gradient_magnitude) * 255),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Cerramos los límites
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed_edges = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, kernel)
            
            # Afinamos los límites
            skeleton = cv2.ximgproc.thinning(closed_edges)
            
            # Ajustamos tamaño a input del modelo
            resized = cv2.resize(skeleton, target_size)
            
            # Volvemos a convertir en imagen de 3 canales
            rgb_image = np.stack([resized / 255.0] * 3, axis=-1).astype(np.float32)
            
            images.append(rgb_image)
            labels.append(label)
    
    return np.array(images), np.array(labels)


def preprocess_images_sobel_v5(directory, target_size=(128, 128), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Cargamos en escala de grises
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
           # # Aplicamos Gaussian Blur para reducir ruido
            denoised = cv2.medianBlur(image, 5)
            
           # Aplicamos CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Aplicamos Sobel y Canny para la deteccion de límites
            sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            canny_edges = cv2.Canny(enhanced, threshold1=30, threshold2=100)
            
            # Cerramos los límites
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)
            
            # Afinamos los límites
            skeleton = cv2.ximgproc.thinning(closed_edges)
            
            # Step 5: Combinamos features
            combined = np.stack([
                enhanced / 255.0,  # CLAHE-enhanced grayscale
                gradient_magnitude / np.max(gradient_magnitude),  # Sobel edge strength
                skeleton / 255.0  # Skeletonized edges
            ], axis=-1)
            
            # Ajustamos tamaño al input del modelo
            resized = cv2.resize(combined, target_size)
            
            images.append(resized)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def create_pkl(filename, modelo):
    with open(filename, 'wb') as archivo_salida:
        pickle.dump(modelo, archivo_salida)


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

def preprocess_images_streamlit_sobel(uploaded_files, target_size=(64, 64)):
    images = []
    for uploaded_file in uploaded_files:
        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)

        # Apply Sobel filter to detect edges
        sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))  

        # Apply Gaussian blur to reduce noise
        denoised_image = cv2.GaussianBlur(sobel_magnitude, (5, 5), 0)
        
        # Resize to target size
        resized_image = cv2.resize(denoised_image, target_size)
        
        # Normalize to [0, 1] and cast to float32
        normalized_image = (resized_image / 255.0).astype(np.float32)
        
        # Convert grayscale (1-channel) to 3-channel RGB
        rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
        
        # Append image
        images.append(rgb_image)
    
    return np.array(images)
