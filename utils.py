import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from PIL import Image 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import pickle
from sklearn.cluster import KMeans

#Plot Confusion Matrix
def plot_conf_matrix(modelo, test_img, test_lbl):
    y_pred = modelo.predict(test_img)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Threshold at 0.5

    # Compute confusion matrix
    cm = confusion_matrix(test_lbl, y_pred_classes)

    # Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fractured', 'Fractured'], 
                yticklabels=['Not Fractured', 'Fractured'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.show()
#Preprocess images
def preprocess_images(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image in grayscale
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            # Apply Gaussian blur
            denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
            
            # Resize to target size
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalize to [0, 1] and cast to float32
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Convert grayscale (1-channel) to 3-channel RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Append image and label
            images.append(rgb_image)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_images_sobel(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image in grayscale
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)

            # Apply Sobel filter to detect edges
            sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))  # Normalize to 0-255

            # Apply Gaussian blur to reduce noise
            denoised_image = cv2.GaussianBlur(sobel_magnitude, (5, 5), 0)
            
            # Resize to target size
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalize to [0, 1] and cast to float32
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Convert grayscale (1-channel) to 3-channel RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Append image and label
            images.append(rgb_image)
            labels.append(label)

    return np.array(images), np.array(labels)

def preprocess_images_negative(directory, target_size=(64, 64), label=None):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Load image in grayscale
            image = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            
            # Apply negative transformation (invert the image)
            negative_image = cv2.bitwise_not(image)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(negative_image)

            # Apply Gaussian blur
            denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
            
            # Resize to target size
            resized_image = cv2.resize(denoised_image, target_size)
            
            # Normalize to [0, 1] and cast to float32
            normalized_image = (resized_image / 255.0).astype(np.float32)
            
            # Convert grayscale (1-channel) to 3-channel RGB
            rgb_image = np.stack([normalized_image] * 3, axis=-1).astype(np.float32)
            
            # Append image and label
            images.append(rgb_image)
            labels.append(label)
    return np.array(images), np.array(labels)


def load_dataset_split(split_dir):
    # Paths to class directories
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Load and preprocess images
    fracture_images, fracture_labels = preprocess_images(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images(non_fracture_dir, label=0)
    
    # Combine classes
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_inv(split_dir):
    # Paths to class directories
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Load and preprocess images
    fracture_images, fracture_labels = preprocess_images_negative(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_negative(non_fracture_dir, label=0)
    
    # Combine classes
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def load_dataset_split_sobel(split_dir):
    # Paths to class directories
    fracture_dir = os.path.join(split_dir, 'fractured')
    non_fracture_dir = os.path.join(split_dir, 'not fractured')
    
    # Load and preprocess images
    fracture_images, fracture_labels = preprocess_images_sobel(fracture_dir, label=1)
    non_fracture_images, non_fracture_labels = preprocess_images_sobel(non_fracture_dir, label=0)
    
    # Combine classes
    images = np.concatenate((fracture_images, non_fracture_images), axis=0)
    labels = np.concatenate((fracture_labels, non_fracture_labels), axis=0)
    
    return images, labels

def plot_roc_curve(modelo, test_img, test_lbl):
    # Compute ROC curve
    y_pred = modelo.predict(test_img)
    fpr, tpr, thresholds = roc_curve(test_lbl, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_pr_curve(modelo, test_img, test_lbl):
    y_pred = modelo.predict(test_img)
    precision, recall, _ = precision_recall_curve(test_lbl, y_pred)
    average_precision = average_precision_score(test_lbl, y_pred)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
            label=f'PR Curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()


def p_proba_dist(modelo, test_img, test_lbl):
    y_pred = modelo.predict(test_img)
    plt.figure(figsize=(8, 6))
    plt.hist(y_pred[test_lbl == 0], bins=30, alpha=0.5, label='Not Fractured')
    plt.hist(y_pred[test_lbl == 1], bins=30, alpha=0.5, label='Fractured')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.show()

def show_ten_samples(modelo, test_img, test_lbl):
# Plot 10 sample test images with predictions
    y_pred = modelo.predict(test_img)
    plt.figure(figsize=(15, 8))
    for i in range(10):
        idx = np.random.randint(0, len(test_img))
        img = test_img[idx]
        true_label = test_lbl[idx]
        pred_prob = y_pred[idx][0]
        pred_class = "Fractured" if pred_prob > 0.5 else "Not Fractured"
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img[:, :, 0], cmap='gray')  # Use first channel (grayscale)
        plt.title(f'True: {"Fractured" if true_label == 1 else "Not"}\nPred: {pred_class} ({pred_prob:.2f})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def loss_accu_train(hist):
    # Plot training vs validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    # Plot training vs validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

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
