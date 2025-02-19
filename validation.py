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

#Visualizar Confusion Matrix
def plot_conf_matrix(modelo, test_img, test_lbl):
    y_pred = modelo.predict(test_img)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Umbral de 0.5 para convertir en 0 o 1

    # Crea la confusion matrix
    cm = confusion_matrix(test_lbl, y_pred_classes)

    # Crea la visualizacion
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fractured', 'Fractured'], 
                yticklabels=['Not Fractured', 'Fractured'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.show()

def plot_conf_matrix_pct(modelo, test_img, test_lbl):
    y_pred = modelo.predict(test_img)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Umbral de 0.5 para convertir en 0 o 1

    # Crea la matriz de confusión
    cm = confusion_matrix(test_lbl, y_pred_classes)
    
    # Normalizar para visualizar porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Crea la visualizacion
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=['Not Fractured', 'Fractured'], 
                yticklabels=['Not Fractured', 'Fractured'])

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (%) - Normalized per Class')

    plt.show()

def plot_roc_curve(modelo, test_img, test_lbl):
# Curva ROC
    y_pred = modelo.predict(test_img)
    fpr, tpr, thresholds = roc_curve(test_lbl, y_pred)
    roc_auc = auc(fpr, tpr)

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
# Curva Prediction vs Recall
    y_pred = modelo.predict(test_img)
    precision, recall, _ = precision_recall_curve(test_lbl, y_pred)
    average_precision = average_precision_score(test_lbl, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
            label=f'PR Curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()


def p_proba_dist(modelo, test_img, test_lbl):
# Mostrar la distribución de Probabilidad
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
# Mostrar 10 Ejemplos
    y_pred = modelo.predict(test_img)
    plt.figure(figsize=(15, 8))
    for i in range(10):
        idx = np.random.randint(0, len(test_img))
        img = test_img[idx]
        true_label = test_lbl[idx]
        pred_prob = y_pred[idx][0]
        pred_class = "Fractured" if pred_prob > 0.5 else "Not Fractured"
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.title(f'True: {"Fractured" if true_label == 1 else "Not"}\nPred: {pred_class} ({pred_prob:.2f})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def loss_accu_train(hist):
    # Training loss vs Val Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')

    # Training Acc vs Val Acc
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()
