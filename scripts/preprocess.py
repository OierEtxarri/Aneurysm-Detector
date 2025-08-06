# Script para preprocesamiento de imágenes y metadatos del reto RSNA Intracranial Aneurysm Detection

import os
import pandas as pd
import numpy as np
import pydicom
import cv2

# Configuración de rutas
DATA_DIR = '../data/'
IMG_DIR = os.path.join(DATA_DIR, 'train_images')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')

# Cargar metadatos
train_df = pd.read_csv(TRAIN_CSV)

# Función para procesar una imagen DICOM

def preprocess_dicom(dicom_path, img_size=(256, 256)):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalización [0,1]
    img_resized = cv2.resize(img, img_size)
    return img_resized


# Procesar todos los estudios y guardar la primera imagen procesada de cada uno
output_dir = os.path.join(DATA_DIR, 'processed_images')
os.makedirs(output_dir, exist_ok=True)

processed_info = []

for i, row in train_df.iterrows():
    study_id = str(row['ID']) if 'ID' in row else None
    if study_id:
        study_path = os.path.join(IMG_DIR, study_id)
        if os.path.exists(study_path):
            dicom_files = [f for f in os.listdir(study_path) if f.endswith('.dcm')]
            if dicom_files:
                dicom_file = os.path.join(study_path, dicom_files[0])
                img = preprocess_dicom(dicom_file)
                # Guardar imagen procesada como .npy
                out_path = os.path.join(output_dir, f'{study_id}.npy')
                np.save(out_path, img)
                processed_info.append({'ID': study_id, 'img_path': out_path})
            else:
                print(f'No DICOM en estudio {study_id}')
        else:
            print(f'No carpeta para estudio {study_id}')
    else:
        print('ID de estudio no encontrado')

# Guardar información de imágenes procesadas
info_df = pd.DataFrame(processed_info)
info_df.to_csv(os.path.join(output_dir, 'processed_images_info.csv'), index=False)
