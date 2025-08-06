# Script para entrenamiento de modelo predictivo de aneurisma cerebral


import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Configuración de rutas

DATA_DIR = '../data/'
IMG_DIR = os.path.join(DATA_DIR, 'processed_images')
INFO_CSV = os.path.join(IMG_DIR, 'processed_images_info.csv')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')


# Cargar metadatos y paths de imágenes
train_df = pd.read_csv(TRAIN_CSV)
info_df = pd.read_csv(INFO_CSV)

# Usar SeriesInstanceUID como identificador
df = pd.merge(train_df, info_df, left_on='SeriesInstanceUID', right_on='ID')


# Cargar imágenes y preparar X, y

# DICOM tags to use as features
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID'
]

import pydicom

def preprocess_dicom_with_tags(dicom_path, img_size=(256, 256)):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img_resized = cv2.resize(img, img_size)
    img_flat = img_resized.flatten()
    tag_features = []
    for tag in DICOM_TAG_ALLOWLIST:
        val = getattr(ds, tag, None)
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else 0
        if val is None:
            val = 0
        elif isinstance(val, str):
            val = hash(val) % 10000
        tag_features.append(float(val))
    return np.concatenate([img_flat, np.array(tag_features)])

X = []
for img_path in df['img_path']:
    X.append(preprocess_dicom_with_tags(img_path))
X = np.array(X)


# Entrenamiento y guardado de 14 modelos independientes
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

import joblib


# Guardar las probabilidades y valores reales para el cálculo global
y_val_matrix = []
y_proba_matrix = []

for label in LABEL_COLS:
    if label not in df.columns:
        print(f"Column {label} not found in dataframe. Skipping.")
        continue
    y = df[label]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:,1]
    print(f"\nModel for {label}")
    print('ROC AUC:', roc_auc_score(y_val, y_proba))
    print(classification_report(y_val, y_pred))
    model_path = os.path.join(DATA_DIR, f"rf_model_{label.replace(' ', '_')}.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")
    y_val_matrix.append(y_val)
    y_proba_matrix.append(y_proba)

# Calcular Mean Weighted Columnwise AUCROC
if len(y_val_matrix) == 14:
    y_val_matrix = np.array(y_val_matrix).T  # shape (n_samples, 14)
    y_proba_matrix = np.array(y_proba_matrix).T
    aucs = [roc_auc_score(y_val_matrix[:,i], y_proba_matrix[:,i]) for i in range(14)]
    final_score = (aucs[-1] + np.mean(aucs[:-1])) / 2
    print(f"\nMean Weighted Columnwise AUCROC: {final_score:.4f}")
