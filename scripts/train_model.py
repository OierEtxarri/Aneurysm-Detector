# Script para entrenamiento de modelo predictivo de aneurisma cerebral

import os
import pandas as pd
import numpy as np
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

df = pd.merge(train_df, info_df, on='ID')

# Cargar imágenes y preparar X, y
X = []
for img_path in df['img_path']:
    img = np.load(img_path)
    X.append(img.flatten())
X = np.array(X)

y = df['any_aneurysm'] if 'any_aneurysm' in df.columns else None

# Separar en train y test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo simple (Random Forest)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluar
y_pred = clf.predict(X_val)
y_proba = clf.predict_proba(X_val)[:,1]
print('ROC AUC:', roc_auc_score(y_val, y_proba))
print(classification_report(y_val, y_pred))

# Guardar modelo entrenado
import joblib
joblib.dump(clf, os.path.join(DATA_DIR, 'rf_model.joblib'))
