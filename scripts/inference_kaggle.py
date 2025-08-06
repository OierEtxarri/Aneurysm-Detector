import os
import shutil
from collections import defaultdict
import pandas as pd
import polars as pl
import pydicom
import numpy as np
import cv2
import joblib
import kaggle_evaluation.rsna_inference_server

ID_COL = 'SeriesInstanceUID'
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

# Preprocesamiento igual al entrenamiento


DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID'
]

def preprocess_dicom_with_tags(dicom_path, img_size=(256, 256)):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
    img_resized = cv2.resize(img, img_size)
    img_flat = img_resized.flatten()
    # Extraer tags permitidos y convertirlos en features
    tag_features = []
    for tag in DICOM_TAG_ALLOWLIST:
        val = getattr(ds, tag, None)
        # Convertir listas/tuplas a valores simples
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else 0
        # Convertir None a 0, strings a hash
        if val is None:
            val = 0
        elif isinstance(val, str):
            val = hash(val) % 10000  # Simple hash para strings
        tag_features.append(float(val))
    # Concatenar imagen y tags
    return np.concatenate([img_flat, np.array(tag_features)])


# Cargar todos los modelos en un diccionario
models = {}
def load_models():
    global models
    for label in LABEL_COLS:
        model_path = f'/kaggle/working/rf_model_{label.replace(" ", "_")}.joblib'
        if os.path.exists(model_path):
            models[label] = joblib.load(model_path)
        else:
            models[label] = None
load_models()


def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    series_id = os.path.basename(series_path)
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    if all_filepaths:
        features = preprocess_dicom_with_tags(all_filepaths[0])
        X = np.array([features])
    else:
        # Si no hay imagen, vector de ceros del tamaño adecuado
        X = np.zeros((1, 256*256 + len(DICOM_TAG_ALLOWLIST)))
    preds = []
    for label in LABEL_COLS:
        model = models.get(label)
        if model is not None:
            proba = model.predict_proba(X)[0, 1]
            preds.append(proba)
        else:
            preds.append(0.5)
    predictions = pl.DataFrame(
        data=[[series_id] + preds],
        schema=[ID_COL, *LABEL_COLS],
        orient='row',
    )
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return predictions.drop(ID_COL)

inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))
