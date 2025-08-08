import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
from scripts.train_3dcnn import Simple3DCNN, LABEL_COLS
def load_dicom_volume(series_uid, target_shape=(64, 128, 128)):
    series_path = os.path.join(SERIES_DIR, str(series_uid))
    if not os.path.exists(series_path):
        return np.zeros(target_shape, dtype=np.float32)
    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not dicom_files:
        return np.zeros(target_shape, dtype=np.float32)
    dicom_files.sort()
    slices = []
    for f in dicom_files:
        dcm = pydicom.dcmread(os.path.join(series_path, f))
        img = dcm.pixel_array.astype(np.float32)
        # Forzar a 2D
        if img.ndim == 1:
            img = np.expand_dims(img, axis=0)
        elif img.ndim > 2:
            img = img.squeeze()
        if img.ndim != 2:
            continue  # descarta imágenes corruptas
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        slices.append(img)
    if len(slices) == 0:
        return np.zeros(target_shape, dtype=np.float32)
    try:
        volume = np.stack(slices, axis=0)
    except Exception:
        return np.zeros(target_shape, dtype=np.float32)
    if volume.ndim == 2:
        volume = np.expand_dims(volume, axis=0)
    if volume.ndim != 3:
        return np.zeros(target_shape, dtype=np.float32)
    from scipy.ndimage import zoom
    factors = [t/s for t, s in zip(target_shape, volume.shape)]
    volume = zoom(volume, factors, order=1)
    return volume

DATA_DIR = 'data'
SERIES_DIR = os.path.join(DATA_DIR, 'series')

TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
LOCALIZERS_CSV = os.path.join(DATA_DIR, 'train_localizers.csv')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Aneurysm3DTestDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = load_dicom_volume(row['SeriesInstanceUID'])
        if vol is None:
            vol = np.zeros((64,128,128), dtype=np.float32)
        vol = np.expand_dims(vol, 0)
        return torch.tensor(vol, dtype=torch.float32), row['SeriesInstanceUID']

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_df = pd.read_csv(TEST_CSV)
    # Fusionar test.csv con localizadores si existen (aunque normalmente test.csv no tiene localizadores)
    if os.path.exists(LOCALIZERS_CSV):
        localizers_df = pd.read_csv(LOCALIZERS_CSV)
        df_test = pd.merge(test_df, localizers_df, on='SeriesInstanceUID', how='left', suffixes=('', '_localizer'))
    else:
        df_test = test_df.copy()
    test_set = Aneurysm3DTestDataset(df_test)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=2)
    model = Simple3DCNN(out_dim=len(LABEL_COLS)).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_3dcnn.pth'), map_location=device))
    model.eval()
    results = []
    with torch.no_grad():
        for x, series_uids in test_loader:
            x = x.to(device)
            out = torch.sigmoid(model(x)).cpu().numpy()
            for i, uid in enumerate(series_uids):
                row = {'SeriesInstanceUID': uid}
                for j, col in enumerate(LABEL_COLS):
                    row[col] = out[i, j]
                results.append(row)
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, '3dcnn_predictions.csv'), index=False)
    print('Predicciones guardadas en', os.path.join(OUTPUT_DIR, '3dcnn_predictions.csv'))

if __name__ == '__main__':
    predict()
