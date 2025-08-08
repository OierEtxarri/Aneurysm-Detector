import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
from scripts.train_3dcnn import Simple3DCNN, load_dicom_volume, LABEL_COLS

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
