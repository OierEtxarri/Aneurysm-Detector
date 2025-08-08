import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pydicom
import nibabel as nib
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm

# Configuración de rutas
DATA_DIR = 'data'
SERIES_DIR = os.path.join(DATA_DIR, 'series')
SEGMENTATIONS_DIR = os.path.join(DATA_DIR, 'segmentations')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
LOCALIZERS_CSV = os.path.join(DATA_DIR, 'train_localizers.csv')
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
]

# Utilidades para cargar volúmenes 3D

def load_dicom_volume(series_uid, target_shape=(64, 128, 128)):
    series_path = os.path.join(SERIES_DIR, str(series_uid))
    if not os.path.exists(series_path):
        return None
    dicom_files = [f for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not dicom_files:
        return None
    dicom_files.sort()
    slices = []
    for f in dicom_files:
        dcm = pydicom.dcmread(os.path.join(series_path, f))
        img = dcm.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        slices.append(img)
    volume = np.stack(slices, axis=0)
    # Resize to target_shape
    volume = resize_volume(volume, target_shape)
    return volume

def resize_volume(vol, target_shape):
    # vol: (D, H, W), target_shape: (D, H, W)
    from scipy.ndimage import zoom
    factors = [t/s for t, s in zip(target_shape, vol.shape)]
    vol = zoom(vol, factors, order=1)
    return vol

# Dataset PyTorch
class Aneurysm3DDataset(Dataset):
    def __init__(self, df, label_cols=LABEL_COLS, augment=False):
        self.df = df
        self.label_cols = label_cols
        self.augment = augment
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vol = load_dicom_volume(row['SeriesInstanceUID'])
        if vol is None:
            vol = np.zeros((64,128,128), dtype=np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                vol = np.flip(vol, axis=2).copy()
        vol = np.expand_dims(vol, 0)  # (1, D, H, W)
        label = row[self.label_cols].values.astype(np.float32)
        return torch.tensor(vol, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# Modelo 3D CNN
class Simple3DCNN(nn.Module):
    def __init__(self, out_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            nn.AdaptiveAvgPool3d((2,4,4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*2*4*4, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.append(torch.sigmoid(out).cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return np.mean(losses), preds, targets

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_df = pd.read_csv(TRAIN_CSV)
    if os.path.exists(LOCALIZERS_CSV):
        localizers_df = pd.read_csv(LOCALIZERS_CSV)
        df = pd.merge(train_df, localizers_df, on='SeriesInstanceUID', how='left', suffixes=('', '_localizer'))
    else:
        df = train_df.copy()
    # Filtrar filas con todas las labels nulas
    df = df.dropna(subset=LABEL_COLS, how='all').reset_index(drop=True)
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    train_set = Aneurysm3DDataset(df.iloc[train_idx].reset_index(drop=True), augment=True)
    val_set = Aneurysm3DDataset(df.iloc[val_idx].reset_index(drop=True), augment=False)
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=2)
    model = Simple3DCNN(out_dim=len(LABEL_COLS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_auc = 0
    for epoch in range(1, 11):
        print(f'Epoch {epoch}')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = eval_epoch(model, val_loader, criterion, device)
        aucs = []
        for i in range(len(LABEL_COLS)):
            try:
                auc = roc_auc_score(val_targets[:,i], val_preds[:,i])
            except:
                auc = np.nan
            aucs.append(auc)
        mean_auc = np.nanmean(aucs)
        print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Mean AUC: {mean_auc:.4f}')
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_3dcnn.pth'))
    print('Entrenamiento finalizado. Mejor modelo guardado.')

if __name__ == '__main__':
    main()
