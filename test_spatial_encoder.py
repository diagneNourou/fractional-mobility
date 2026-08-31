import numpy as np
import torch

from models.spatial_encoder import SpatialEncoder


# ==========================================
# 1. Charger les données
# ==========================================

X = np.load(
    "data/processed/X_spatial.npy"
)

GPS = np.load(
    "data/processed/GPS_spatial.npy"
)

y = np.load(
    "data/processed/y_spatial.npy"
)


print("X :", X.shape)
print("GPS :", GPS.shape)
print("y :", y.shape)


# ==========================================
# 2. Transformer en tenseurs PyTorch
# ==========================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

GPS_tensor = torch.tensor(
    GPS,
    dtype=torch.float32
)


# ==========================================
# 3. Vérifier les dimensions
# ==========================================

print(
    "X_tensor :",
    X_tensor[:32].shape
)

print(
    "GPS_tensor :",
    GPS_tensor[:32].shape
)


# ==========================================
# 4. Créer le modèle
# ==========================================

model = SpatialEncoder(
    input_size=7,
    embedding_size=64,
    hidden_size=128
)


# ==========================================
# 5. Tester le modèle
# ==========================================

with torch.no_grad():

    h_spatial, attention = model(
        X_tensor[:32],
        GPS_tensor[:32]
    )


# ==========================================
# 6. Afficher les résultats
# ==========================================

print()
print("========== RÉSULTATS ==========")

print(
    "h_spatial :",
    h_spatial.shape
)

print(
    "attention :",
    attention.shape
)

print(
    "Somme attention exemple 0 :",
    attention[0].sum().item()
)

print(
    "Attention exemple 0 :",
    attention[0]
)