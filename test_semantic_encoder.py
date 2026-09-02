import torch
import numpy as np

from models.semantic_encoder import SemanticEncoder


# ==========================================
# 1. Charger les données
# ==========================================

X_semantic = np.load(
    "data/processed/X_semantic.npy"
)

print("X_semantic :", X_semantic.shape)


# ==========================================
# 2. Conversion en tenseur PyTorch
# ==========================================

X_tensor = torch.tensor(
    X_semantic,
    dtype=torch.float32
)

print(
    "X_tensor :",
    X_tensor.shape
)


# ==========================================
# 3. Créer le modèle
# ==========================================

model = SemanticEncoder(
    input_size=5,
    embedding_size=64,
    hidden_size=128
)


# ==========================================
# 4. Tester avec un batch de 32
# ==========================================

h_semantic, H_semantic = model(
    X_tensor[:32],
    return_sequence=True
)


# ==========================================
# 5. Résultats
# ==========================================

print("\n========== RÉSULTATS ==========")

print(
    "h_semantic :",
    h_semantic.shape
)

print(
    "H_semantic :",
    H_semantic.shape
)