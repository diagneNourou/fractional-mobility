import torch
import numpy as np

from models.spatial_encoder import SpatialEncoder
from models.semantic_encoder import SemanticEncoder
from models.cross_attention import CrossAttention


# ==========================================
# 1. Charger les données
# ==========================================

X_spatial = np.load(
    "data/processed/X_spatial.npy"
)

GPS = np.load(
    "data/processed/GPS_spatial.npy"
)

X_semantic = np.load(
    "data/processed/X_semantic.npy"
)


print("X_spatial  :", X_spatial.shape)
print("GPS        :", GPS.shape)
print("X_semantic :", X_semantic.shape)


# ==========================================
# 2. Conversion PyTorch
# ==========================================

X_spatial_tensor = torch.tensor(
    X_spatial[:32],
    dtype=torch.float32
)

GPS_tensor = torch.tensor(
    GPS[:32],
    dtype=torch.float32
)

X_semantic_tensor = torch.tensor(
    X_semantic[:32],
    dtype=torch.float32
)


# ==========================================
# 3. Encodeur spatial
# ==========================================

spatial_encoder = SpatialEncoder(
    input_size=7,
    embedding_size=64,
    hidden_size=128
)

h_spatial, H_spatial, geographical_attention = (
    spatial_encoder(
        X_spatial_tensor,
        GPS_tensor,
        return_sequence=True
    )
)


# ==========================================
# 4. Encodeur sémantique
# ==========================================

semantic_encoder = SemanticEncoder(
    input_size=5,
    embedding_size=64,
    hidden_size=128
)

h_semantic, H_semantic = semantic_encoder(
    X_semantic_tensor,
    return_sequence=True
)


# ==========================================
# 5. Cross-Attention
# ==========================================

cross_attention = CrossAttention(
    hidden_size=128,
    num_heads=4
)

(
    spatial_from_semantic,
    semantic_from_spatial,
    attention_s2m,
    attention_m2s
) = cross_attention(
    H_spatial,
    H_semantic
)


# ==========================================
# 6. Résultats
# ==========================================

print("\n========== RÉSULTATS ==========")

print(
    "h_spatial :",
    h_spatial.shape
)

print(
    "H_spatial :",
    H_spatial.shape
)

print(
    "h_semantic :",
    h_semantic.shape
)

print(
    "H_semantic :",
    H_semantic.shape
)

print(
    "spatial_from_semantic :",
    spatial_from_semantic.shape
)

print(
    "semantic_from_spatial :",
    semantic_from_spatial.shape
)

print(
    "attention spatial → semantic :",
    attention_s2m.shape
)

print(
    "attention semantic → spatial :",
    attention_m2s.shape
)

print(
    "Somme attention s→m exemple 0 :",
    attention_s2m[0].sum().item()
)

print(
    "Somme attention m→s exemple 0 :",
    attention_m2s[0].sum().item()
)