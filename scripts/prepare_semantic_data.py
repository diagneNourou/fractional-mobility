import pandas as pd
import numpy as np
import os


# ==========================================
# 1. Charger les données
# ==========================================

input_file = "data/processed/geolife.csv"

df = pd.read_csv(input_file)

print("Nombre de lignes :", len(df))


# ==========================================
# 2. Préparation
# ==========================================

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.dropna(
    subset=["user_id", "datetime"]
)

df = df.sort_values(
    ["user_id", "datetime"]
).reset_index(drop=True)


# ==========================================
# 3. Extraction du contexte temporel
# ==========================================

# Heure de la journée
df["hour"] = df["datetime"].dt.hour

# Jour de la semaine
# Lundi = 0, ..., Dimanche = 6
df["day_of_week"] = df["datetime"].dt.dayofweek

# Week-end
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(float)


# ==========================================
# 4. Encodage cyclique de l'heure
# ==========================================

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


# ==========================================
# 5. Encodage cyclique du jour
# ==========================================

df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# ==========================================
# 6. Features sémantiques
# ==========================================

semantic_features = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_weekend"
]


# ==========================================
# 7. Nettoyage
# ==========================================

data = df.dropna(
    subset=semantic_features
).copy()


# ==========================================
# 8. Affichage
# ==========================================

print("\n========== FEATURES SÉMANTIQUES ==========")

print(
    data[
        [
            "datetime",
            "hour",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos"
        ]
    ].head(10)
)

print(
    "\nNombre de lignes après nettoyage :",
    len(data)
)


# ==========================================
# 9. Création des séquences
# ==========================================

SEQUENCE_LENGTH = 10

X_semantic = []

for i in range(10135):

    sequence = (
        data[semantic_features]
        .iloc[
            i:i + SEQUENCE_LENGTH
        ]
        .values
    )

    X_semantic.append(sequence)


X_semantic = np.array(
    X_semantic,
    dtype=np.float32
)


# ==========================================
# 10. Résultats
# ==========================================

print("\n========== SÉQUENCES SÉMANTIQUES ==========")

print(
    "X_semantic :",
    X_semantic.shape
)


# ==========================================
# 11. Sauvegarde
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_file = (
    "data/processed/X_semantic.npy"
)

np.save(
    output_file,
    X_semantic
)

print("\nDataset sémantique sauvegardé !")
print("Fichier :", output_file)