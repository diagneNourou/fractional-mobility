import pandas as pd
import numpy as np
import os


# ==========================================
# 1. Charger les données
# ==========================================

input_file = "data/processed/geolife.csv"

df = pd.read_csv(input_file)

print("Données chargées :", len(df), "lignes")


# ==========================================
# 2. Préparation
# ==========================================

df["datetime"] = pd.to_datetime(df["datetime"])

df = df.dropna(
    subset=["user_id", "lat", "lon", "datetime"]
)

df = df.sort_values(
    ["user_id", "datetime"]
).reset_index(drop=True)


# ==========================================
# 3. Normalisation
# ==========================================

lat_mean = df["lat"].mean()
lat_std = df["lat"].std()

lon_mean = df["lon"].mean()
lon_std = df["lon"].std()

df["lat_norm"] = (
    (df["lat"] - lat_mean) / lat_std
)

df["lon_norm"] = (
    (df["lon"] - lon_mean) / lon_std
)


# ==========================================
# 4. Création des séquences
# ==========================================

SEQUENCE_LENGTH = 10

X = []
y = []


# IMPORTANT :
# On traite chaque utilisateur séparément

for user_id, user_df in df.groupby("user_id"):

    user_df = user_df.sort_values("datetime")

    positions = user_df[
        ["lat_norm", "lon_norm"]
    ].values

    # Pas assez de points pour créer une séquence
    if len(positions) <= SEQUENCE_LENGTH:
        continue

    for i in range(
        len(positions) - SEQUENCE_LENGTH
    ):

        sequence = positions[
            i:i + SEQUENCE_LENGTH
        ]

        target = positions[
            i + SEQUENCE_LENGTH
        ]

        X.append(sequence)
        y.append(target)


# ==========================================
# 5. Conversion NumPy
# ==========================================

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)


# ==========================================
# 6. Affichage
# ==========================================

print("\n========== RÉSULTATS ==========")

print("Nombre de séquences :", len(X))

print("Dimension X :", X.shape)

print("Dimension y :", y.shape)


# ==========================================
# 7. Sauvegarde
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

np.save(
    "data/processed/X.npy",
    X
)

np.save(
    "data/processed/y.npy",
    y
)


print("\nSéquences sauvegardées !")

print("X -> data/processed/X.npy")
print("y -> data/processed/y.npy")