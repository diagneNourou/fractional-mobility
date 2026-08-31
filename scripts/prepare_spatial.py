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
    subset=["user_id", "lat", "lon", "datetime"]
)

df = df.sort_values(
    ["user_id", "datetime"]
).reset_index(drop=True)


# ==========================================
# 3. Distance de Haversine
# ==========================================

def haversine(lat1, lon1, lat2, lon2):

    R = 6371.0  # rayon terrestre en km

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return R * c


# ==========================================
# 4. Calcul des caractéristiques
# ==========================================

spatial_data = []


for user_id, user_df in df.groupby("user_id"):

    user_df = user_df.sort_values(
        "datetime"
    ).copy()

    # Position précédente
    user_df["prev_lat"] = user_df["lat"].shift(1)
    user_df["prev_lon"] = user_df["lon"].shift(1)
    user_df["prev_datetime"] = user_df["datetime"].shift(1)

    # --------------------------------------
    # Distance
    # --------------------------------------

    user_df["delta_d_km"] = haversine(
        user_df["prev_lat"],
        user_df["prev_lon"],
        user_df["lat"],
        user_df["lon"]
    )

    # --------------------------------------
    # Temps écoulé
    # --------------------------------------

    user_df["delta_t_sec"] = (
        user_df["datetime"]
        - user_df["prev_datetime"]
    ).dt.total_seconds()

    # --------------------------------------
    # Vitesse
    # --------------------------------------

    # km/h
    user_df["speed_ms"] = (
        user_df["delta_d_km"]* 1000
        / user_df["delta_t_sec"]
    )

    # --------------------------------------
    # Direction theta
    # --------------------------------------

    lat1 = np.radians(
        user_df["prev_lat"]
    )

    lat2 = np.radians(
        user_df["lat"]
    )

    delta_lon = np.radians(
        user_df["lon"]
        - user_df["prev_lon"]
    )

    delta_lat = (
        user_df["lat"]
        - user_df["prev_lat"]
    )

    user_df["theta"] = np.arctan2(
        delta_lon * np.cos(lat2),
        delta_lat
    )

    # --------------------------------------
    # Sinus et cosinus
    # --------------------------------------

    user_df["sin_theta"] = np.sin(
        user_df["theta"]
    )

    user_df["cos_theta"] = np.cos(
        user_df["theta"]
    )

    spatial_data.append(user_df)


# ==========================================
# 5. Fusionner
# ==========================================

data = pd.concat(
    spatial_data,
    ignore_index=True
)


# ==========================================
# 6. Normalisation latitude / longitude
# ==========================================

lat_mean = data["lat"].mean()
lat_std = data["lat"].std()

lon_mean = data["lon"].mean()
lon_std = data["lon"].std()

data["lat_norm"] = (
    (data["lat"] - lat_mean)
    / lat_std
)

data["lon_norm"] = (
    (data["lon"] - lon_mean)
    / lon_std
)


# ==========================================
# 7. Nettoyage
# ==========================================

features = [
    "lat_norm",
    "lon_norm",
    "delta_d_km",
    "delta_t_sec",
    "speed_ms",
    "sin_theta",
    "cos_theta"
]

data = data.dropna(
    subset=features
)


# Supprimer les valeurs infinies
data = data.replace(
    [np.inf, -np.inf],
    np.nan
)

data = data.dropna(
    subset=features
)


# ==========================================
# 8. Affichage
# ==========================================

print("\n========== CARACTÉRISTIQUES SPATIALES ==========")

print(
    data[
        [ 
            "user_id",
            "lat",
            "lon",
            "lat_norm",
            "lon_norm",
            "delta_d_km",
            "delta_t_sec",
            "speed_ms",
            "sin_theta",
            "cos_theta"
        ]
    ].head(10)
)


print("\nNombre de lignes après nettoyage :",
      len(data))


# ==========================================
# 9. Création des séquences
# ==========================================

SEQUENCE_LENGTH = 10

X = []
y = []
GPS = []

for i in range(len(data) - SEQUENCE_LENGTH):

    # Séquence des 7 caractéristiques spatiales
    X.append(
        data[features]
        .iloc[i:i + SEQUENCE_LENGTH]
        .values
    )

    # Coordonnées GPS originales
    GPS.append(
        data[["lat", "lon"]]
        .iloc[i:i + SEQUENCE_LENGTH]
        .values
    )

    # Position suivante à prédire
    y.append(
        data[["lat_norm", "lon_norm"]]
        .iloc[i + SEQUENCE_LENGTH]
        .values
    )

X = np.array(X, dtype=np.float32)
GPS = np.array(GPS, dtype=np.float32)
y = np.array(y, dtype=np.float32)


# ==========================================
# 10. Résultats
# ==========================================

print("\n========== SÉQUENCES ==========")

#print("X :", X.shape)

#print("y :", y.shape)


# ==========================================
# 11. Sauvegarde
# ==========================================

os.makedirs("data/processed", exist_ok=True)

np.save(
    "data/processed/X_spatial.npy",
    X
)

np.save(
    "data/processed/GPS_spatial.npy",
    GPS
)

np.save(
    "data/processed/y_spatial.npy",
    y
)

print("X :", X.shape)
print("GPS :", GPS.shape)
print("y :", y.shape)