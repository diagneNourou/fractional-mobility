import pandas as pd
import glob
import os


# Chemin vers les fichiers .plt
path = "data/raw/Geolife Trajectories 1.3/Data/*/Trajectory/*.plt"

files = glob.glob(path)

print("Nombre de fichiers trouvés :", len(files))


def read_plt(file):

    df = pd.read_csv(
        file,
        skiprows=6,
        header=None,
        names=[
            'lat',
            'lon',
            'zero',
            'alt',
            'days',
            'date',
            'time'
        ]
    )

    # Créer la colonne datetime
    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time']
    )

    # Garder uniquement les colonnes utiles
    return df[['lat', 'lon', 'datetime']]


# Lire les fichiers
dfs = []

for file in files[:20]:

    try:

        df = read_plt(file)

        # Récupérer l'identifiant utilisateur
        # Structure :
        # Data / utilisateur / Trajectory / fichier.plt
        parts = file.replace("\\", "/").split("/")

        user_id = parts[-3]

        # Ajouter l'identifiant utilisateur
        df['user_id'] = user_id

        dfs.append(df)

        print(
            "Fichier chargé :",
            os.path.basename(file),
            "| utilisateur :",
            user_id,
            "| lignes :",
            len(df)
        )

    except Exception as e:

        print(
            "Erreur dans",
            file,
            ":",
            e
        )


# Vérifier qu'on a bien récupéré des données
if not dfs:

    print("Aucune donnée chargée.")
    exit()


# Fusionner les données
data = pd.concat(
    dfs,
    ignore_index=True
)


# Trier les données
data = data.sort_values(
    ['user_id', 'datetime']
).reset_index(drop=True)


# Vérifications
print("\n========== APERÇU ==========")

print(data.head())

print("\n========== INFORMATIONS ==========")

print(data.info())

print("\n========== VALEURS MANQUANTES ==========")

print(data.isnull().sum())

print("\n========== STATISTIQUES ==========")

print(data[['lat', 'lon']].describe())

print("\nNombre total de lignes :", len(data))

print(
    "Nombre d'utilisateurs :",
    data['user_id'].nunique()
)


# Créer le dossier de sortie
os.makedirs(
    "data/processed",
    exist_ok=True
)


# Sauvegarder
output_file = "data/processed/geolife.csv"

data.to_csv(
    output_file,
    index=False
)


print("\nDataset sauvegardé !")
print("Fichier :", output_file)