import pandas as pd
import glob
import os

# chemin vers les fichiers .plt
path = "data/raw/Geolife Trajectories 1.3/Data/*/Trajectory/*.plt"

files = glob.glob(path)

print("Nombre de fichiers trouvés :", len(files))


def read_plt(file):
    df = pd.read_csv(
        file,
        skiprows=6,  # ignorer les 6 premières lignes
        header=None,
        names=['lat', 'lon', 'alt', 'days', 'date', 'time']
    )
    # Voir nombre de lignes
    print(df.info())

    # Types de donnees
    print(df.describe())

    #Verifier pas de valeurs nulles et coherence des donnees
    print(df.head())
    print(df.isnull().sum())
   
    # créer colonne datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    return df[['lat', 'lon', 'datetime']]


# lire quelques fichiers (test)
dfs = []

for file in files[:20]:  # commence petit !
    try:
        df = read_plt(file)
        dfs.append(df)
    except Exception as e:
        print("Erreur :", e)

# fusionner
data = pd.concat(dfs, ignore_index=True)

print(data.head())
print("Nombre de lignes :", len(data))
os.makedirs("data/processed", exist_ok=True)

data.to_csv("data/processed/geolife.csv", index=False)

#Geolife contient plusieurs utilisateurs 
df['user_id'] = 0
print("Dataset sauvegardé !")
print("OK")
