import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Chargement et transposition (individus en lignes, variables en colonnes)
X = np.transpose(pd.read_csv("notes.csv", sep=";", index_col=0))

# Récupération des noms (attention : les commentaires de l'énoncé sont inversés)
nomi = list(X.index)    # noms des individus (les 9 élèves)
nomv = list(X.columns)  # noms des variables (les 5 matières)

print(X)

# Tracé de l'histogramme pour la matière "français"
plt.figure(figsize=(7, 4))
plt.hist(X["fran"], bins=5, range=(0, 20), edgecolor="black", color="skyblue")

# Habillage du graphique
plt.title("Distribution des notes en français")
plt.xlabel("Notes (/20)")
plt.ylabel("Nombre d'élèves")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Tracé de l'histogramme pour la matière "latin"
plt.figure(figsize=(7, 4))
plt.hist(X["lati"], bins=5, range=(0, 20), edgecolor="black", color="lightcoral")

# Habillage du graphique
plt.title("Distribution des notes en latin")
plt.xlabel("Notes (/20)")
plt.ylabel("Nombre d'élèves")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Extraction des deux colonnes sous forme de tableau NumPy (n x 2)
Data = X[["math", "scie"]].values
n = len(nomi)  # Nombre d'individus (n = 9)

# Création d'une 3e figure dédiée au nuage de points
plt.figure(figsize=(7, 5))

# Tracé point par point avec annotation du nom de l'élève
for i in range(0, n):
    plt.scatter(Data[i, 0], Data[i, 1], color="steelblue")
    plt.text(Data[i, 0] + 0.15, Data[i, 1] + 0.15, nomi[i], fontsize=9)

# Habillage du graphique
plt.title("Nuage de points : Mathématiques vs Sciences")
plt.xlabel("Mathématiques (/20)")
plt.ylabel("Sciences (/20)")
plt.xlim(0, 21)
plt.ylim(0, 21)
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()