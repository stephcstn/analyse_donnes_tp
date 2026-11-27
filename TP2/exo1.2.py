import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Chargement et transposition (individus en lignes, variables en colonnes)
X = np.transpose(pd.read_csv("notes.csv", sep=";", index_col=0))

# Récupération des noms (attention : les commentaires de l'énoncé sont inversés)
nomi = list(X.index)    # noms des individus (les 9 élèves)
nomv = list(X.columns)  # noms des variables (les 5 matières)

# Nombre de variables (p = 5)
p = X.shape[1]

# Initialisation et calcul de l'ACP (sur données centrées uniquement)
acp = PCA(n_components=p, random_state=42)
cc = acp.fit_transform(X)  # Coordonnées des 9 élèves sur les 5 axes principaux

# Affichage des valeurs propres (inertie de chaque axe)
print("Valeurs propres (variance expliquée) :", acp.explained_variance_.round(2))
print("Pourcentage d'inertie expliquée :", (acp.explained_variance_ratio_ * 100).round(2))

# Affichage des axes principaux (vecteurs propres en lignes)
axes = pd.DataFrame(acp.components_, columns=nomv, index=[f"Axe {i+1}" for i in range(p)])
print("\nVecteurs propres (axes principaux) :")
print(axes.round(3))