"""
Partie 2.2 & 2.3 : Clustering de données de températures (CAH et K-Means)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# ==============================================================================
# PRÉPARATION DES DONNÉES
# ==============================================================================

print("Chargement des données...")
data_temperature = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)

# extraction (Latitude = Y, Longitude = X)
Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values
nom_ville = list(data_temperature.index)

# Suppression des colonnes non pertinentes pour le clustering (on ne garde que les 12 mois)
data = data_temperature.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])

# Normalisation des données (centrage-réduction) : crucial pour que les variations 
# de températures ne soient pas écrasées par des échelles différentes.
data_scaled = scale(data)

# ==============================================================================
# SECTION 2.2 : CLASSIFICATION ASCENDANTE HIÉRARCHIQUE (CAH)
# ==============================================================================

print("\n--- Analyse CAH ---")
# Calcul de la matrice de dissimilarité et construction de l'arbre (méthode de Ward)
Z = linkage(data_scaled, method='ward', metric='euclidean')

# Affichage du dendrogramme pour choisir le seuil
plt.figure("CAH - Dendrogramme", figsize=(12, 6))
plt.title("Dendrogramme des températures (Méthode de Ward)")
# On affiche les noms des villes en bas
dendro = dendrogram(Z, labels=nom_ville, leaf_rotation=90, leaf_font_size=8)

# On choisit arbitrairement une coupe pour obtenir 4 classes (le coude sur l'axe Y)
seuil_cah = 10 
plt.axhline(y=seuil_cah, color='r', linestyle='--', label=f'Seuil de coupe = {seuil_cah}')
plt.legend()
plt.tight_layout()

# Extraction des groupes
groupes_cah = fcluster(Z, t=seuil_cah, criterion='distance')
nb_classes_cah = len(np.unique(groupes_cah))
print(f"CAH : Découpage au seuil {seuil_cah} -> {nb_classes_cah} classes obtenues.")

# Affichage géographique des classes CAH
plt.figure("CAH - Carte", figsize=(8, 8))
plt.scatter(Coord[:, 1], Coord[:, 0], c=groupes_cah, s=50, cmap='Set1', edgecolors='k')
for i, txt in enumerate(nom_ville):
    plt.annotate(txt, (Coord[i, 1] + 0.1, Coord[i, 0] + 0.1), fontsize=8)
plt.title(f"Répartition géographique - CAH ({nb_classes_cah} classes)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()


# ==============================================================================
# SECTION 2.3 : MÉTHODE DES K-MEANS
# ==============================================================================

print("\n--- Analyse K-Means ---")
# Règle du coude : on teste K de 2 à 10
liste_k = range(2, 11)
inerties = []

for k in liste_k:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    inerties.append(kmeans.inertia_)

# Affichage de la courbe d'inertie
plt.figure("K-Means - Règle du coude", figsize=(8, 5))
plt.plot(liste_k, inerties, marker='o', color='b')
plt.title("Règle du coude pour K-Means")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie intra-classe")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Choix du K optimal (par exemple K=4 au vu d'un changement de pente supposé)
k_optimal = 4
kmeans_opt = KMeans(n_clusters=k_optimal, n_init=10, random_state=42)
kmeans_opt.fit(data_scaled)
groupes_kmeans = kmeans_opt.labels_

# Affichage géographique des classes K-Means
plt.figure("K-Means - Carte", figsize=(8, 8))
plt.scatter(Coord[:, 1], Coord[:, 0], c=groupes_kmeans, s=50, cmap='Set1', edgecolors='k')
for i, txt in enumerate(nom_ville):
    plt.annotate(txt, (Coord[i, 1] + 0.1, Coord[i, 0] + 0.1), fontsize=8)
plt.title(f"Répartition géographique - K-Means (K = {k_optimal})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

print(f"K-Means : Modèle entraîné pour K = {k_optimal}.")
print("Affichage des 4 figures (fermer les fenêtres pour terminer le script)...")

# Affichage de tous les graphiques
plt.show()