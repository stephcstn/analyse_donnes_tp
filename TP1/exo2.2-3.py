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

"""
Partie 3 : Réalisation algorithmique (K-Means "from scratch")
"""

import numpy as np
import matplotlib.pyplot as plt

def coalescence(x, K, g):
    """
    Implémente l'algorithme des centres mobiles (K-Means).
    
    Paramètres:
    x : numpy.ndarray, matrice (N, 2) contenant les individus
    K : int, nombre de classes
    g : numpy.ndarray, matrice (K, 2) des centroïdes initiaux
    
    Retourne:
    clas : numpy.ndarray, vecteur (N,) des indices de classe
    g2 : numpy.ndarray, matrice (K, 2) des centroïdes finaux
    """
    N = x.shape[0]
    g2 = np.copy(g)
    clas = np.zeros(N, dtype=int)
    
    while True:
        distances = np.zeros((N, K))
        
        # 1. Étape d'affectation : calcul de d(x1, x2) = (x1-x2)^T (x1-x2)
        for i in range(K):
            # Différence vectorielle entre chaque individu de x et le centre g2[i]
            diff = x - g2[i, :]
            # np.sum(diff**2) par ligne équivaut au produit scalaire du vecteur par lui-même
            distances[:, i] = np.sum(diff**2, axis=1) 
            
        new_clas = np.argmin(distances, axis=1)
        
        # Condition d'arrêt : les individus ne changent plus de classe
        if np.array_equal(clas, new_clas):
            break
            
        clas = np.copy(new_clas)
        
        # 2. Étape de mise à jour des centres de gravité
        for i in range(K):
            # Extraction des individus appartenant à la classe i
            points_classe = x[clas == i]
            # Sécurité anti-division par zéro si un cluster se vide
            if len(points_classe) > 0:
                g2[i, :] = np.mean(points_classe, axis=0)
                
    return clas, g2

# ==============================================================================
# SCRIPT DE TEST (Question 2, 3 et 4)
# ==============================================================================

# 1. Génération du corpus (Reprise de la partie 1)
rng = np.random.default_rng()

mean1 = np.array([2.0, 2.0])
cov1 = np.array([[2.0, 0.0], [0.0, 2.0]])
X1 = rng.multivariate_normal(mean1, cov1, size=128)

mean2 = np.array([-4.0, -4.0])
cov2 = np.array([[6.0, 0.0], [0.0, 6.0]])
X2 = rng.multivariate_normal(mean2, cov2, size=128)

x_data = np.concatenate((X1, X2), axis=0)
K_test = 2

# Exécution du test plusieurs fois pour observer la sensibilité à l'initialisation
nb_tests = 3
fig, axes = plt.subplots(1, nb_tests, figsize=(15, 5))
fig.suptitle("Test de l'algorithme de coalescence (K=2) avec différentes initialisations aléatoires", fontsize=14)

for t in range(nb_tests):
    # 2. Choix aléatoire de K individus pour les centres initiaux
    indices_init = rng.choice(x_data.shape[0], size=K_test, replace=False)
    g_initial = x_data[indices_init, :]
    
    # Appel de la fonction de coalescence
    classes, centres_finaux = coalescence(x_data, K_test, g_initial)
    
    # 3. Affichage graphique de la partition
    ax = axes[t]
    ax.scatter(x_data[:, 0], x_data[:, 1], c=classes, cmap='viridis', alpha=0.7, s=30)
    
    # Affichage des centres initiaux (Croix noires) et finaux (Croix rouges)
    ax.scatter(g_initial[:, 0], g_initial[:, 1], c='black', marker='X', s=100, label='Centres initiaux')
    ax.scatter(centres_finaux[:, 0], centres_finaux[:, 1], c='red', marker='X', s=100, label='Centres finaux')
    
    ax.set_title(f"Test n°{t+1}")
    ax.grid(True, linestyle='--', alpha=0.5)
    if t == 0:
        ax.legend()

plt.tight_layout()
plt.show()