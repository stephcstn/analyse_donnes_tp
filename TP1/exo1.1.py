"""
Projet Clustering : K-Means & Classification Ascendante Hiérarchique (CAH)
Corpus synthétique de 256 individus issus de 2 lois normales 2D.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage as CAH
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Fixer la graine aléatoire pour la reproductibilité exacte
rng = np.random.default_rng(seed=42)

# ==============================================================================
# SECTION 1 : GÉNÉRATION DU CORPUS DE TEST (DONNÉES GAUSSIENNES)
# ==============================================================================

# Simulation Classe 1 : N=128, Centre (2, 2), Variance isotrope Var=2
mean1 = np.array([2.0, 2.0])
cov1 = np.array([[2.0, 0.0],
                 [0.0, 2.0]])
X1 = rng.multivariate_normal(mean1, cov1, size=128)

# Simulation Classe 2 : N=128, Centre (-4, -4), Variance isotrope Var=6
mean2 = np.array([-4.0, -4.0])
cov2 = np.array([[6.0, 0.0],
                 [0.0, 6.0]])
X2 = rng.multivariate_normal(mean2, cov2, size=128)

# Matrice globale (256 lignes x 2 colonnes) et vecteur de vérité terrain
data = np.concatenate((X1, X2), axis=0)
true_labels = np.concatenate((np.zeros(128, dtype=int), np.ones(128, dtype=int)))

# Fenêtre 1 : Nuage de points réel (Vérité terrain)
fig_truth = plt.figure("Figure 1 - Vérité Terrain", figsize=(7, 5))
plt.scatter(X1[:, 0], X1[:, 1], color='#1f77b4', marker='o', alpha=0.7, label='Classe 1 (Centre: 2, 2)')
plt.scatter(X2[:, 0], X2[:, 1], color='#d62728', marker='s', alpha=0.7, label='Classe 2 (Centre: -4, -4)')
plt.title("Vérité Terrain : 2 Classes Gaussiennes Réelles")
plt.xlabel("Mesure X1")
plt.ylabel("Mesure X2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# ==============================================================================
# SECTION 2 : ÉTUDE DE K-MEANS (K = 2 à 6)
# ==============================================================================

liste_k = [2, 3, 4, 5, 6]
inerties = []
silhouettes = []
scores_ari_km = []

# Fenêtre 2 : Partitions K-Means sur grille 2x3
fig_clusters, axes = plt.subplots(2, 3, figsize=(14, 8), num="Figure 2 - Partitions K-Means")
axes_flat = axes.flatten()

print("\n" + "=" * 65)
print("  PARTIE 1.2 : ÉVALUATION DES PERFORMANCES K-MEANS")
print("=" * 65)
print(f"{'K':<5} | {'Inertie':<12} | {'Score Silhouette':<18} | {'Score ARI':<10}")
print("-" * 65)

for i, k in enumerate(liste_k):
    # Entraînement de K-Means
    kmeans_k = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
    kmeans_k.fit(data)
    cluster_labels = kmeans_k.labels_

    # Métriques
    inertie = kmeans_k.inertia_
    sil_score = silhouette_score(data, cluster_labels, random_state=42)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    inerties.append(inertie)
    silhouettes.append(sil_score)
    scores_ari_km.append(ari)

    print(f"{k:<5} | {inertie:<12.2f} | {sil_score:<18.4f} | {ari:<10.4f}")

    # Affichage du sous-graphe
    ax = axes_flat[i]
    ax.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=25, alpha=0.7)
    ax.scatter(kmeans_k.cluster_centers_[:, 0], kmeans_k.cluster_centers_[:, 1],
               c='red', marker='X', s=120, edgecolors='black', label='Centroïdes')
    ax.set_title(f"K = {k} (ARI = {ari:.3f})")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True, linestyle='--', alpha=0.5)

# Suppression de la 6e sous-fenêtre vide
fig_clusters.delaxes(axes_flat[5])
fig_clusters.suptitle("Partitions obtenues pour différentes valeurs de K", fontsize=14)
fig_clusters.tight_layout()
print("=" * 65)

# Fenêtre 3 : Règle du coude et silhouette
fig_eval, (ax_inertie, ax_sil) = plt.subplots(1, 2, figsize=(12, 4), num="Figure 3 - Critères de sélection de K")

ax_inertie.plot(liste_k, inerties, marker='o', color='tab:blue', linewidth=2)
ax_inertie.set_title("Inertie intra-classe (Règle du Coude)")
ax_inertie.set_xlabel("Nombre de clusters K")
ax_inertie.set_ylabel("Inertie intra-classe")
ax_inertie.grid(True, linestyle='--', alpha=0.5)

ax_sil.plot(liste_k, silhouettes, marker='s', color='tab:red', linewidth=2)
ax_sil.set_title("Coefficient de Silhouette moyen")
ax_sil.set_xlabel("Nombre de clusters K")
ax_sil.set_ylabel("Score Silhouette")
ax_sil.grid(True, linestyle='--', alpha=0.5)

fig_eval.tight_layout()

# ==============================================================================
# SECTION 3 : CLUSTERING ASCENDANT HIÉRARCHIQUE (CAH)
# ==============================================================================

print("\n" + "=" * 65)
print("  PARTIE 1.3 : CLUSTERING HIÉRARCHIQUE ASCENDANT (CAH)")
print("=" * 65)

# 1. Lien complet et dendrogramme brut
Z_complete = CAH(data, method='complete', metric='euclidean')

fig_dendro_brut = plt.figure("Figure 4 - Dendrogramme Brut", figsize=(10, 5))
dendrogram(Z_complete, color_threshold=0)
plt.title("Dendrogramme Brut (Méthode du lien complet, color_threshold=0)")
plt.xlabel("Index des observations")
plt.ylabel("Hauteur de fusion (Distance euclidienne)")
plt.tight_layout()

# 2. Coupe automatique pour K = 3 clusters
# La coupe pour K branches se situe entre Z[-(K-1), 2] et Z[-K, 2]
seuil_k3 = (Z_complete[-2, 2] + Z_complete[-3, 2]) / 2
groupes_cah_k3 = fcluster(Z_complete, t=seuil_k3, criterion='distance')

print(f"Seuil de coupe automatique calculé pour K=3 : {seuil_k3:.3f}")

# Fenêtre 5 : Dendrogramme coupé à K=3
fig_dendro_k3 = plt.figure("Figure 5 - Dendrogramme Seuil K=3", figsize=(10, 5))
dendrogram(Z_complete, color_threshold=seuil_k3)
plt.axhline(y=seuil_k3, color='red', linestyle='--', linewidth=1.5,
            label=f'Seuil coupe K=3 : y = {seuil_k3:.2f}')
plt.title("Dendrogramme avec Seuil de Coupe pour K = 3 Clusters")
plt.xlabel("Index des observations")
plt.ylabel("Distance")
plt.legend()
plt.tight_layout()

# Fenêtre 6 : Nuage de points classé par CAH à K=3
fig_cah_pts = plt.figure("Figure 6 - Partition CAH K=3", figsize=(7, 5))
plt.scatter(data[:, 0], data[:, 1], c=groupes_cah_k3, cmap='Set1', s=35, alpha=0.8)
plt.title("Partition obtenue par la CAH pour K = 3 (Lien complet)")
plt.xlabel("Mesure X1")
plt.ylabel("Mesure X2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 3. Comparaison des 4 méthodes d'agrégation (pour K = 2)
methodes = ['single', 'complete', 'average', 'ward']
fig_comp_cah, axes_cah = plt.subplots(2, 2, figsize=(14, 9), num="Figure 7 - Comparaison des Méthodes CAH")
axes_cah_flat = axes_cah.flatten()

print("\n--- COMPARAISON DES MÉTHODES CAH (K = 2) ---")
print(f"{'Méthode':<15} | {'Score ARI vs Vérité':<20}")
print("-" * 40)

for i, methode in enumerate(methodes):
    Z = CAH(data, method=methode, metric='euclidean')
    labels_cah_k2 = fcluster(Z, t=2, criterion='maxclust')
    ari_cah = adjusted_rand_score(true_labels, labels_cah_k2)
    print(f"{methode.capitalize():<15} | {ari_cah:<20.4f}")

    ax = axes_cah_flat[i]
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-1, 2])
    ax.set_title(f"{methode.upper()} — ARI = {ari_cah:.3f}")
    ax.set_ylabel("Hauteur")

fig_comp_cah.suptitle("Dendrogrammes selon le critère d'agrégation (K = 2)", fontsize=14)
fig_comp_cah.tight_layout()
print("=" * 65 + "\n")

# ==============================================================================
# AFFICHAGE FINAL NON-BLOQUANT DE TOUTES LES FIGURES
# ==============================================================================
# Toutes les figures s'ouvrent en une seule fois sans interrompre l'exécution
plt.show()