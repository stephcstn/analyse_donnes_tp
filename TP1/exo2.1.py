"""
Partie 2.1 : Utilisation de la méthode K-Means pour la réduction de couleur
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

img = np.float32(mpimg.imread('visage.bmp'))

# normalement 256x256 pixels, 3 canaux de couleur
h, w, c = img.shape
print(f"Dimensions de l'image originale : {h}x{w} avec {c} canaux de couleur.")

# (h * w) lignes et 3 colonnes
data_img = np.reshape(img, (h * w, c))
valeurs_K = [2, 4, 16, 64]

plt.figure("Réduction de couleurs par K-Means", figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img / 255.0)
plt.title("Image originale (16 millions de couleurs)")
plt.axis('off')

for i, K in enumerate(valeurs_K):
    print(f"Traitement en cours pour K = {K} couleurs...")
    
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
    kmeans.fit(data_img)
    
    # codage decodage : 
    # On remplace chaque label de pixel par la valeur (R,G,B) de son centroïde
    # L'indexation avancée de NumPy permet de faire cela sans boucle 'for'
    img_codee_1d = kmeans.cluster_centers_[kmeans.labels_]
    
    # 5. Ré-organiser en tableau 3D (h x w x 3) pour l'affichage
    img_compressee_3d = np.reshape(img_codee_1d, (h, w, c))
    
    # Ajout du sous-graphique
    plt.subplot(2, 3, i + 2)
    plt.imshow(img_compressee_3d / 255.0)
    plt.title(f"K = {K} couleurs")
    plt.axis('off')

print("Génération terminée. Affichage des résultats...")
plt.tight_layout()
plt.show()