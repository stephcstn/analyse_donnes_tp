import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from affichage_acp import my_biplot

# Chargement et transposition (individus en lignes, variables en colonnes)
X = np.transpose(pd.read_csv("notes.csv", sep=";", index_col=0))

# Récupération des noms (attention : les commentaires de l'énoncé sont inversés)
nomi = list(X.index)    # noms des individus (les 9 élèves)
nomv = list(X.columns)  # noms des variables (les 5 matières)

# Nombre de variables (p = 5)
p = X.shape[1]

# Initialisation et calcul de l'ACP (sur données centrées uniquement)
acp = PCA(n_components=p, random_state=42)

cc = acp.fit_transform(X)

# Pourcentage d'inertie expliquée et inertie cumulée
inertie_pct = acp.explained_variance_ratio_ * 100
inertie_cum = np.cumsum(inertie_pct)
axes_num = np.arange(1, p + 1)

plt.figure(figsize=(8, 5))
plt.bar(axes_num, inertie_pct, alpha=0.6, color="steelblue", label="Inertie par axe (%)")
plt.plot(axes_num, inertie_cum, marker="o", color="crimson", linewidth=2, label="Inertie cumulée (%)")

# Affichage des pourcentages au-dessus des points
for i, txt in enumerate(inertie_cum):
    plt.annotate(f"{txt:.1f}%", (axes_num[i], inertie_cum[i] + 2), ha="center", fontsize=9, fontweight="bold")

plt.axhline(80, color="gray", linestyle="--", linewidth=1, label="Seuil 80%")
plt.title("Évolution de l'inertie expliquée (Scree-plot)")
plt.xlabel("Composantes principales (Axes)")
plt.ylabel("Pourcentage d'inertie (%)")
plt.xticks(axes_num)
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Partie 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fig.tight_layout()

# --- Plan E1 U E2 (Axes 1 et 2) ---
ax1.scatter(cc[:, 0], cc[:, 1], color="royalblue", s=60)
for i in range(len(nomi)):
    ax1.text(cc[i, 0] + 0.15, cc[i, 1] + 0.15, nomi[i], fontsize=10)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_title(f"Plan factoriel E1 U E2 ({inertie_pct[0]+inertie_pct[1]:.1f}% d'inertie)")
ax1.set_xlabel(f"Axe 1 ({inertie_pct[0]:.1f}%)")
ax1.set_ylabel(f"Axe 2 ({inertie_pct[1]:.1f}%)")
ax1.grid(True, linestyle=":", alpha=0.6)

# --- Plan E1 U E3 (Axes 1 et 3) ---
ax2.scatter(cc[:, 0], cc[:, 2], color="seagreen", s=60)
for i in range(len(nomi)):
    ax2.text(cc[i, 0] + 0.15, cc[i, 2] + 0.15, nomi[i], fontsize=10)
ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax2.set_title(f"Plan factoriel E1 U E3 ({inertie_pct[0]+inertie_pct[2]:.1f}% d'inertie)")
ax2.set_xlabel(f"Axe 1 ({inertie_pct[0]:.1f}%)")
ax2.set_ylabel(f"Axe 3 ({inertie_pct[2]:.1f}%)")
ax2.grid(True, linestyle=":", alpha=0.6)

# Partie 3

# Variances initiales de chaque matière (ddof=1 comme dans PCA de sklearn)
var_init = X.var(axis=0, ddof=1).values      # taille (p,)
lambdas = acp.explained_variance_            # taille (p,)

# Calcul de la matrice des corrélations (variables en lignes, axes en colonnes)
corr_var_axes = np.zeros((p, p))
for k in range(p):      # indice variable (matière)
    for j in range(p):  # indice composante (axe)
        corr_var_axes[k, j] = acp.components_[j, k] * np.sqrt(lambdas[j] / var_init[k])

df_corr = pd.DataFrame(corr_var_axes, index=nomv, columns=[f"Axe {j+1}" for j in range(p)])
print("Corrélations entre variables initiales et composantes principales :")
print(df_corr[["Axe 1", "Axe 2", "Axe 3"]].round(3))

# Partie 4

# Biplot dans le plan E1 U E2 (Axes 1 et 2)
my_biplot(
    score=cc[:, 0:2],
    coeff=np.transpose(acp.components_[0:2, :]),
    coeff_labels=nomv,
    score_labels=nomi,
    nomx="PC1",
    nomy="PC2",
    titre="Biplot dans le plan E1 U E2"
)

# Biplot dans le plan E1 U E3 (Axes 1 et 3)
my_biplot(
    score=cc[:, [0, 2]],
    coeff=np.transpose(acp.components_[[0, 2], :]),
    coeff_labels=nomv,
    score_labels=nomi,
    nomx="PC1",
    nomy="PC3",
    titre="Biplot dans le plan E1 U E3"
)

plt.show()