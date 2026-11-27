import matplotlib.pyplot as plt
import numpy as np

def my_biplot(score=None, coeff=None, coeff_labels=None, score_labels=None, nomx=None, nomy=None, titre=None, ax=None):
    xs = score[:, 0]
    ys = score[:, 1]

    if (len(xs) != len(ys)):
        print("Warning ! len(x) != len(y)")

    x_c = (xs - xs.mean()) / (xs.max() - xs.min())
    y_c = (ys - ys.mean()) / (ys.max() - ys.min())

    # Crée une nouvelle figure uniquement si aucun axe 'ax' n'est fourni
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='w')

    # Affichage des points (individus)
    ax.scatter(x_c, y_c)
    if score_labels is not None:
        for i in range(len(x_c)):
            ax.text(x_c[i] + 0.02, y_c[i], score_labels[i], fontsize=9)

    # Cercle unité
    x_circle = np.linspace(-1, 1, 100)
    y_circle = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_circle, y_circle)
    F = X ** 2 + Y ** 2 - 1.0
    ax.contour(X, Y, F, [0], colors='purple')

    # Affichage des flèches (variables)
    p = coeff.shape[0]
    for i in range(p):
        ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5, head_width=0.05, head_length=0.05)
        label = "Var" + str(i + 1) if coeff_labels is None else coeff_labels[i]
        ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, label, color='g', ha='center', va='center', fontsize=9)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(nomx)
    ax.set_ylabel(nomy)
    ax.set_aspect('equal')  # Garantit que le cercle unité reste bien rond
    if titre is not None:
        ax.set_title(titre)
    ax.grid(linestyle='--')