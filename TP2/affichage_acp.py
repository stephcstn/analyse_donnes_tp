import matplotlib.pyplot as plt
import numpy as np

def my_biplot(score=None, coeff=None, coeff_labels=None, score_labels=None, nomx=None, nomy=None, titre=None):
    xs = score[:, 0]
    ys = score[:, 1]

    if (len(xs) != len(ys)):
        print("Warning ! len(x) != len(y)")

    x_c = (xs - xs.mean()) / (xs.max() - xs.min())
    y_c = (ys - ys.mean()) / (ys.max() - ys.min())

    fig = plt.figure(figsize=(6, 6), facecolor='w')
    ax = fig.add_subplot(111)

    # Affichage des points (individus)
    ax.scatter(x_c, y_c)
    if score_labels is not None:
        n = len(x_c)
        for i in range(0, n):
            plt.text(x_c[i] + 0.01, y_c[i], score_labels[i])

    # Cercle unité (suppression du second plt.subplots() qui séparait les graphiques)
    x_circle = np.linspace(-1, 1, 100)
    y_circle = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_circle, y_circle)
    F = X ** 2 + Y ** 2 - 1.0
    plt.contour(X, Y, F, [0])

    # Affichage des flèches (variables)
    p = coeff.shape[0]
    for i in range(0, p):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5, head_width=0.05, head_length=0.05)
        label = "Var" + str(i + 1) if coeff_labels is None else coeff_labels[i]
        plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, label, color='g', ha='center', va='center')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel(nomx)
    plt.ylabel(nomy)
    if titre is not None:
        plt.title(titre)
    plt.grid(linestyle='--')