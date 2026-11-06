import numpy as np
import matplotlib.pyplot as plt

# Simulation de n=128 données X=(X1, X2) suivant une loi normale
# bidimensionnelle de moyenne (2.0, 2.0), de variance Var(X1)=2.0, Var(X2)=2.0
# et de covariance Cov(X1, X2)=0
mean1 = np.array([2.0, 2.0])
cov1 = np.array([[2.0, 0], [0, 2.0]])
X1 = np.random.multivariate_normal(mean1, cov1, 128)

# Simulation de n=128 données X=(X1, X2) suivant une loi normale
# bidimensionnelle de moyenne (-4.0, -4.0), de variance Var(X1)=6.0, Var(X2)=6.0
# et de covariance Cov(X1, X2)=0
mean2 = np.array([-4.0, -4.0])
cov2 = np.array([[6.0, 0], [0, 6.0]])
X2 = np.random.multivariate_normal(mean2, cov2, 128)

# ensemble de test de taille 256 individus
data = np.concatenate((X1, X2), axis=0)

# Affichage des données
plt.scatter(X1[:, 0], X1[:, 1], color='blue', marker='o', label='Classe 1 (Centre 2, 2)')
plt.scatter(X2[:, 0], X2[:, 1], color='red', marker='s', label='Classe 2 (Centre -4, -4)')
plt.legend()
plt.show()