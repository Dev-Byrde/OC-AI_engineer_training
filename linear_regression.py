#import des librairies nécéssaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#chargement du dataset
housing_data = pd.read_csv('house.csv')

#nettoyage du dataset
housing_data = housing_data[housing_data['loyer']<10000] #suppression des valeurs extrêmes

# On décompose le dataset et on le transforme en matrices pour pouvoir effectuer notre calcul
X = np.matrix([np.ones(housing_data.shape[0]), housing_data['surface'].values]).T
y = np.matrix(housing_data['loyer']).T

# On effectue le calcul exact du paramètre theta (demonstration: https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta)

#affichage du nuage de points

plt.xlabel('Surface')
plt.ylabel('Loyer')

plt.plot(housing_data['surface'], housing_data['loyer'], 'ro', markersize=4)

# On affiche la droite entre 0 et 250
plt.plot([0,250], [theta.item(0),theta.item(0) + 250 * theta.item(1)], linestyle='--', c='#000000')

plt.show()

