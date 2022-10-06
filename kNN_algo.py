from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#On charge le dataset MNIST (Chiffres Manuscrits) depuis SKLEARN Datasets
mnist= fetch_openml('mnist_784', version=1)

#On affiche les données de ce dataset
print(mnist.data.shape)
print(mnist.target.shape)

#on echantillonne les données pour limiter le nombre d'entrée 
sample = np.random.randint(70000, size=5000) #on prend 5.000 chiffre aléatoirement entre 0 et 70.000
data = mnist.data.values[sample] #On selectionne les entrée de 'data' qui corresponde aux nombre dans sample
target = mnist.target.values[sample] #idem pour 'target'

#On divise notre dataset en 2 sets un de training, un de test (ration 80/20 pour training)
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8) 

#On applique la méthode des kNN (K-nearest neighbors) pour "3 voisins" grâce à un fonction neighbors de Scikit
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)

#On test l'erreur sur les set de test avec la fontion score
error = 1 - knn.score(xtest,ytest)
print('Erreur:%f'% error)

#On cherche maintenant à optimiser le taux d'erreur du test on recalcul l'erreur pour différent K
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

#On affiche les resultat pour quelques données
# On récupère le classifieur le plus performant
knn = neighbors.KNeighborsClassifier(4)
knn.fit(xtrain, ytrain)

# On récupère les prédictions sur les données test
predicted = knn.predict(xtest)

# On redimensionne les données sous forme d'images
images = xtest.reshape((-1, 28, 28))

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)

# On affiche les images avec la prédiction associée
fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()
