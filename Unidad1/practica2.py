import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Cargar el conjunto de datos Titanic
df = pd.read_csv("Titanic-Dataset.csv")

# Eliminar las filas donde 'Age' tiene valores faltantes
df.dropna(subset=["Age"], inplace=True)

# Seleccionar características y etiquetas
# Para este ejemplo, seleccionaremos 'Age' y 'Fare' como características
X = df[["Age", "Fare"]].values
y = df["Survived"].values


# Definición de la clase Perceptron
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Entrenar el perceptrón:
ppn = Perceptron(eta=0.1, n_iter=6)
ppn.fit(X, y)

# Graficar el número de actualizaciones de clasificación errónea
# en función de los Epochs
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Numero de errores")
plt.show()
