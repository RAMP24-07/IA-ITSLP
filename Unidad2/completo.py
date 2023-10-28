import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
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
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


#############################################################################
## MUESTRA LOS DATOS DE ENTRENAMIENTO CON PERCEPTRON ##
print(50 * "=")
print("Sección: Entrenando un modelo de perceptrón en el conjunto de datos Iris")
print(50 * "-")
## AQUÍ VAN A TOMAR SUS DATOS DEL PROGRAMA ANTERIOR ##
# Cargar el conjunto de datos Titanic
df = pd.read_csv("Titanic-Dataset.csv")

# Eliminar las filas donde 'Age' tiene valores faltantes
df.dropna(subset=["Age"], inplace=True)

print(df.tail())

#############################################################################
print(50 * "=")
print("Graficando los datos titanic")
print(50 * "-")

# Seleccionar características y etiquetas
# Para este ejemplo, seleccionaremos 'Age' y 'Fare' como características
X = df[["Age", "Fare"]].values
y = df["Survived"].values

# plot data   --- CLASIFICA POR COLORES
# Graficar los datos del Titanic
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="o", label="No Sobrevivió")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="x", label="Sobrevivió")

plt.xlabel("Edad")
plt.ylabel("Tarifa")
plt.legend(loc="upper right")

plt.show()


#############################################################################
print(50 * "=")
print("Entrenando el modelo perceptron")
print(50 * "-")

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Numero de clasificaciones")


plt.show()

#############################################################################
print(50 * "=")
print("Una función para trazar regiones de decisión")
print(50 * "-")


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=cmap(idx),
            marker=markers[idx],
            label=cl,
        )


# Clasificar puntos en el espacio bidimensional
plot_decision_regions(X, y, classifier=ppn)

# Etiquetas y título
plt.xlabel("Edad")
plt.ylabel("Tarifa")
plt.legend(loc="upper right")

plt.show()


#############################################################################
## IMPLEMENTACIÓN DE ADAPTIVE ##
print(50 * "=")
print("Implementando una neurona lineal adaptable en Python")
print(50 * "-")


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


# Crear subplots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Entrenar Adaline con diferentes tasas de aprendizaje
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)

# Subplot 1: Adaline con learning rate 0.01
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker="o")
ax[0].set_xlabel("Épocas")
ax[0].set_ylabel("log(Sum-squared-error)")
ax[0].set_title("Adaline - Tasa de Aprendizaje 0.01")

# Subplot 2: Adaline con learning rate 0.0001
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker="o")
ax[1].set_xlabel("Épocas")
ax[1].set_ylabel("Sum-squared-error")
ax[1].set_title("Adaline - Tasa de Aprendizaje 0.0001")

# Ajustar el diseño de los subplots
plt.tight_layout()

# Mostrar la gráfica
plt.show()

# Estandarizar características
print("Estandarizar características")
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Entrenar Adaline en datos estandarizados
ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

# Visualizar regiones de decisión
plot_decision_regions(X_std, y, classifier=ada)

# Configuración de la gráfica
plt.title("Adaline - Gradient Descent")
plt.xlabel("Edad (estandarizada)")
plt.ylabel("Tarifa (estandarizada)")
plt.legend(loc="upper right")

# Mostrar la gráfica
plt.show()

# Graficar el costo en función de las épocas
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
plt.xlabel("Épocas")
plt.ylabel("Sum-squared-error")

# Mostrar la gráfica
plt.show()

#############################################################################
print(50 * "=")
print("Aprendizaje automático a gran escala y descenso de gradiente estocástico")
print(50 * "-")


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value averaged over all
        training samples in each epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)

# Entrenar Adaline en datos estandarizados
ada.fit(X_std, y)

# Visualizar regiones de decisión
plot_decision_regions(X_std, y, classifier=ada)

# Configuración de la gráfica
plt.title("Adaline - Stochastic Gradient Descent")
plt.xlabel("Edad (estandarizada)")
plt.ylabel("Tarifa (estandarizada)")
plt.legend(loc="upper right")

# Mostrar la gráfica
plt.show()

# Graficar el costo promedio en función de las épocas
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker="o")
plt.xlabel("Épocas")
plt.ylabel("Costo Promedio")

# Mostrar la gráfica
plt.show()

ada = ada.partial_fit(X_std[0, :], y[0])
