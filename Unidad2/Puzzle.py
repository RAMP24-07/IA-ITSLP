import numpy as np
import heapq

# Define el estado objetivo y los movimientos posibles
estado_objetivo = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
movimientos_posibles = [(0, 1), (1, 0), (0, -1), (-1, 0)]


# Define una función para calcular la distancia de Manhattan
def distancia_manhattan(estado):
    distancia = 0
    for i in range(3):
        for j in range(3):
            if estado[i, j] != 0:
                fila_objetivo, col_objetivo = divmod(estado[i, j], 3)
                distancia += abs(i - fila_objetivo) + abs(j - col_objetivo)
    return distancia


# Define la clase Nodo para representar el estado del rompecabezas
class Nodo:
    def __init__(self, estado, padre=None, movimiento=(0, 0)):
        self.estado = estado
        self.padre = padre
        self.movimiento = movimiento
        self.profundidad = 0 if padre is None else padre.profundidad + 1
        self.costo = self.profundidad + distancia_manhattan(self.estado)

    def __lt__(self, otro):
        return self.costo < otro.costo


# Define una función para recuperar el camino desde el estado inicial hasta el estado objetivo
def recuperar_camino(nodo):
    camino = []
    while nodo is not None:
        camino.append((nodo.estado, nodo.movimiento, nodo.costo))
        nodo = nodo.padre
    return list(reversed(camino))


# Define el algoritmo A* para resolver el rompecabezas
def astar(estado_inicial, estado_objetivo):
    lista_abierta = []
    conjunto_cerrado = set()

    nodo_inicial = Nodo(estado_inicial)
    heapq.heappush(lista_abierta, nodo_inicial)

    while lista_abierta:
        nodo_actual = heapq.heappop(lista_abierta)

        if np.array_equal(nodo_actual.estado, estado_objetivo):
            return recuperar_camino(nodo_actual)

        conjunto_cerrado.add(tuple(nodo_actual.estado.flatten()))

        for movimiento in movimientos_posibles:
            nuevo_estado = nodo_actual.estado.copy()
            fila_vacia, col_vacia = np.where(nuevo_estado == 0)
            nueva_fila, nueva_col = (
                fila_vacia + movimiento[0],
                col_vacia + movimiento[1],
            )

            if 0 <= nueva_fila < 3 and 0 <= nueva_col < 3:
                (
                    nuevo_estado[fila_vacia, col_vacia],
                    nuevo_estado[nueva_fila, nueva_col],
                ) = (nuevo_estado[nueva_fila, nueva_col], 0)

                if tuple(nuevo_estado.flatten()) not in conjunto_cerrado:
                    nuevo_nodo = Nodo(nuevo_estado, nodo_actual, movimiento)
                    heapq.heappush(lista_abierta, nuevo_nodo)


# Define el estado inicial y llama al solucionador A*
estado_inicial = np.array([[7, 2, 4], [5, 0, 6], [8, 3, 1]])
camino = astar(estado_inicial, estado_objetivo)

# Imprime los pasos para llegar al estado objetivo
for paso, (estado, movimiento, costo) in enumerate(camino):
    print(f"Paso {paso}:")
    print(estado)
    print(f"Movimiento: {movimiento}")
    print(f"Costo: {costo}\n")
