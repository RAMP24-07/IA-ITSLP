# -*- coding: utf-8 -*-
"""Practica1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1axP8YXE8ahZ9Tswqx4Vox5enGhxsTO_Z

# Ejecución 1
"""

def suma (a, b):
  resultado = a + b
  return resultado

print(suma(2, 5))

"""# Práctica 1 - Parte 1"""

import tensorflow as tf
import numpy as np

celcius = np.array([-15, -5, 0, 5, 15], dtype=float)
fahrenheit = np.array([5, 23, 32, 41, 59], dtype=float)

#capa = tf.keras.layers.Dense(units=1, input_shape=[1])
#modelo = tf.keras.Sequential([capa])
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial=modelo.fit(celcius, fahrenheit, epochs=100, verbose=False)
print("Modelo entrenado!!!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Realizar una predicción!!!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + "°F")

modelo.save('celsius_a_fahrenheit.h5')

!ls

!pip install tensorflowjs

!mkdir temperatura

!tensorflowjs_converter --input_format keras celsius_a_fahrenheit.h5 temperatura

!ls temperatura

"""# Práctica 1 - Parte 2
Implementar la práctica anterior reciclar el código, para cambiar el tipo de predicción donde realice la conversión diferente (km a m, gal a litros, etc)
"""

import tensorflow as tf
import numpy as np

km = np.array([-4, 1, 2, 5, 7, 8.5, 9, 11], dtype=float)
m = np.array([-4000, 1000, 2000, 5000, 7000, 8500, 9000, 11000], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial=modelo.fit(km, m, epochs=1000, verbose=False)
print("Modelo entrenado!!!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Realizar una predicción!!!")
resultado = modelo.predict([3])
print("El resultado es " + str(resultado[0][0]) + "m")

modelo.save('kilometros_a_metros.h5')

!mkdir distancia

!tensorflowjs_converter --input_format keras kilometros_a_metros.h5 distancia