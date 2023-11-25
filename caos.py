import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x, iterations):
    results = []
    # Iterar a través del número especificado de generaciones
    for _ in range(iterations):
        # Aplicar la ecuación del mapa logístico: x_n+1 = r * x_n * (1 - x_n)
        x = r * x * (1 - x)
        # Guardar el resultado para esta iteración
        results.append(x)
    return results

# Configuración para la visualización del diagrama de bifurcación
# r_values: Crear una secuencia de valores de 'r' desde 2.5 hasta 4.0
r_values = np.linspace(2.5, 4.0, 1000)
# Número total de iteraciones para ejecutar el mapa logístico
iterations = 1000
# Número de puntos a mostrar para cada valor de 'r'
last = 100

# Inicializar figura para la visualización
plt.figure(figsize=(10, 6))

# Iterar a través de cada valor de 'r' en nuestra secuencia
for r in r_values:
    # Inicializar el valor de 'x' (población inicial) para este valor de 'r'
    x = 0.1
    # Obtener los resultados del mapa logístico para este valor de 'r'
    values = logistic_map(r, x, iterations)
    # Dibujar los últimos 'last' puntos en el diagrama de bifurcación
    plt.plot([r] * last, values[-last:], ',k', alpha=0.25)

# Añadir título y etiquetas a los ejes del gráfico
plt.title("Bifurcation diagram of the logistic map")
plt.xlabel("r")
plt.ylabel("Population")
# Mostrar el gráfico
plt.show()
