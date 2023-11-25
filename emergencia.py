import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Inicialización de partículas
def initialize_particles(num_particles, space_size):
    positions = np.random.rand(num_particles, 2) * space_size
    angles = 2 * np.pi * np.random.rand(num_particles)
    velocities = np.column_stack((np.cos(angles), np.sin(angles)))
    return positions, velocities

# Actualización de las partículas con reglas de interacción
def update_particles(positions, velocities, space_size):
    # Parámetros del modelo
    align_strength = 0.5
    repulse_strength = 10.0
    attract_strength = 0.1
    noise_strength = 0.1

    # Alineación y repulsión
    for i, position in enumerate(positions):
        distances = np.linalg.norm(positions - position, axis=1)
        direction = np.mean(velocities[distances < 1.0], axis=0)
        repulsion = np.sum((position - positions[distances < 0.5]) / (distances[distances < 0.5, np.newaxis] + 0.01), axis=0)
        velocities[i] += align_strength * direction - repulse_strength * repulsion
        velocities[i] /= np.linalg.norm(velocities[i])  # Normalizar la velocidad

    # Atracción a largo plazo
    center_of_mass = np.mean(positions, axis=0)
    direction_to_com = center_of_mass - positions
    velocities += attract_strength * direction_to_com

    # Ruido aleatorio
    noise = noise_strength * (np.random.rand(num_particles, 2) - 0.5)
    velocities += noise

    # Actualizar posiciones
    positions += velocities
    # Condiciones de frontera periódicas
    positions %= space_size

    return positions, velocities

# Función de animación
def animate(frame_num, positions, velocities, scatter, space_size):
    new_positions, new_velocities = update_particles(positions, velocities, space_size)
    scatter.set_offsets(new_positions)
    return scatter,

# Configuración inicial
num_particles = 100
space_size = 10
positions, velocities = initialize_particles(num_particles, space_size)

# Configuración de la figura
fig, ax = plt.subplots()
scatter = ax.scatter(positions[:, 0], positions[:, 1])
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.set_xticks([])
ax.set_yticks([])

# Crear y ejecutar la animación
ani = FuncAnimation(fig, animate, fargs=(positions, velocities, scatter, space_size), frames=200, interval=1000, blit=True)
plt.show()
