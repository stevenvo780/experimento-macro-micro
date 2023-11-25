import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_particles(num_particles):
    return np.zeros((num_particles, 2)), np.zeros((num_particles, 2))

def random_step(particles, velocities, center_force_strength, repulsion_strength, inertia):
    # Aplicamos un paso aleatorio con mayor varianza
    velocities += np.random.normal(0, 0.5, particles.shape)
    
    # Aplicar fuerza central
    center_vector = -particles
    distances = np.linalg.norm(center_vector, axis=1, keepdims=True)
    # Evitamos la división por cero
    distances[distances == 0] = np.inf
    center_vector /= distances  # Normalizar vectores
    center_force = center_vector * center_force_strength
    velocities += center_force

    # Aplicar fuerza de repulsión entre partículas
    for i, particle in enumerate(particles):
        diff = particle - particles  # Vector de diferencia entre partículas
        distance = np.linalg.norm(diff, axis=1)  # Distancia entre partículas
        close_particles = (distance < 5) & (distance > 0)  # Partículas cercanas
        repulsion = np.sum(-diff[close_particles] / distance[close_particles, np.newaxis]**2, axis=0)
        velocities[i] += repulsion_strength * repulsion

    # Aplicamos inercia a las velocidades
    velocities *= inertia
    # Actualizar posiciones
    particles += velocities

def animate(i):
    global particles, velocities
    random_step(particles, velocities, center_force_strength, repulsion_strength, inertia)
    scatter.set_offsets(particles)

# Parámetros de la simulación
num_particles = 300
center_force_strength = 0.05
repulsion_strength = 0.05
inertia = 0.5

# Inicialización
particles, velocities = initialize_particles(num_particles)

# Crear figura para la animación
fig, ax = plt.subplots()
scatter = ax.scatter(particles[:, 0], particles[:, 1])
ax.axis('equal')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_xticks([])
ax.set_yticks([])

# Crear y mostrar animación
ani = FuncAnimation(fig, animate, frames=200, interval=50)
plt.show()
