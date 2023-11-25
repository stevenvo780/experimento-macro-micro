import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constantes
G = 6.67430e-11
dt = 1000  # Paso de tiempo en segundos
num_steps = 1000  # Número de pasos en la simulación
num_bodies = 50  # Número de cuerpos

# Inicialización de los cuerpos celestes
masses = cp.random.uniform(1e28, 1e29, num_bodies)
positions = cp.random.uniform(-1e11, 1e11, (num_bodies, 2))
velocities = cp.random.uniform(-3e4, 3e4, (num_bodies, 2))

def compute_gravitational_force(positions, masses):
    force_matrix = cp.zeros((num_bodies, num_bodies, 2))
    for i in range(num_bodies):
        for j in range(i + 1, num_bodies):
            r = positions[j] - positions[i]
            distance = cp.linalg.norm(r)
            if distance > 0:
                force_magnitude = G * masses[i] * masses[j] / distance**2
                force_direction = r / distance
                force_matrix[i, j] = force_magnitude * force_direction
                force_matrix[j, i] = -force_matrix[i, j]
    return cp.sum(force_matrix, axis=1)

def update_simulation(positions, velocities, masses):
    states = []
    for _ in range(num_steps):
        forces = compute_gravitational_force(positions, masses)
        velocities += forces / masses[:, None] * dt
        positions += velocities * dt
        states.append(cp.asnumpy(positions))
    return states

states = update_simulation(positions, velocities, masses)

# Función de actualización para la animación
def update_visualization(frame, states, points):
    for point, position in zip(points, states[frame]):
        point.set_data(position[0], position[1])
    return points

# Configuración para la animación
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)
points = [ax.plot([], [], 'o', ms=3)[0] for _ in range(num_bodies)]

# Crear la animación
ani = FuncAnimation(fig, update_visualization, frames=num_steps, fargs=(states, points), blit=True, interval=20)

plt.show()
