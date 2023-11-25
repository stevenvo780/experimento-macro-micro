import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_particles(num_particles):
    return np.zeros((num_particles, 2))

def random_step(particles):
    particles += np.random.normal(0, 1, particles.shape)
    return particles

def animate(i):
    random_step(particles)
    scat.set_offsets(particles)

# Parámetros de la simulación
num_particles = 1000
steps = 200  # Total de pasos para la animación

# Inicializar partículas
particles = initialize_particles(num_particles)

# Configurar la gráfica para la animación
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
scat = ax.scatter(particles[:, 0], particles[:, 1], alpha=0.5)

# Crear y ejecutar la animación
ani = FuncAnimation(fig, animate, frames=steps, interval=50)
plt.show()
