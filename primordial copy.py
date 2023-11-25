import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Inicializar las partículas en posiciones y direcciones aleatorias
def initialize_particles(num_particles):
    positions = cp.random.rand(num_particles, 2) * space_size
    angles = 2 * cp.pi * cp.random.rand(num_particles)
    velocities = cp.column_stack((cp.cos(angles), cp.sin(angles)))
    return positions, velocities

# Actualizar las partículas según el modelo NetLogo
def update_particles(positions, velocities, alpha, beta, speed):
    for i in range(len(positions)):
        position = positions[i]
        velocity = velocities[i]

        # Dirección actual de la partícula
        heading = cp.arctan2(velocity[1], velocity[0])

        # Encuentra vecinos y ajusta la dirección
        distances = cp.linalg.norm(positions - position, axis=1)
        angles_to_others = cp.arctan2(positions[:, 1] - position[1], positions[:, 0] - position[0])

        # Filtrar vecinos basados en vision_radius y la dirección
        is_neighbor = distances < vision_radius
        is_forward = cp.logical_and(-cp.pi/2 <= heading - angles_to_others, heading - angles_to_others <= cp.pi/2)
        forward_neighbors = cp.sum(cp.logical_and(is_neighbor, is_forward))
        total_neighbors = cp.sum(is_neighbor)

        # Ajustar la dirección según las reglas
        if forward_neighbors > total_neighbors - forward_neighbors:
            heading += alpha + total_neighbors * beta
        elif forward_neighbors < total_neighbors - forward_neighbors:
            heading -= alpha + total_neighbors * beta

        # Actualizar la posición y velocidad
        new_velocity = speed * cp.array([cp.cos(heading), cp.sin(heading)])
        positions[i] += new_velocity
        velocities[i] = new_velocity

    # Condiciones de contorno periódicas
    positions %= space_size

    return positions, velocities

# Parámetros
num_particles = 300
space_size = 100  # Tamaño del espacio de simulación
vision_radius = 5  # Radio de visión de las partículas
alpha = cp.radians(180)  # Ángulo de rotación base
beta = cp.radians(17)  # Ángulo de rotación por vecino
speed = 0.67  # Velocidad de las partículas

# Inicialización
positions, velocities = initialize_particles(num_particles)

# Configuración de la figura de animación
fig, ax = plt.subplots()
scatter = ax.scatter(positions.get()[:, 0], positions.get()[:, 1], s=10)
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.axis('off')

# Actualización de la animación cada N pasos
update_interval = 1  # Actualizar la figura cada N pasos

# Función de animación para actualizar la figura
def animate(frame_num):
    global positions, velocities
    for _ in range(update_interval):
        positions, velocities = update_particles(positions, velocities, alpha, beta, speed)
    scatter.set_offsets(positions.get())

# Crear y mostrar la animación
ani = FuncAnimation(fig, animate, frames=400 // update_interval, interval=1, blit=False)
plt.show()
