import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

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
        heading = cp.arctan2(velocity[1], velocity[0])
        distances = cp.linalg.norm(positions - position, axis=1)
        angles_to_others = cp.arctan2(positions[:, 1] - position[1], positions[:, 0] - position[0])
        is_neighbor = distances < vision_radius
        is_forward = cp.logical_and(-cp.pi/2 <= heading - angles_to_others, heading - angles_to_others <= cp.pi/2)
        forward_neighbors = cp.sum(cp.logical_and(is_neighbor, is_forward))
        total_neighbors = cp.sum(is_neighbor)
        if forward_neighbors > total_neighbors - forward_neighbors:
            heading += alpha + total_neighbors * beta
        elif forward_neighbors < total_neighbors - forward_neighbors:
            heading -= alpha + total_neighbors * beta
        new_velocity = speed * cp.array([cp.cos(heading), cp.sin(heading)])
        positions[i] += new_velocity
        velocities[i] = new_velocity
    positions %= space_size
    return positions, velocities

# Parámetros iniciales
num_particles = 300
space_size = 100
vision_radius = 5
alpha = cp.radians(180)
beta = cp.radians(17)
speed = 0.67

# Inicialización
positions, velocities = initialize_particles(num_particles)

# Configuración de la figura de animación y widgets
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
scatter = ax.scatter(positions.get()[:, 0], positions.get()[:, 1], s=10)
ax.set_xlim(0, space_size)
ax.set_ylim(0, space_size)
ax.axis('off')

# Crear sliders para parámetros
ax_vision_radius = plt.axes([0.25, 0.1, 0.65, 0.03]) # type: ignore
s_vision_radius = Slider(ax_vision_radius, 'Vision Radius', 0.1, 10.0, valinit=vision_radius)

ax_alpha = plt.axes([0.25, 0.15, 0.65, 0.03]) # type: ignore
s_alpha = Slider(ax_alpha, 'Alpha', 0, 360, valinit=cp.degrees(alpha).get())

ax_beta = plt.axes([0.25, 0.20, 0.65, 0.03]) # type: ignore
s_beta = Slider(ax_beta, 'Beta', 0, 360, valinit=cp.degrees(beta).get())

ax_speed = plt.axes([0.25, 0.25, 0.65, 0.03]) # type: ignore
s_speed = Slider(ax_speed, 'Speed', 0.1, 2.0, valinit=speed)

# Función de actualización para sliders
def update(val):
    global vision_radius, alpha, beta, speed
    vision_radius = s_vision_radius.val
    alpha = cp.radians(float(s_alpha.val))  # Convierte a float y luego a radianes
    beta = cp.radians(float(s_beta.val))    # Convierte a float y luego a radianes
    speed = s_speed.val

s_vision_radius.on_changed(update)
s_alpha.on_changed(update)
s_beta.on_changed(update)
s_speed.on_changed(update)

# Función de animación para actualizar la figura
def animate(frame_num):
    global positions, velocities
    positions, velocities = update_particles(positions, velocities, alpha, beta, speed)
    scatter.set_offsets(positions.get())

# Crear y mostrar la animación
ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=False)
plt.show()