import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Definición de la clase para los cuerpos celestes
class CelestialBody:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.force = np.zeros(2)

    def update_position(self, dt):
        self.position += self.velocity * dt

    def update_velocity(self, dt):
        self.velocity += self.force / self.mass * dt

def compute_gravitational_force(body1, body2):
    G = 6.67430e-11
    r = body2.position - body1.position
    distance = np.linalg.norm(r)
    if distance == 0:
        return np.zeros(2)
    force_magnitude = G * body1.mass * body2.mass / distance**2
    force_direction = r / distance
    return force_magnitude * force_direction

# Crear un conjunto más grande de cuerpos celestes
np.random.seed(0)  # Para reproducibilidad
num_bodies = 100
bodies = [CelestialBody(1e30, [0, 0], [0, 0])]  # Cuerpo masivo central
for _ in range(1, num_bodies):
    mass = np.random.uniform(1e28, 1e29)
    position = np.random.uniform(-1e11, 1e11, 2)
    velocity = np.random.uniform(-3e4, 3e4, 2)
    bodies.append(CelestialBody(mass, position, velocity))

# Configuración para la animación
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

points = [ax.plot([], [], 'o', ms=3)[0] for _ in bodies]

def init():
    for point in points:
        point.set_data([], [])
    return points

def update(frame):
    for body in bodies:
        body.force = np.zeros(2)
        for other_body in bodies:
            if body is not other_body:
                body.force += compute_gravitational_force(body, other_body)

    for body in bodies:
        body.update_velocity(dt)
        body.update_position(dt)

    for point, body in zip(points, bodies):
        point.set_data(body.position[0], body.position[1])
    return points

dt = 1000  # Paso de tiempo en segundos
num_steps = 10000  # Número de pasos en la simulación

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=20)

plt.show()
