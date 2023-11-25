import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

bodies = [CelestialBody(1e30, [0, 0], [0, 0])]  # Cuerpo central masivo
num_orbiting_bodies = 5
for i in range(1, num_orbiting_bodies + 1):
    radius = 1e11 * i
    velocity = np.sqrt(6.67430e-11 * bodies[0].mass / radius)
    bodies.append(CelestialBody(1e28, [radius, 0], [0, velocity]))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-6e11, 6e11)
ax.set_ylim(-6e11, 6e11)

points = [ax.plot([], [], 'o', ms=5)[0] for _ in bodies]

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

dt = 10000  # Paso de tiempo en segundos
num_steps = 20000  # Número de pasos en la simulación

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, blit=True, interval=50)

plt.show()
