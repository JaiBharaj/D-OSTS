import time
import math
import random
from CrudeInitialConditions import InitialConditions

true_filename = 'trajectory_3d.txt'
pred_filename = 'predicted_trajectory_3d.txt'

# Earth parameters
earth_radius = InitialConditions.earthRadius
altitude = InitialConditions.initSatAlt
orbital_radius = earth_radius + altitude
angular_velocity = 2 * math.pi / 90  # Rad p/s
dt = 0.02

# Clear files at start
open(true_filename, 'w').close()
open(pred_filename, 'w').close()

theta = 0.0
phi = math.pi/4
step = 0

print("Writing 3D fake trajectory and predicted data...")

while orbital_radius >= earth_radius:

    x = orbital_radius * math.sin(phi) * math.cos(theta)
    y = orbital_radius * math.sin(phi) * math.sin(theta)
    z = orbital_radius * math.cos(phi)

    with open(true_filename, 'a') as f:
        f.write(f"{x} {y} {z}\n")

    # Simulate prediction with decreasing noise over time
    uncertainty = max(10000, 40000 - step * 100)
    pred_x = x + random.gauss(0, uncertainty)
    pred_y = y + random.gauss(0, uncertainty)
    pred_z = z + random.gauss(0, uncertainty)
    is_measurement = 1 if step % 25 == 0 else 0

    with open(pred_filename, 'a') as f:
        f.write(f"{pred_x} {pred_y} {pred_z} {uncertainty} {uncertainty} {uncertainty} {is_measurement}\n")
        f.flush()

    theta += angular_velocity * dt
    phi += 0.001 * math.pi * dt
    # time.sleep(dt)
    orbital_radius -= 1
    step += 1