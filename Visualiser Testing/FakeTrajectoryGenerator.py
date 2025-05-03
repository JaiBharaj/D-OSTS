import time
import math
import random
from CrudeInitialConditions import InitialConditions

true_filename = 'trajectory.txt'
pred_filename = 'predicted_trajectory.txt'

# Orbit parameters
earth_radius = InitialConditions.earthRadius
altitude = InitialConditions.initSatAlt
orbital_radius = earth_radius + altitude
angular_velocity = 2 * math.pi / 90 # Rad p/s
dt = 0.2

# Clear files at start
open(true_filename, 'w').close()
open(pred_filename, 'w').close()

theta = 0.0
step = 0

print("Writing fake trajectory and predicted data...")

while orbital_radius >= earth_radius:
    x = orbital_radius * math.cos(theta)
    y = orbital_radius * math.sin(theta)

    # Write true position
    with open(true_filename, 'a') as f:
        f.write(f"{x} {y}\n")

    # Simulate prediction with decreasing noise over time
    uncertainty = max(10000, 40000 - step * 100)
    pred_x = x + random.gauss(0, uncertainty)
    pred_y = y + random.gauss(0, uncertainty)
    is_measurement = 1 if step % 25 == 0 else 0  # Every 25 steps is a measurement from radar station

    # Write prediction
    with open(pred_filename, 'a') as f:
        f.write(f"{pred_x} {pred_y} {uncertainty} {uncertainty} {is_measurement}\n")
        f.flush()

    theta += angular_velocity * dt
    time.sleep(dt)
    orbital_radius -= 1000
    step += 1
