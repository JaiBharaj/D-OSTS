import numpy as np

import CoordinateTransformations
from CrudeInitialConditions import InitialConditions
import Atmospheric_Density
import RadarClass
from NumericalIntegrator import Integrator
from ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic
from Visualiser import Visualiser2D

########## GENERATING TRUE TRAJECTORY ##########
Integrator.initialize(mu_value=InitialConditions.gravConstant * InitialConditions.earthMass)
Integrator.get_trajectory()

########## RADAR CODE GOES HERE ##########
# ---- ---- ---- ---- ---- ---- ---- ----
#   ---- ---- ---- ---- ---- ---- ---- ---

################# GENERATING FAKE RADAR DATA FOR TESTING #####################
############## DELETE AFTER REAL RADAR CODE IS INSERTED ABOVE ################
input_path = "trajectory_without_noise.txt"                                 ##
output_path = "fake_radar_data.txt"                                         ##
with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:    ##
    for i, line in enumerate(infile):                                       ##
        if i % 100 == 0:                                                    ##
            outfile.write(line)                                             ##
##############################################################################

# Load radar data from the fake radar file
measurement_times = []
measurements = []

with open(output_path, 'r') as f:
    for line in f:
        t_str, r_str, theta_str = line.strip().split()
        t = float(t_str)
        r = float(r_str)
        theta = float(theta_str)
        measurement_times.append(t)
        measurements.append(np.array([r, theta]))

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# Measurement model
H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

# Measurement noise
sigma_r_meas = 5.0        # m
sigma_theta_meas = 1e-4   # rad
R = np.diag([sigma_r_meas**2, sigma_theta_meas**2])

# Process noise (initially zero)
Q = np.zeros((4, 4))

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2       # omega
])

CD = InitialConditions.dragCoeff
A = InitialConditions.crossSec
m = InitialConditions.satMass
GM = InitialConditions.gravConstant * InitialConditions.earthMass

f_jacobian = lambda x: compute_F_analytic(x, CD, A, m, GM, Atmospheric_Density.atmos_ussa1976_rho())
x0 = np.array([Integrator.r0, 0.0, 0.0, np.sqrt(GM / Integrator.r0) / Integrator.r0])

# Load radar data
data = np.loadtxt(output_path)
times = data[:, 0]
measurements = data[:, 1:3]

ekf = ExtendedKalmanFilter(
    f_dynamics=CoordinateTransformations.PolarAccelerations,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator
)

input_file = input_path
output_file = "ekf_predicted_trajectory.txt"

states = []
covariances = []
times = []
is_measured_flags = []

x = x0.copy()
P = P0.copy()

for i in range(len(measurement_times)):
    t = measurement_times[i]
    z = measurements[i]  # `None` if no measurement at this time

    dt = t - times[-1] if times else t  # Use full time if first step

    # Predict
    x, P = ekf.predict(dt)
    is_measured = False

    # Update if measurement is available
    if z is not None:
        x, P = ekf.update(z)
        is_measured = True

    # Record everything
    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

# Save to file
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r = x[0]
        theta = x[2]
        r_uncertainty = np.sqrt(P[0, 0])
        theta_uncertainty = np.sqrt(P[2, 2])
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {r_uncertainty:.6f} {theta_uncertainty:.8f} {measured}\n")

vis = Visualiser2D(input_file, output_file)
vis.visualise()