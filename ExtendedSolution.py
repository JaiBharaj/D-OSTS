import numpy as np
from CrudeInitialConditions import InitialConditions
import RadarCombineMeasurements
from Atmospheric_Density import atmos_ussa1976_rho
from RadarClassNew import Radar
from RadarDistribution import distribute_radars3D
from CoordinateTransformations import SphericalAccelerations
from NumericalIntegrator import Integrator
from ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_spherical
from Visualiser import Visualiser3D
from PredictorIntegrator import Integrator3D

########## GENERATING TRUE TRAJECTORY ##########
rk = Integrator()
rk.get_trajectory_3d()

########## RADAR STATION NOISY MEASUREMENTS ##########
input_path = "trajectory_without_noise_3d.txt"
output_path = "noisy_radar_data_3d.txt"

H_dark = 200000  # Radar min altitude (m)
radar_positions = distribute_radars3D(H_dark, InitialConditions.earthRadius)
radars = []

# Initialise radar stations
for i, (r_radar, theta_radar, phi_radar) in enumerate(radar_positions):
    radar = Radar(
        mode='3d',
        ID=f"Radar_{i}",
        location=[r_radar, theta_radar, phi_radar]
    )
    radars.append(radar)

# Load true trajectory
true_traj = np.loadtxt(input_path)  # Columns: time, r, theta

# Record satellite positions in each radar
for time, r, theta, phi in true_traj:
    sat_pos = [r, theta, phi]
    for radar in radars:
        radar.record_satellite(time, sat_pos)

# Add measurement noise
for radar in radars:
    radar.add_noise()

# Combine measurements from all radars
noisy_traj = RadarCombineMeasurements.combine_radar_measurements_3d(radars, true_traj)
RadarCombineMeasurements.write_to_file_3d(output_path, noisy_traj)

# Load radar data from the noisy radar measurements file
measurement_times = []
measurements = []

with open(output_path, 'r') as f:
    for line in f:
        t_str, r_str, theta_str, phi_str = line.strip().split()
        t = float(t_str)
        r = float(r_str)
        theta = float(theta_str)
        phi = float(phi_str)
        measurement_times.append(t)
        measurements.append(np.array([r, theta, phi]))

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########
# Measurement model
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

# Measurement noise
sigma_r_meas = 100.0        # m
sigma_theta_meas = 1e-4   # rad
sigma_phi_meas = 1e-4 # rad
R = np.diag([sigma_r_meas**2, sigma_theta_meas**2, sigma_phi_meas**2])

# Process noise (initially zero)
Q = np.diag([1000, 0, 1e-7, 0, 1e-7, 0])
# Q = np.diag([10000, 0, 0, 0, 0, 0])

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2,       # vtheta
    (1e-4)**2,      # phi
    (1e-4)**2       # vphi
])

CD = InitialConditions.dragCoeff
A = InitialConditions.crossSec
m = InitialConditions.satMass
GM = InitialConditions.gravConstant * InitialConditions.earthMass

rho_func = lambda r: atmos_ussa1976_rho(r - InitialConditions.earthRadius)

f_jacobian = lambda x: compute_F_spherical(
    x, CD, A, m, GM,
    rho_func=rho_func
)

f_dynamics = lambda x: SphericalAccelerations.accelerations(x[0], x[1], x[2], x[3], x[4], x[5])
x0 = np.array([rk.r0, 
               InitialConditions.initSatRdot,
               InitialConditions.initSatPhi,
               InitialConditions.initSatPhidot,
               InitialConditions.initSatLam,
               rk.lam_dot0])

# Load radar data
data = np.loadtxt(output_path)
times = data[:, 0]
measurements = data[:, 1:5]

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator3D()
)

input_file = input_path
output_file = "ekf_predicted_trajectory_3d.txt"

states = []
covariances = []
times = []
is_measured_flags = []

x = x0.copy()
P = P0.copy()

for i in range(len(measurement_times)):
    t = measurement_times[i]
    z = measurements[i]

    if i == 0:
        dt = 1e-3  # Small nonzero dt for first Jacobian estimate
    else:
        dt = t - measurement_times[i - 1]

    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

# Save to file
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r = x[0]
        theta = x[2]
        phi = x[4]
        r_uncertainty = np.sqrt(P[0, 0])
        theta_uncertainty = np.sqrt(P[2, 2])
        phi_uncertainty = np.sqrt(P[4, 4])
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {phi:.8f} {r_uncertainty:.6f} {theta_uncertainty:.8f} {phi_uncertainty:.8f} {int(measured)}\n")

vis = Visualiser3D(input_file, output_file, mode='prewritten')
vis.visualise()