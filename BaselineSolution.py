import numpy as np
from CrudeInitialConditions import InitialConditions
import RadarCombineMeasurements
from Atmospheric_Density import atmos_ussa1976_rho
from RadarClassNew import Radar
from RadarDistribution import distribute_radars2D
from CoordinateTransformations import PolarAccelerations
from NumericalIntegrator import Integrator
from ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic
from Visualiser import Visualiser2D
from PredictorIntegrator import RK45Integrator
import matplotlib.pyplot as plt

######## GENERATING TRUE TRAJECTORY ##########
rk = Integrator()
rk.get_trajectory()

########## RADAR STATION NOISY MEASUREMENTS ##########
input_path = "trajectory_without_noise.txt"
output_path = "noisy_radar_data.txt"

H_dark = 200000  # Radar min altitude (m)
radar_positions = distribute_radars2D(H_dark, InitialConditions.earthRadius)
radars = []

# Initialise radar stations
for i, (r_radar, theta_radar) in enumerate(radar_positions):
    radar = Radar(
        mode='2d',
        ID=f"Radar_{i}",
        location=[r_radar, theta_radar],
#         H_dark=H_dark,
#         sigma=100
    )
    radars.append(radar)

# Load true trajectory
true_traj = np.loadtxt(input_path)  # Columns: time, r, theta

# Record satellite positions in each radar
for time, r, theta in true_traj:
    sat_pos = [r, theta]
    for radar in radars:
        radar.record_satellite(time, sat_pos)

# Add measurement noise
for radar in radars:
    radar.add_noise()

# Combine measurements from all radars
noisy_traj = RadarCombineMeasurements.combine_radar_measurements(radars, true_traj)
RadarCombineMeasurements.write_to_file(output_path, noisy_traj)

# Load radar data from the noisy radar measurements file
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
sigma_r_meas = 100.0        # m
sigma_theta_meas = 1e-4   # rad
R = np.diag([sigma_r_meas**2, sigma_theta_meas**2])

# Process noise (initially zero)
# Q = np.zeros((4, 4))
Q = np.diag([1000, 0, 1e-7, 0])

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

rho_func = lambda r: atmos_ussa1976_rho(r - InitialConditions.earthRadius)

f_jacobian = lambda x: compute_F_analytic(
    x, CD, A, m, GM,
    rho_func=rho_func
)

f_dynamics = lambda x: PolarAccelerations.accelerations(x[0], x[1], x[2], x[3])
x0 = np.array([rk.r0, 0.0, rk.theta0, np.sqrt(GM / rk.r0) / rk.r0])

# Load radar data
data = np.loadtxt(output_path)
times = data[:, 0]
measurements = data[:, 1:3]

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H,
    Q=Q,
    R=R,
    x0=x0,
    P0=P0,
    integrator=RK45Integrator(CD, A, m, GM, rho_func)
)

########## EKF LOOP ##########
crash_heatmap_file = "crash_heatmap_data.txt"
with open(crash_heatmap_file, 'w') as f:  # Clear existing file
    f.write("")

data = np.loadtxt(output_path)
measurement_times = data[:, 0]
measurements = data[:, 1:3]

states, covariances, times, is_measured_flags = [], [], [], []
crash_means, crash_stds = [], []

for i, (t, z) in enumerate(zip(measurement_times, measurements)):
    dt = 1e-3 if i == 0 else t - measurement_times[i - 1]
    x, P = ekf.predict(dt)

    is_measured = False
    if not np.isnan(z).any():
        x, P = ekf.update(z)
        is_measured = True

    times.append(t)
    states.append(x.copy())
    covariances.append(P.copy())
    is_measured_flags.append(is_measured)

    if i % 1000 == 0:
        crash_angles = ekf.crash(N=10, max_steps=5000)
        print(f"Time {t:.1f}s: {len(crash_angles)} crash predictions")
        if len(crash_angles) > 0:
            # Write timestamp followed by angles
            with open(crash_heatmap_file, 'a') as f:
                f.write(f"{t:.6f} " + ' '.join(f"{angle:.6f}" for angle in crash_angles) + '\n')

########## SAVE OUTPUT ##########
with open("ekf_predicted_trajectory.txt", 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta = x[0], x[2]
        r_std = np.sqrt(P[0, 0])
        theta_std = np.sqrt(P[2, 2])
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {r_std:.6f} {theta_std:.8f} {int(measured)}\n")

########## VISUALISE ##########
if crash_means:
    steps = np.arange(0, len(crash_means) * 5, 5)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, crash_means, label='Crash θ Estimate')
    plt.fill_between(steps, np.array(crash_means) - np.array(crash_stds),
                     np.array(crash_means) + np.array(crash_stds), alpha=0.3, label='1σ Band')
    plt.xlabel('Time Step')
    plt.ylabel('Crash θ (rad)')
    plt.title('Crash Angle Prediction Over Time')
    plt.legend()
    plt.grid()
    plt.show()

vis = Visualiser2D("trajectory_without_noise.txt", "ekf_predicted_trajectory.txt", "crash_heatmap_data.txt", mode='prewritten')
vis.visualise()