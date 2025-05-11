import numpy as np
import matplotlib.pyplot as plt
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

########## TRUE TRAJECTORY ##########
rk = Integrator()
rk.get_trajectory_3d()

########## RADAR SIMULATION ##########
input_path = "trajectory_without_noise_3d.txt"
output_path = "noisy_radar_data_3d.txt"

H_dark = 100000
radar_positions = distribute_radars3D(H_dark, InitialConditions.earthRadius)
radars = [Radar(mode='3d', ID=f"Radar_{i}", location=pos) for i, pos in enumerate(radar_positions)]

true_traj = np.loadtxt(input_path)
for time, r, theta, phi in true_traj:
    for radar in radars:
        radar.record_satellite(time, [r, theta, phi])
for radar in radars:
    radar.add_noise()

noisy_traj = RadarCombineMeasurements.combine_radar_measurements_3d(radars, true_traj)
RadarCombineMeasurements.write_to_file_3d(output_path, noisy_traj)

########## EKF SETUP ##########
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])
R = np.diag([100.0**2, 1e-4**2, 1e-4**2])
Q = np.zeros((6, 6))
Q[0, 0] = 100**2
Q[2, 2] = 1e-4**2
Q[4, 4] = 1e-4**2
Q[2, 4] = Q[4, 2] = 0.5 * 1e-4**2

P0 = np.diag([100.0, 1.0, 1e-8, 1e-8, 1e-8, 1e-8])
CD, A, m = InitialConditions.dragCoeff, InitialConditions.crossSec, InitialConditions.satMass
GM = InitialConditions.gravConstant * InitialConditions.earthMass
rho_func = lambda r: atmos_ussa1976_rho(r - InitialConditions.earthRadius)
f_jacobian = lambda x: compute_F_spherical(x, CD, A, m, GM, rho_func)
f_dynamics = lambda x: SphericalAccelerations.accelerations(*x)

x0 = np.array([rk.r0,
               InitialConditions.initSatRdot,
               InitialConditions.initSatPhi,
               InitialConditions.initSatPhidot,
               InitialConditions.initSatLam,
               rk.lam_dot0])

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H, Q=Q, R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator3D()
)

########## EKF EXECUTION ##########
crash_heatmap_file = "crash_heatmap_data_3d.txt"
with open(crash_heatmap_file, 'w') as f:
    f.write("")

crash_heatmap_file_thrust = "crash_heatmap_data_3d_thrust.txt"
with open(crash_heatmap_file_thrust, 'w') as f:
    f.write("")

data = np.loadtxt(output_path)
measurement_times = data[:, 0]
measurements = data[:, 1:4]

states, covariances, times, is_measured_flags = [], [], [], []
crash_theta_means, crash_phi_means = [], []
crash_theta_stds, crash_phi_stds = [], []

crash_theta_means_thrust, crash_phi_means_thrust = [], []
crash_theta_stds_thrust, crash_phi_stds_thrust = [], []

delta_v = 80.0  # m/s
h_thrust = InitialConditions.hThrust  # m

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

    if i % 250 == 0:
        crash_angles = ekf.crash3D(N=20, max_steps=10000)
        crash_angles_thrust = ekf.crash3D_with_thrust(delta_v=delta_v, h_thrust=h_thrust, N=20, max_steps=10000)

        with open(crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles for angle in pair) + '\n')

        with open(crash_heatmap_file_thrust, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles_thrust for angle in pair) + '\n')

########## SAVE TRAJECTORY ##########
with open("ekf_predicted_trajectory_3d.txt", 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta, phi = x[0], x[2], x[4]
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {phi:.8f} "
                f"{np.sqrt(P[0, 0]):.3f} {np.sqrt(P[2, 2]):.3e} {np.sqrt(P[4, 4]):.3e} {int(measured)}\n")

########## VISUALISATION ##########
vis = Visualiser3D("trajectory_without_noise_3d.txt", "ekf_predicted_trajectory_3d.txt", "crash_heatmap_data_3d.txt", mode='prewritten')
vis.visualise()

vis_thrust = Visualiser3D("trajectory_without_noise_3d.txt", "ekf_predicted_trajectory_3d.txt", "crash_heatmap_data_3d.txt", "crash_heatmap_data_3d_thrust.txt", mode='prewritten')
vis_thrust.visualise()
