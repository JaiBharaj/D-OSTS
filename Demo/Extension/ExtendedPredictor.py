import numpy as np
from dosts.CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import SphericalAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_spherical
from dosts.PredictorIntegrator import Integrator3D
from dosts.WriteToFiles import write_to_file_2d

# Define here for now
def proportion_in_populated(crash_samples, integrator):
    count = 0
    for theta, phi in crash_samples:
        if integrator.in_populated(phi, theta):  # phi = latitude, theta = longitude
            count += 1
    return count / len(crash_samples) if len(crash_samples) > 0 else 0.0

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# file names
input_file = f"Trajectories/3d_noisy_trajectory.txt"
output_file = f"Trajectories/3d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/3d_crash_heatmap_data.txt"
thrust_crash_heatmap_file = f"Trajectories/3d_thrust_crash_heatmap_data.txt"

# Initial parameters
r0 = IC.earthRadius + IC.initSatAlt
theta0 = IC.initSatTheta
phi0 = IC.initSatPhi

CD = IC.dragCoeff
A = IC.crossSec
m = IC.satMass
GM = IC.gravConstant * IC.earthMass

# Measurement noise
sigma_r, sigma_theta, sigma_phi = 100.0, 0.0001, 0.0001      # meters, radians, radians

# Measurement model
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

R = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])

# Process noise
Q = np.zeros((6, 6))
Q[0, 0] = 100**2
Q[2, 2] = 1e-4**2
Q[4, 4] = 1e-4**2
Q[2, 4] = Q[4, 2] = 0.5 * 1e-4**2

# Initial uncertainty
P0 = np.diag([100.0, 1.0, 1e-8, 1e-8, 1e-8, 1e-8])

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_spherical(x, CD, A, m, GM, rho_func)
f_dynamics = lambda x: SphericalAccelerations.accelerations(*x)

x0 = np.array([r0,
               IC.initSatRdot,
               IC.initSatPhi,
               IC.initSatPhidot,
               IC.initSatLam,
               IC.initSatLamdot])

ekf = ExtendedKalmanFilter(
    f_dynamics=f_dynamics,
    f_jacobian=f_jacobian,
    H=H, Q=Q, R=R,
    x0=x0,
    P0=P0,
    integrator=Integrator3D()
)
########## EKF EXECUTION ##########
with open(crash_heatmap_file, 'w') as f:
    f.write("")
with open(thrust_crash_heatmap_file, 'w') as f:
    f.write("")
with open(output_file, 'w') as f:
    f.write("")

data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:4]

states, covariances, times, is_measured_flags = [], [], [], []
crash_theta_means, crash_phi_means = [], []
crash_theta_stds, crash_phi_stds = [], []

crash_theta_means_thrust, crash_phi_means_thrust = [], []
crash_theta_stds_thrust, crash_phi_stds_thrust = [], []

delta_v = 5000.0  # m/s
h_thrust = IC.hThrust  # m

POP_THRESHOLD = 0.25
SAFE_THRESHOLD = 0.05

log_path = "Trajectories/thrust_decision_log.txt"
with open(log_path, 'w') as f:
    f.write("Time\tp_pop\tp_pop_thrust\tThrustDecision\n")

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
    '''
    if i % 2500 == 0:
        crash_angles = ekf.crash3D(N=20, max_steps=10000)
        crash_angles_thrust = ekf.crash3D_with_thrust(delta_v=delta_v, h_thrust=h_thrust, N=20, max_steps=10000)

        with open(crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles for angle in pair) + '\n')

        with open(thrust_crash_heatmap_file, 'a') as f:
            f.write(f"{t:.6f} ")
            f.write(' '.join(f"{angle:.6f}" for pair in crash_angles_thrust for angle in pair) + '\n')

        # --- Decision Logic ---
        p_pop = proportion_in_populated(crash_angles, rk)
        p_pop_thrust = proportion_in_populated(crash_angles_thrust, rk)

        trigger_thrust = (p_pop > POP_THRESHOLD and p_pop_thrust < SAFE_THRESHOLD)

        print(f"[t = {t:.1f}s] p_pop = {p_pop:.3f}, p_thrust = {p_pop_thrust:.3f}, thrust = {trigger_thrust}")

        with open(log_path, 'a') as f:
            f.write(f"{t:.2f}\t{p_pop:.5f}\t{p_pop_thrust:.5f}\t{int(trigger_thrust)}\n")
    '''

########## SAVE TRAJECTORY ##########
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta, phi = x[0], x[2], x[4]
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {phi:.8f} "
                f"{np.sqrt(P[0, 0]):.3f} {np.sqrt(P[2, 2]):.3e} {np.sqrt(P[4, 4]):.3e} {int(measured)}\n")
