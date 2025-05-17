import numpy as np
from dosts.CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import PolarAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic
from dosts.PredictorIntegrator import RK45Integrator

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# file names
input_file = f"Trajectories/2d_noisy_trajectory.txt"
output_file = f"Trajectories/2d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/2d_crash_heatmap_data.txt"

# Initial parameters
r0 = IC.earthRadius + IC.initSatAlt
theta0 = IC.initSatTheta

CD = IC.dragCoeff
A = IC.crossSec
m = IC.satMass
GM = IC.gravConstant * IC.earthMass

# Measurement noise
sigma_r, sigma_theta = 100.0, 0.0001      # meters, radians

# Measurement model
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

R = np.diag([sigma_r**2, sigma_theta**2])

# Process noise
Q = np.diag([1000, 0, 1e-7, 0])

# Initial uncertainty
P0 = np.diag([
    10.0**2,        # r
    1.0**2,         # vr
    (1e-4)**2,      # theta
    (1e-4)**2       # omega
])

# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_analytic(x, CD, A, m, GM, rho_func=rho_func)
f_dynamics = lambda x: PolarAccelerations.accelerations(x[0], x[1], x[2], x[3])
x0 = np.array([r0, 0.0, theta0, np.sqrt(GM / r0) / r0])

# Load radar data
data = np.loadtxt(input_file)
measurement_times = data[:, 0]
measurements = data[:, 1:]

# Instantiate EKF and simulate the predicted trajectory and uncertainty
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

# Reset files
with open(crash_heatmap_file, 'w') as f:  # Clear existing file
    f.write("")
with open(output_file, 'w') as f:  # Clear existing file
    f.write("")

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

    if i % 50 == 0:
        crash_angles = ekf.crash(N=50, max_steps=4000)
        print(f"Time {t:.1f}s: {len(crash_angles)} crash predictions")
        if len(crash_angles) > 0:
            # Write timestamp followed by angles
            with open(crash_heatmap_file, 'a') as f:
                f.write(f"{t:.6f} " + ' '.join(f"{angle:.6f}" for angle in crash_angles) + '\n')

# Save predicted trajectory
with open(output_file, 'w') as f:
    for t, x, P, measured in zip(times, states, covariances, is_measured_flags):
        r, theta = x[0], x[2]
        r_std = np.sqrt(P[0, 0])
        theta_std = np.sqrt(P[2, 2])
        f.write(f"{t:.6f} {r:.6f} {theta:.8f} {r_std:.6f} {theta_std:.8f} {int(measured)}\n")
