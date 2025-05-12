import numpy as np
from dosts.CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import PolarAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic
from dosts.PredictorIntegrator import RK45Integrator
from dosts.WriteToFiles import write_to_file_2d

########## TRAJECTORY PREDICTIONS WITH EXTENDED KALMAN FILTER ##########

# file names
input_file = f"Trajectories/{IC.index}_2d_noisy_trajectory.txt"
output_file = f"Trajectories/{IC.index}_2d_pred_trajectory.txt"
uncertainty_file = f"Trajectories/{IC.index}_2d_pred_uncertainty.txt"

# Initial parameteres
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

Q = 1e-4*np.eye(4)  # Process noise (initially zero)

# Initial uncertainty
P0 = np.diag([10.0**2, 1.0**2, 1e-8, 1e-8]) # [r, vr, theta, omega]


# Model dynamics
rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
f_jacobian = lambda x: compute_F_analytic(x, CD, A, m, GM, rho_func=rho_func)
f_dynamics = lambda x: PolarAccelerations.accelerations(x[0], x[1], x[2], x[3])
x0 = np.array([r0, 0.0, theta0, np.sqrt(GM / r0) / r0])


# Load radar data
data = np.loadtxt(input_file)
times = data[:, 0]
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

ekf.predict_trajectory(times, measurements, x0.copy(), P0.copy())

pred_traj = ekf.get_trajectory()
write_to_file_2d(output_file, pred_traj)

pred_uncertainty = ekf.get_uncertainty()
write_to_file_2d(uncertainty_file, pred_uncertainty)