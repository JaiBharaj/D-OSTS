import numpy as np
from CrudeInitialConditions import InitialConditions as IC
from dosts.AtmosphericDensity import atmos_ussa1976_rho
from dosts.ModelDynamics import PolarAccelerations, SphericalAccelerations
from dosts.ExtendedKalmanFilters import ExtendedKalmanFilter, compute_F_analytic, compute_F_spherical
from dosts.PredictorIntegrator import RK45Integrator, Integrator3D
from WriteToFiles import write_to_file_2d, write_to_file_3d



def run_predictor(mode, input_file_name, output_file_name, uncertainty_file_name):
    if mode.upper()=='2D':
        run_predictor_2d(input_file_name, output_file_name, uncertainty_file_name)
    elif mode.upper()=='3D':
        run_predictor_3d(input_file_name, output_file_name, uncertainty_file_name)
    else:
        print("Mode unidentified. Choose between '2d' or '3d'.")


def run_predictor_2d(input_file_name, output_file_name, uncertainty_file_name):
    # file names
    input_file = input_file_name
    output_file = output_file_name
    uncertainty_file = uncertainty_file_name

    # Initial paramteres
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
    # Q = 1e-4 * np.eye(4)
    Q = np.diag([1000, 0, 1e-6, 0])

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

    ekf.predict_trajectory('2d', times, measurements)

    pred_traj = ekf.get_trajectory('2d')
    write_to_file_2d(output_file, pred_traj)

    pred_uncertainty = ekf.get_uncertainty('2d')
    write_to_file_2d(uncertainty_file, pred_uncertainty)




def run_predictor_3d(input_file_name, output_file_name, uncertainty_file_name):
    # file names
    input_file = input_file_name
    output_file = output_file_name
    uncertainty_file = uncertainty_file_name

    # Initial paramteres
    r0 = IC.earthRadius + IC.initSatAlt
    phi0 = IC.initSatPhi
    lam0 = IC.initSatLam

    CD = IC.dragCoeff
    A = IC.crossSec
    m = IC.satMass
    GM = IC.gravConstant * IC.earthMass

    # Measurement noise
    sigma_r = 100.0  # m
    sigma_theta = 1e-4  # rad
    sigma_phi = 1e-4  # rad
    R = np.diag([sigma_r ** 2, sigma_theta ** 2, sigma_phi ** 2])

    # Measurement model
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    # Process noise
    Q = np.diag([1000, 0, 1e-7, 0, 1e-7, 0])

    # Initial uncertainty
    P0 = np.diag([
        10.0 ** 2,  # r
        1.0 ** 2,  # vr
        (1e-4) ** 2,  # theta
        (1e-4) ** 2,  # vtheta
        (1e-4) ** 2,  # phi
        (1e-4) ** 2  # vphi
    ])

    # Model dynamics
    rho_func = lambda r: atmos_ussa1976_rho(r - IC.earthRadius)
    f_jacobian = lambda x: compute_F_spherical(x, CD, A, m, GM, rho_func=rho_func)
    f_dynamics = lambda x: SphericalAccelerations.accelerations(x[0], x[1], x[2], x[3], x[4], x[5])
    x0 = np.array([r0, IC.initSatRdot, phi0, IC.initSatPhidot, lam0, IC.initSatLamdot])

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
        integrator=Integrator3D()
    )

    ekf.predict_trajectory('3d', times, measurements)

    pred_traj = ekf.get_trajectory('3d')
    write_to_file_3d(output_file, pred_traj)

    pred_uncertainty = ekf.get_uncertainty('3d')
    write_to_file_3d(uncertainty_file, pred_uncertainty)



