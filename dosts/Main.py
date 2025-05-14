from Simulator import run_simulator
from Predictor import run_predictor
from CrudeInitialConditions import InitialConditions as IC
import numpy as np

#====== CHANGE BELOW WITH YOUR INITIAL CONDITIONS ======#

IC.index = 1 # index for keeping track when running different
IC.dragCoeff = 2
IC.crossSec = 10
IC.satMass = 60

IC.earthMass = 6E+24
IC.earthRadius = 6.37E+6
IC.gravConstant = 6.67E-11

# initial values of satellite
IC.deltaV = 80.0
IC.initSatAlt = 400_000
IC.initSatTheta = 0.0
IC.initSatPhi = np.pi / 2  # equatorial
IC.initSatLam = 0.0
IC.initSatRdot = 0.0
IC.initSatPhidot = np.radians(0.0) / 1.0

# initial settings for bonus
IC.populatedRadius = 50000  # radius of populated area (m)
IC.populatedCenters = [
    (np.radians(51.5074), np.radians(-0.1278)),  # London
    (np.radians(40.7128), np.radians(-74.0060)),  # New York
    (np.radians(48.8566), np.radians(2.3522)),  # Paris
    (np.radians(34.0522), np.radians(-118.2437)),  # Los Angeles
]
IC.hThrust = 100000  # height of thrust (m)
IC.deltaV_from_thrust = 200  # velocity increasing value (m/s)

#=== RUN SIMULATOR/PREDICTOR WITH DESIRED FUNCTIONALITY ===#
mode = '3d'         # or '3d'
H_dark = 50_000     # altitude below which satellite may not be visible to any radar
radar_angle = np.pi/2
radar_noise_base = 50          # meters
radar_noise_scalefactor = 0.01   # m/km
# recorded_times = np.linspace(0, 9000, 9001) # this gives evenly spaced measurements at every 5 seconds
recorded_times = None

true_traj_path = f"Trajectories/{IC.index}_{mode}_true_trajectory.txt"
noisy_traj_path = f"Trajectories/{IC.index}_{mode}_noisy_trajectory.txt"

pred_traj_path = f"Trajectories/{IC.index}_{mode}_pred_trajectory.txt"
pred_uncertainty_path = f"Trajectories/{IC.index}_{mode}_pred_uncertainty.txt"

run_simulator(mode, H_dark, recorded_times,
              radar_angle, radar_noise_base, radar_noise_scalefactor,
              true_traj_path, noisy_traj_path)

run_predictor(mode, noisy_traj_path, pred_traj_path, pred_uncertainty_path)




