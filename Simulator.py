import WriteToFiles
from dosts import RadarModule
from dosts.RadarModule import *
from CrudeInitialConditions import InitialConditions as IC
from NumericalIntegrator import Integrator


def run_simulator(mode, H_dark=20_000, recorded_times=np.linspace(0, 6000, 6001), radar_angle=np.pi/2,
                  radar_noise_base=100, radar_noise_scalefactor=0.05, input_path=None, output_path=None):
    input_path = input_path
    output_path = output_path
    write_to_file = getattr(WriteToFiles, f"write_to_file_{mode}")

    ### GET TRUE TRAJECTORY ###
    rk = Integrator(recorded_times)
    get_trajectory = getattr(rk, f"get_trajectory_{mode}")
    true_traj = get_trajectory()
    write_to_file(input_path, true_traj)

    ### NOISY RADAR MEASUREMENTS ###
    distribute_radars = getattr(RadarModule, f"distribute_radars{mode.upper()}")
    radar_positions = distribute_radars(H_dark, IC.earthRadius)

    # Initialise radar stations
    radars = initialise_radar_stations(mode, radar_positions, radar_angle, radar_noise_base, radar_noise_scalefactor)

    # Record satellite positions in each radar
    for measurement in true_traj:
        sat_pos = measurement[1:]
        for radar in radars:
            radar.record_satellite(measurement[0], sat_pos)

    # Add measurement noise
    for radar in radars:
        radar.add_noise()

    # Combine measurements from all radars and write to file
    noisy_traj = combine_radar_measurements(mode, radars, true_traj)
    write_to_file(output_path, noisy_traj)

