import numpy as np


def distribute_radars2D(Hdark, R_earth):
    distribution_angle = np.arccos(R_earth / (R_earth + Hdark))
    num_stations = int(np.ceil(np.pi / distribution_angle))
    interval_points = np.linspace(0, 2 * np.pi, num_stations, endpoint=False)
    radar_coors = np.column_stack((np.full(num_stations, R_earth), interval_points))
    return radar_coors


def fibonacci_sphere_grid(N):
    gold = 0.5 * (1 + np.sqrt(5))
    ind = np.array([i for i in range(0, N)])
    x = (ind / gold) % gold
    y = ind / (N - 1)
    theta = 2 * np.pi * x
    phi = np.arccos(1 - 2 * y)
    return theta, phi


def distribute_radars3D(H_dark, R_Earth):
    theta = np.arccos(R_Earth / (R_Earth + H_dark))  # angle of visibility
    area_cap = 0.5 * (1 - np.cos(theta))
    num_stations = int(np.ceil(1 / area_cap))
    theta, phi = fibonacci_sphere_grid(num_stations)
    return np.column_stack((np.full(num_stations, R_Earth), theta, phi))

def initialise_radar_stations(mode, radar_positions, radar_angle, radar_noise_base, radar_noise_scalefactor):
    radars = []
    for i, position in enumerate(radar_positions):
        radar = Radar(
            mode=mode,
            ID=f"Radar_{i}",
            location=position,
            visibility_angle=radar_angle,
            sigma_0=radar_noise_base,
            k=radar_noise_scalefactor
        )
        radars.append(radar)
    return radars


def weighted_average(x, noise):
    weights = 1 / noise ** 2
    return np.sum(weights * x) / np.sum(weights)


def combine_radar_measurements(mode, radars, true_traj):
    times = true_traj[:, 0]
    theta_arr = true_traj[:, 2]
    r_arr = np.zeros(len(times))

    if mode.upper()=='3D':
        phi_arr = true_traj[:, 3]

    for i, time in enumerate(times):
        seen_radius = []
        noise = []
        for rad in radars:
            is_visible = rad.satellite_measurements['visibility'][i]
            if is_visible:
                seen_radius.append(rad.satellite_measurements['r'][i])
                noise.append(rad.get_noise()[i])

        if len(seen_radius) == 0:
            r_arr[i] = np.nan  # <-- key fix: no visible radars
        else:
            seen_radius = np.array(seen_radius)
            noise = np.array(noise)
            r_arr[i] = weighted_average(seen_radius, noise)

    if mode.upper() == '3D':
        return np.array([times, r_arr, theta_arr, phi_arr]).T
    else:
        return np.array([times, r_arr, theta_arr]).T




class Radar:
    def __init__(self, mode, ID, location, visibility_angle=np.pi/2, sigma_0=50, k=0.05):
        self.__ID = ID
        self.__mode = mode.upper()  # 2D or 3D
        self.__location = np.array(location)
        self.__visibility_angle = visibility_angle # 80-90 degrees
        self.__sigma_0 = sigma_0  # baseline error typically 10-50m
        self.__k = k  # scaling factor typically 0.01-0.05 m/km
        self.__noise = None
        self.satellite_measurements = {'time': [], 'visibility': [], 'r': [], 'theta': [], 'phi': []}
        if mode.upper() != '2D' and mode.upper() != '3D':
            raise Exception('Mode unclear. Set to 2D or 3D as a string.')

    # method to get ID
    def get_ID(self):
        return self.__ID

    # method to get location
    def get_location(self):
        return self.__location

    # method to get visibility angle
    def get_visibility_angle(self):
        return self.__visibility_angle

    # method to get the noise vector
    def get_noise(self):
        return self.__noise

    @staticmethod
    def polar_to_cartesian(position):
        r, theta = position
        x = r * np.cos(theta % (2 * np.pi))
        y = r * np.sin(theta % (2 * np.pi))
        return np.array([x, y])

    @staticmethod
    def spherical_to_cartesian(position):
        r, theta, phi = position
        x = r * np.sin(phi) * np.cos(theta)  # phi is polar angle, theta is azimuthal
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.array([x, y, z])

    # method to compute distnce between two position vectors in polar/spherical coordinates
    def distance(self, mode, u, v):
        if mode == '2D':
            cart_u = self.polar_to_cartesian(np.array(u))
            cart_v = self.polar_to_cartesian(np.array(v))
            n = 2
        else:
            cart_u = self.spherical_to_cartesian(np.array(u))
            cart_v = self.spherical_to_cartesian(np.array(v))
            n = 3
        cart_u = cart_u.reshape(n, 1)
        return np.linalg.norm(cart_u - cart_v, axis=0)

    # method to check whether satellite is visible by a radar station
    def check_visibility(self, satellite_position):
        if self.__mode == '2D':
            cart_rad = self.polar_to_cartesian(self.__location)
            cart_sat = self.polar_to_cartesian(satellite_position)
        else:
            cart_rad = self.spherical_to_cartesian(self.__location)
            cart_sat = self.spherical_to_cartesian(satellite_position)

        rad_to_sat = (cart_sat - cart_rad)  # vector from radar station to satellite
        rad_to_sat /= np.linalg.norm(rad_to_sat)  # normalise the vector
        rad_normal = cart_rad / np.linalg.norm(cart_rad)  # normalise radar station vector
        cos_angle_difference = np.dot(rad_to_sat, rad_normal)  # cosine of angle between radar and satellite
        return cos_angle_difference >= np.cos(self.__visibility_angle)

    # method to record satellites position at a time, if not visible then position recorded as 0,0
    def record_satellite(self, time, satellite_position):
        if self.__mode == '2D':
            r, theta = satellite_position
            theta = theta % (2 * np.pi)
        elif self.__mode == '3D':
            r, theta, phi = satellite_position
            theta = theta % np.pi  # theta: polar angle (0 to π)
            phi = phi % (2 * np.pi)  # phi: azimuthal angle (0 to 2π)

        # check if satellite is visible by radar station
        visible = self.check_visibility(satellite_position)
        if not visible:
            r, theta, phi = np.nan, np.nan, np.nan
        self.satellite_measurements['time'].append(time)
        self.satellite_measurements['visibility'].append(visible)
        self.satellite_measurements['r'].append(r)
        self.satellite_measurements['theta'].append(theta)
        if self.__mode == '3D':
            self.satellite_measurements['phi'].append(phi)

    # method to add noise and return noisy satellite recordings
    def add_noise(self):
        if self.__mode == '2D':
            R, theta_rad = self.__location
        else:
            R, theta_rad, phi_rad = self.__location
        r = self.satellite_measurements['r']
        theta_sat = self.satellite_measurements['theta']
        if self.__mode == '3D':
            phi_sat = self.satellite_measurements['phi']
            sat_positions = np.array([r, theta_sat, phi_sat])
        else:
            sat_positions = np.array([r, theta_sat])

        distance = self.distance(self.__mode, self.__location, sat_positions)
        r_std = self.__sigma_0 + self.__k * distance
        th_std = 0.001
        eps_r = np.random.normal(0, r_std, len(r))
        eps_th = np.random.normal(0, th_std, len(r))
        self.__noise = eps_r  # save the noise vector
        self.satellite_measurements['r'] = r + eps_r
        self.satellite_measurements['theta'] = theta_sat + eps_th

