import numpy as np

class Radar:
    def __init__(self, ID, location, sigma=10, k=0.02):
        self.__ID = ID
        self.__location = np.array(location)
        self.__visibility_angle = np.pi / 2  # 80-90 degrees
        self.__sigma_0 = sigma  # baseline error typically 10-50m
        self.__min_sigma = 1.0  # ensures noise doesn't get unrealistically small
        self.__k = k  # scaling factor typically 0.01-0.05 m/km
        self.satellite_measurements = {'time': [], 'visibility': [], 'r': [], 'theta': []}
        self.__last_recorded_time = None
        self.__measurement_interval = 25.0  # seconds

    # method to get ID
    def get_ID(self):
        return self.__ID

    # method to get location
    def get_location(self):
        return self.__location

    # method to get visibility angle
    def get_visibility_angle(self):
        return self.__visibility_angle

    def get_noise(self):
        noise_values = []
        for i, visible in enumerate(self.satellite_measurements['visibility']):
            if visible:
                r = self.satellite_measurements['r'][i]
                theta = self.satellite_measurements['theta'][i]
                R = self.__location[0]
                distance = np.sqrt(r ** 2 + R ** 2 - 2 * r * R * np.cos(theta - self.__location[1]))
                noise_values.append(self.__sigma_0 + self.__k * distance)
            else:
                noise_values.append(np.nan)
        return noise_values

    @staticmethod
    def polar_to_cartesian(position):
        r, theta = position
        x = r * np.cos(theta % (2 * np.pi))
        y = r * np.sin(theta % (2 * np.pi))
        return np.array([x, y])

    # method to check whether satellite is visible by a radar station
    def check_visibility(self, satellite_position):
        cart_rad = self.polar_to_cartesian(self.__location)
        cart_sat = self.polar_to_cartesian(satellite_position)
        rad_to_sat = (cart_sat - cart_rad)  # vector from radar station to satellite
        rad_to_sat /= np.linalg.norm(rad_to_sat)  # normalise the vector
        rad_normal = cart_rad / np.linalg.norm(cart_rad)  # normalise radar station vector
        cos_angle_difference = np.dot(rad_to_sat, rad_normal)  # cosine of angle between radar and satellite
        return cos_angle_difference >= np.cos(self.__visibility_angle)

    # method to record satellites position at a time, if not visible then position recorded as 0,0
    def record_satellite(self, time, satellite_position):
        if self.__last_recorded_time is not None:
            if time - self.__last_recorded_time < self.__measurement_interval:
                # Still pad with NaNs to keep alignment
                self.satellite_measurements['time'].append(time)
                self.satellite_measurements['visibility'].append(False)
                self.satellite_measurements['r'].append(np.nan)
                self.satellite_measurements['theta'].append(np.nan)
                return

        r, theta = satellite_position
        theta = theta % (2 * np.pi)
        visible = self.check_visibility(satellite_position)

        if not visible:
            r, theta = np.nan, np.nan
        else:
            self.__last_recorded_time = time

        self.satellite_measurements['time'].append(time)
        self.satellite_measurements['visibility'].append(visible)
        self.satellite_measurements['r'].append(r)
        self.satellite_measurements['theta'].append(theta)

    # method to add noise and return noisy satellite recordings
    def add_noise(self):
        R, theta_rad = self.__location
        r = np.array(self.satellite_measurements['r']).astype(np.float64)
        theta_sat = np.array(self.satellite_measurements['theta']).astype(np.float64)
        distance = np.sqrt(r * r + R * R - 2 * r * R * np.cos(theta_rad - theta_sat))
        eps_std = np.maximum(self.__sigma_0 - self.__k * distance, self.__min_sigma)
        eps = np.random.normal(0, eps_std)
        r += eps  # only add noise to the radial distance
        self.satellite_measurements['r'] = r
        self.satellite_measurements['theta'] = theta_sat
