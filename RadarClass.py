class Radar:
    def __init__(self, ID, visibility_angle, location):
        self.__ID = ID
        self.__location = location
        self.__visibility_angle = visibility_angle
        self.__sigma_0 = 10 # baseline error typically 10-50m
        self.__k = 0.02 # scaling factor typically 0.01-0.05 m/km
        self.satellite_measurements = {'time': [], 'visibility': [], 'r': [], 'theta': []}
    
    # method to get ID
    def get_ID(self):
        return self.__ID
    
    # method to get location
    def get_location(self):
        return self.__location
    
    # method to get visibility angle
    def get_visibility_angle(self):
        return self.__visibility_angle
        
    # method to record satellites position at a time, if not visible then position recorded as 0,0
    def record_satellite(self, time, satellite_position):
        r, theta = satellite_position
        theta = theta%(2*np.pi)
        # check if satellite is visible by radar station
        visible = np.abs(theta - self.__location[1]) <= self.__visibility_angle
        if not visible:
            r,theta = np.nan, np.nan
        self.satellite_measurements['time'].append(time)
        self.satellite_measurements['visibility'].append(visible)
        self.satellite_measurements['r'].append(r)
        self.satellite_measurements['theta'].append(theta)
        
    # method to add noise and return noisy satellite recordings
    def add_noise(self):
        R, theta_rad = self.__location
        r = np.array(self.satellite_measurements['r']).astype(np.float64)
        theta_sat = np.array(self.satellite_measurements['theta']).astype(np.float64)
        distance = np.sqrt(r*r + R*R - 2*r*R*np.cos(theta_rad - theta_sat))
        eps = np.random.normal(0, self.__sigma_0 + self.__k*distance)
        r += eps # only add noise to the radial distance
        self.satellite_measurements['r'] = r
        self.satellite_measurements['theta'] = theta_sat
        
