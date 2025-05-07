import numpy as np

class Radar:
    def __init__(self, mode, ID, location):
        self.__ID = ID
        self.__mode = mode.upper() # 2D or 3D
        self.__location = np.array(location)
        self.__visibility_angle = np.pi/2 # 80-90 degrees
        self.__sigma_0 = 50 # baseline error typically 10-50m
        self.__min_sigma= 1.0
        self.__k = 0.05 # scaling factor typically 0.01-0.05 m/km
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
        x = r*np.cos(theta%(2*np.pi))
        y = r*np.sin(theta%(2*np.pi))
        return np.array([x,y])
    @staticmethod
    def spherical_to_cartesian(position):
        r, theta, phi = position
        x = r*np.cos(theta)*np.sin(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(phi)
        return np.array([x,y,z])
    
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
        cart_u = cart_u.reshape(n,1)
        return np.linalg.norm(cart_u - cart_v, axis=0)
    
    # method to check whether satellite is visible by a radar station
    def check_visibility(self,satellite_position):
        if self.__mode == '2D':
            cart_rad = self.polar_to_cartesian(self.__location)
            cart_sat = self.polar_to_cartesian(satellite_position)
        else:
            cart_rad = self.spherical_to_cartesian(self.__location)
            cart_sat = self.spherical_to_cartesian(satellite_position)
            
        rad_to_sat = (cart_sat - cart_rad) # vector from radar station to satellite
        rad_to_sat /= np.linalg.norm(rad_to_sat) # normalise the vector
        rad_normal = cart_rad/np.linalg.norm(cart_rad) # normalise radar station vector
        cos_angle_difference = np.dot(rad_to_sat, rad_normal) #cosine of angle between radar and satellite
        return cos_angle_difference >= np.cos(self.__visibility_angle)
        
    # method to record satellites position at a time, if not visible then position recorded as 0,0
    def record_satellite(self, time, satellite_position):
        if self.__mode == '2D':
            r, theta = satellite_position
            theta = theta%(2*np.pi)
        elif self.__mode == '3D':
            t, theta, phi = satellite_position
            theta = theta%(2*np.pi)
            phi = phi%(2*np.pi)
            if phi > np.pi:
                phi = 2*np.pi - phi
                
        # check if satellite is visible by radar station
        visible = self.check_visibility(satellite_position)
        if not visible:
            r,theta, phi = np.nan, np.nan, np.nan
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
        if self.__mode =='3D':
            phi_sat = self.satellite_measurements['phi']
            sat_positions = np.array([r, theta_sat, phi_sat])
        else:
            sat_positions = np.array([r, theta_sat])
            
        distance = self.distance(self.__mode, self.__location, sat_positions) #np.sqrt(r*r + R*R - 2*r*R*np.cos(theta_rad - theta_sat))
        eps_std = np.maximum(self.__sigma_0 - self.__k * distance, self.__min_sigma)
        eps = np.random.normal(0, eps_std)
        r += eps # only add noise to the radial distance
        self.__noise = eps #save the noise vector
        self.satellite_measurements['r'] = r
        self.satellite_measurements['theta'] = theta_sat
        
