import numpy as np
from Atmospheric_Density import atmos_ussa1976_rho as atmospheric_density
from CrudeInitialConditions import InitialConditions

class PolarAccelerations:

    # Polar Velocity Magnitude
    @staticmethod
    def comb_velocity(rad, radVel, angVel):
        return np.sqrt(radVel**2 + (rad * angVel)**2)

    # First part of Drag term: consistent for Spherical coordinates
    @staticmethod
    def drag_start(C_d, A, m, r):
        Re = 6371e3
        h = r - Re
        result = 0.5 * atmospheric_density(h) * C_d * A / m
        return result

    # Polar Acceleration functions to be passed into RK45
    @staticmethod
    def accelerations(u1, u2, u3, u4):

        '''
        :param u1: Radial Position
        :param u2: Radial Velocity
        :param u3: Angular Position
        :param u4: Angular Velocity

        :return: [Radial Velocity, Radial Acceleration, Angular Velocity, Angular Acceleration]
        '''

        dt_start = PolarAccelerations.drag_start(InitialConditions.dragCoeff,
                                                 InitialConditions.crossSec,
                                                 InitialConditions.satMass,
                                                 u1)

        vel_mag = PolarAccelerations.comb_velocity(u1, u2, u4)

        u1_dot = u2
        u2_dot = u1 * u4**2 - InitialConditions.gravConstant * InitialConditions.earthMass / u1**2 - dt_start * u2 * vel_mag
        u3_dot = u4
        u4_dot = - 2 * u2 * u4 / u1 - dt_start * u4 * vel_mag

        return np.array([u1_dot, u2_dot, u3_dot, u4_dot])

class SphericalAccelerations(PolarAccelerations):

    @staticmethod
    def comb_velocity(rad, radVel, polAng, polVel, aziVel):
        return np.sqrt(radVel**2 + rad**2 * (polVel**2 + np.sin(polAng)**2 * aziVel**2))

    # Spherical Acceleration functions to be passed into RK45
    @staticmethod
    def accelerations(u1, u2, u3, u4, u5, u6):
        '''
        :param u1: Radial Position
        :param u2: Radial Velocity
        :param u3: Polar Angular Position
        :param u4: Polar Angular Velocity
        :param u5: Azimuthal Angular Position
        :param u6: Azimuthal Angular Velocity

        :return: [Radial Velocity, Radial Acceleration, Polar Angular Velocity, Polar Angular Acceleration, Azimuthal Angular Velocity, Azimuthal Angular Acceleration]
        '''

        dt_start = SphericalAccelerations.drag_start(C_d=InitialConditions.dragCoeff,
                                                 A=InitialConditions.crossSec,
                                                 m=InitialConditions.satMass)

        vel_mag = SphericalAccelerations.comb_velocity(u1, u2, u3, u4, u6)

        u1_dot = u2
        u2_dot = u1 * (u4**2 + np.sin(u3)**2 * u6**2) - InitialConditions.gravConstant * InitialConditions.earthMass / u1**2 - dt_start * u2 * vel_mag
        u3_dot = u4
        u4_dot = - 2 * u2 * u4 / u1 + np.sin(u3) * np.cos(u3) * u6**2 - dt_start * u4 * vel_mag
        u5_dot = u6
        u6_dot = - 2 * u2 * u6 / u1 - 2 * (1 / np.tan(u3)) * u4 * u6 - dt_start * u6 * vel_mag

        return np.array([u1_dot, u2_dot, u3_dot, u4_dot, u5_dot, u6_dot])
