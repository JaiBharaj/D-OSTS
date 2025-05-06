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
                                                     m=InitialConditions.satMass,
                                                     r=u1)

        vel_mag = SphericalAccelerations.comb_velocity(u1, u2, u3, u4, u6)

        u1_dot = u2
        u2_dot = u1 * (u4**2 + np.sin(u3)**2 * u6**2) - InitialConditions.gravConstant * InitialConditions.earthMass / u1**2 - dt_start * u2 * vel_mag
        u3_dot = u4
        u4_dot = - 2 * u2 * u4 / u1 + np.sin(u3) * np.cos(u3) * u6**2 - dt_start * u4 * vel_mag
        u5_dot = u6
        u6_dot = - 2 * u2 * u6 / u1 - 2 * (1 / np.tan(u3)) * u4 * u6 - dt_start * u6 * vel_mag

        return np.array([u1_dot, u2_dot, u3_dot, u4_dot, u5_dot, u6_dot])


class BonusAccelerations(SphericalAccelerations):
    def __init__(self, thrust_impulse=0.0, thrust_time=None):
        self.thrust_impulse = thrust_impulse  # Fixed impulse in Ns
        self.thrust_time = thrust_time  # Time at which to apply thrust
        self.thrust_applied = False  # Track if thrust has been used

    def accelerations(self, t, u1, u2, u3, u4, u5, u6):
        '''
        :param t: Current time (needed to check thrust timing)
        :param u1: Radial Position
        :param u2: Radial Velocity
        :param u3: Polar Angular Position
        :param u4: Polar Angular Velocity
        :param u5: Azimuthal Angular Position
        :param u6: Azimuthal Angular Velocity

        :return: [Radial Velocity, Radial Acceleration, Polar Angular Velocity,
                 Polar Angular Acceleration, Azimuthal Angular Velocity,
                 Azimuthal Angular Acceleration]
        '''
        dt_start = self.drag_start(C_d=InitialConditions.dragCoeff,
                                   A=InitialConditions.crossSec,
                                   m=InitialConditions.satMass,
                                   r=u1)

        vel_mag = self.comb_velocity(u1, u2, u3, u4, u6)

        # Calculate thrust acceleration if it's time to apply thrust
        thrust_accel = 0.0
        if (self.thrust_time is not None and
                t >= self.thrust_time and
                not self.thrust_applied and
                self.thrust_impulse > 0):

            # Calculate instantaneous velocity direction
            velocity_vector = np.array([
                u2,  # radial
                u1 * u4,  # polar
                u1 * np.sin(u3) * u6  # azimuthal
            ])

            # Normalise to get direction
            if np.linalg.norm(velocity_vector) > 0:
                thrust_direction = velocity_vector / np.linalg.norm(velocity_vector)
            else:
                thrust_direction = np.array([1, 0, 0])  # default direction if velocity is zero

            # Velocity-Impulse relation
            delta_v = self.thrust_impulse / InitialConditions.satMass

            # Apply the delta_v in the velocity direction
            # We model this as an instantaneous change in velocity components
            u2 += delta_v * thrust_direction[0]  # radial
            u4 += delta_v * thrust_direction[1] / u1  # polar
            if u1 * np.sin(u3) > 0:
                u6 += delta_v * thrust_direction[2] / (u1 * np.sin(u3))  # azimuthal

            self.thrust_applied = True
            print(f"Applied thrust at t={t}s, delta_vv={delta_v} m/s")

            # Recalculate velocity magnitude after thrust
            vel_mag = self.comb_velocity(u1, u2, u3, u4, u6)

        u1_dot = u2
        u2_dot = u1 * (u4 ** 2 + np.sin(
            u3) ** 2 * u6 ** 2) - InitialConditions.gravConstant * InitialConditions.earthMass / u1 ** 2 - dt_start * u2 * vel_mag
        u3_dot = u4
        u4_dot = -2 * u2 * u4 / u1 + np.sin(u3) * np.cos(u3) * u6 ** 2 - dt_start * u4 * vel_mag
        u5_dot = u6
        u6_dot = -2 * u2 * u6 / u1 - 2 * (1 / np.tan(u3)) * u4 * u6 - dt_start * u6 * vel_mag

        return np.array([u1_dot, u2_dot, u3_dot, u4_dot, u5_dot, u6_dot])