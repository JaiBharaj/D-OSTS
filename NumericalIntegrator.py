import numpy as np
from CrudeInitialConditions import InitialConditions
from CoordinateTransformations import PolarAccelerations, SphericalAccelerations, BonusAccelerations
from scipy.integrate import solve_ivp

class Integrator:

    def __init__(self):
        self.Re = InitialConditions.earthRadius
        self.alt0 = InitialConditions.initSatAlt
        self.r0 = self.Re + self.alt0
        self.theta0 = InitialConditions.initSatTheta
        self.phi0 = InitialConditions.initSatPhi
        self.lam0 = InitialConditions.initSatLam
        self.r_dot0 = InitialConditions.initSatRdot
        self.phi_dot0 = InitialConditions.initSatPhidot
        self.mu = InitialConditions.gravConstant * InitialConditions.earthMass

        v_circ = np.sqrt(self.mu / self.r0)
        v_target = v_circ - InitialConditions.deltaV
        term_sq = v_target ** 2 - (self.r0 * self.phi_dot0) ** 2
        if term_sq < 0:
            raise ValueError("phi_dot0 too large for given Δv or orbit height")
        self.lam_dot0 = - np.sqrt(term_sq) / (self.r0 * np.sin(self.phi0))
        self.y0 = np.array([
            self.r0,      # r
            self.r_dot0,  # ṙ
            self.phi0,    # φ
            self.phi_dot0,# φ̇
            self.lam0,    # λ
            self.lam_dot0 # λ̇
        ], dtype=float)

        # Bonus
        self.delta_v_thrust = InitialConditions.deltaV_from_thrust
        self.h_thrust = InitialConditions.hThrust
        self.populated_radius = InitialConditions.populatedRadius
        self.populated_centers = InitialConditions.populatedCenters

    @staticmethod
    def rhs_polar(t, y):
        r, r_dot, theta, theta_dot = y
        return PolarAccelerations.accelerations(r, r_dot, theta, theta_dot)

    @staticmethod
    def rhs_spherical(t, y):
        r, r_dot, phi, phi_dot, lam, lam_dot = y
        return SphericalAccelerations.accelerations(r, r_dot, phi, phi_dot, lam, lam_dot)
    
    def hit_ground(self):
        def events(t, y):
            return y[0] - self.Re
        events.terminal = True
        events.direction = -1
        return events
    
    def at_thrust(self):
        def events(t, y):
            return y[0] - (self.Re + self.h_thrust)
        events.terminal = True
        events.direction = -1
        return events

    def runge_kutta45_2d(self):
        v0 = np.sqrt(self.mu / self.r0)
        theta_dot0 = (v0 - 80) / self.r0

        y0 = np.array([
            self.r0,
            self.r_dot0,
            self.theta0,
            theta_dot0], dtype=float)

        sol = solve_ivp(
            Integrator.rhs_polar,
            (0.0, 1.3e8),
            y0,
            rtol=1e-6,
            atol=1e-8,
            method="RK45",
            first_step=1.0,
            max_step=5.0,
            events=self.hit_ground())

        return sol
    
    def runge_kutta45_3d(self):
        sol = solve_ivp(
            Integrator.rhs_spherical,
            (0.0, 1.3e8),
            self.y0,
            method="RK45",
            rtol=1e-7, 
            atol=1e-9,
            first_step=1.0, 
            max_step=1.5,
            events=self.hit_ground())

        return sol

    def downsample_indices(n_total, n_target):
        if n_target >= n_total:
            return np.arange(n_total)
        return np.linspace(0, n_total - 1, n_target, dtype=int)
    
    @staticmethod
    def sph_to_cart(r, phi, lam):
        x = r*np.sin(phi)*np.cos(lam)
        y = r*np.sin(phi)*np.sin(lam)
        z = r*np.cos(phi)
        return np.array([x, y, z])

    @staticmethod
    def cart_to_sph(vec):
        x, y, z = vec
        r   = np.linalg.norm(vec)
        phi = np.arccos(z / r)
        lam = np.arctan2(y, x)
        return r, phi, lam
    
    @staticmethod
    def great_circle_distance(phi1, lam1, phi2, lam2):
        Δσ = np.arccos(
            np.sin(phi1)*np.sin(phi2) +
            np.cos(phi1)*np.cos(phi2)*np.cos(lam1-lam2)
        )
        return InitialConditions.earthRadius * Δσ 
    
    @staticmethod
    def in_populated(phi, lam):
        for pc in InitialConditions.populatedCenters:
            if Integrator.great_circle_distance(phi, lam, pc[0], pc[1]) < InitialConditions.populatedRadius:
                return True
        return False

    def get_trajectory_2d(self, n_save=20_000):
        res = self.runge_kutta45_2d()
        print(res.t_events[0])

        t_arr = res.t
        r_arr = res.y[0]
        th_arr = res.y[2]

        # -------- Optional ----------
        # keep = self.downsample_indices(len(t_arr), n_save)
        # t_arr, r_arr, th_arr = t_arr[keep], r_arr[keep], th_arr[keep]

        # -------- File output ----------
        true_file = 'trajectory_without_noise.txt'

        with open(true_file, 'w') as f_true:
            for (t, r, th) in zip(t_arr, r_arr, th_arr):
                f_true.write(f"{t} {r} {th}\n")

    def get_trajectory_3d(self, n_save=20_000, to_xyz=False, bonus=False):
        # traj without thrust
        sol0 = self.runge_kutta45_3d()
        r_imp, phi_imp, lam_imp = sol0.y[0,-1], sol0.y[2,-1], sol0.y[4,-1]
        
        # use thrust or not
        if Integrator.in_populated(phi_imp, lam_imp) or bonus:
            print("Need thrust")
            # integrate to thrust height
            sol1 = solve_ivp(
                Integrator.rhs_spherical, 
                (0, 1.3e8), 
                self.y0,
                rtol=1e-7, 
                atol=1e-9, 
                first_step=1.0, 
                max_step=1.5,
                method='RK45',
                events=self.at_thrust())
            y_thrust = sol1.y[:, -1]
            print("thrust: ", y_thrust)

            # velocity vector plus delta v
            r, vr, phi, vphi, lam, vlam = y_thrust
            e_r     = Integrator.sph_to_cart(1, phi, lam)
            e_phi   = Integrator.sph_to_cart(1, phi+np.pi/2, lam)
            e_lam   = Integrator.sph_to_cart(1, np.pi/2, lam+np.pi/2)
            v_vec = vr*e_r + r*vphi*e_phi + r*np.sin(phi)*vlam*e_lam
            v_hat = v_vec / np.linalg.norm(v_vec)
            v_vec_new = v_vec + self.delta_v_thrust * v_hat

            # new velocity after thrust
            vr_new   = np.dot(v_vec_new, e_r)
            vphi_new = np.dot(v_vec_new, e_phi) / r
            vlam_new = np.dot(v_vec_new, e_lam) / (r*np.sin(phi))
            y_thrust[1] = vr_new
            y_thrust[3] = vphi_new
            y_thrust[5] = vlam_new

            # continue integrating to ground
            sol2 = solve_ivp(
                Integrator.rhs_spherical, 
                (sol1.t[-1], 1.3e8),
                y_thrust, 
                rtol=1e-7, 
                atol=1e-9, 
                first_step=1.0, 
                max_step=1.5,
                method='RK45',
                events=self.hit_ground())
            
            # concat two parts
            t_all = np.hstack([sol1.t[:-1], sol2.t])
            y_all = np.hstack([sol1.y[:, :-1], sol2.y])
            sol_all = sol2
            sol_all.t, sol_all.y = t_all, y_all
            res = sol_all
        else:
            print("Don't need thrust")
            res = sol0

        # get trajectory
        t_arr  = res.t
        r_arr  = res.y[0]
        phi_arr = res.y[2]
        lam_arr = res.y[4]

        # ------ option ------
        # keep = self.downsample_indices(len(t_arr), n_save)
        # t_arr, r_arr, phi_arr, lam_arr = t_arr[keep], r_arr[keep], phi_arr[keep], lam_arr[keep]

        fname = 'trajectory_without_noise_3d_xyz.txt' if to_xyz else 'trajectory_without_noise_3d.txt'
        with open(fname, 'w') as f:
            if to_xyz:
                x = r_arr * np.sin(phi_arr) * np.cos(lam_arr)
                y = r_arr * np.sin(phi_arr) * np.sin(lam_arr)
                z = r_arr * np.cos(phi_arr)
                for t, xi, yi, zi in zip(t_arr, x, y, z):
                    f.write(f"{t:.3f} {xi:.3f} {yi:.3f} {zi:.3f}\n")
            else:
                for t, r, ph, la in zip(t_arr, r_arr, phi_arr, lam_arr):
                    f.write(f"{t:.3f} {r:.3f} {ph:.6f} {la:.6f}\n")

        print(f"Wrote {len(t_arr)} points to {fname}")


# ------ An example to use this class ------

# from NumericalIntegrator import Integrator
# from CrudeInitialConditions import InitialConditions

# integrator = Integrator()
# integrator.get_trajectory_3d()

# ------ Then you will get a text file named 'trajectory_without_noise_3d.txt' in same path ------
