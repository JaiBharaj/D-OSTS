import time
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Wedge
from CrudeInitialConditions import InitialConditions
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import threading
from queue import Queue, Empty
from collections import deque

class Visualiser2D:
    def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_map, break_point=0, mode='prewritten'):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt
        self.current_predict_t = 0.0
        self.focus_target = 'true'  # or 'predicted'
        self.zoom_factor = 1.0  # 1.0 is default, <1 is zoom in, >1 is zoom out
        self.mode = mode

        # File paths
        self.TRAJECTORY_FILE = trajectory_file_path
        self.PREDICTION_FILE = prediction_file_path
        self.HEATMAP_FILE = heatmap_file_map

        # Data storage
        self.trajectory = []
        self.predictions = []
        self.data_lock = threading.Lock()
        self.position_queue = Queue()
        self.prediction_queue = Queue()
        self.uncertainty_polygon = None
        self.crash_heatmap = None
        self.num_heatmap_bins = 36  # 10° per bin
        self.heatmap_file = "crash_angles.txt"

        # Setup figure
        self.setup_plots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def setup_plots(self):
        # Plot layout
        self.fig, (self.ax_full, self.ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.suptitle('Satellite Trajectory with Prediction Uncertainty', fontsize=14, y=0.98)

        # Configure axes
        for ax in (self.ax_full, self.ax_zoom):
            ax.set_aspect('equal')
            earth_circle = plt.Circle((0, 0), self.earth_radius, color='blue', alpha=0.3, zorder=1)
            ax.add_patch(earth_circle)
            ax.grid(True, alpha=0.3)

            # Add overlay with ticks and degree labels inside the circle
            self.add_radial_ticks_and_labels(ax)

        # Set full view limits
        plot_radius = 7E+6
        self.ax_full.set_xlim(-plot_radius, plot_radius)
        self.ax_full.set_ylim(-plot_radius, plot_radius)
        self.ax_full.set_title('Full Trajectory View')
        self.ax_full.set_xlabel('X position (m)')
        self.ax_full.set_ylabel('Y position (m)')

        # Initialise true elements
        self.trajectory_line_full, = self.ax_full.plot([], [], 'r-', lw=1.5, zorder=3, label='Actual')
        self.satellite_dot_full, = self.ax_full.plot([], [], 'ro', markersize=5, zorder=4)
        self.trajectory_line_zoom, = self.ax_zoom.plot([], [], 'r-', lw=1.5, zorder=3)
        self.satellite_dot_zoom, = self.ax_zoom.plot([], [], 'ro', markersize=5, zorder=4)

        # Prediction elements
        self.pred_line_full, = self.ax_full.plot([], [], 'b-', lw=1.2, alpha=0.9, zorder=5, label='Predicted')
        self.pred_dot_full, = self.ax_full.plot([], [], 'bo', markersize=5, zorder=4)
        self.pred_line_zoom, = self.ax_zoom.plot([], [], 'b-', lw=1.2, alpha=0.9, zorder=5)
        self.pred_dot_zoom, = self.ax_zoom.plot([], [], 'bo', markersize=5, zorder=4)
        self.pred_measurements_zoom, = self.ax_zoom.plot([], [], 'go', markersize=6, zorder=6, label='Measurements')

        # Altitude display
        self.altitude_text = self.ax_zoom.text(0.02, 0.98, "Altitude: Initializing...",
                                               transform=self.ax_zoom.transAxes, ha='left', va='top',
                                               fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

        self.ax_zoom.set_title('Zoomed View with Uncertainty')
        self.ax_zoom.legend(loc='upper right')

    def add_radial_ticks_and_labels(self, ax):
        num_ticks = 12

        for i in range(num_ticks):
            angle_deg = i * (360 / num_ticks)
            angle_rad = np.deg2rad(angle_deg)

            # Coordinates for the tick position on the edge of the circle
            x_tick = self.earth_radius * np.cos(angle_rad)
            y_tick = self.earth_radius * np.sin(angle_rad)

            # Draw tick
            tick_length = - 0.01 * self.earth_radius  # -1.0 for insisde
            tick_x = [x_tick, x_tick + tick_length * np.cos(angle_rad)]
            tick_y = [y_tick, y_tick + tick_length * np.sin(angle_rad)]
            ax.plot(tick_x, tick_y, color='black', lw=2, zorder=2)

            # Add angle label
            label_x = self.earth_radius * 0.85 * np.cos(angle_rad)
            label_y = self.earth_radius * 0.85 * np.sin(angle_rad)
            ax.text(label_x, label_y, f'{int(angle_deg)}°', color='black', ha='center', va='center', fontsize=10)

    def on_key_press(self, event):
        if event.key == 't':
            self.focus_target = 'true'
            print("Focusing on true trajectory")
        elif event.key == 'p':
            self.focus_target = 'predicted'
            print("Focusing on predicted trajectory")
        elif event.key in ['+', 'up']:
            self.zoom_factor *= 0.9  # Zoom in
            print(f"Zoom factor: {self.zoom_factor:.2f}")
        elif event.key in ['-', 'down']:
            self.zoom_factor *= 1.1  # Zoom out
            print(f"Zoom factor: {self.zoom_factor:.2f}")

    def read_next_position(self):
        with open(self.TRAJECTORY_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        _, r, theta = map(float, line.strip().split())
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        yield x, y
                        # time.sleep(0.05)  # Simulate streaming delay
                    except ValueError:
                        continue
            else:  # 'realtime'
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(pos)
                        # time.sleep(0.01)
                        continue
                    try:
                        _, r, theta = map(float, line.strip().split())
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        yield x, y
                    except ValueError:
                        continue

    def read_next_prediction(self):
        with open(self.PREDICTION_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        t_pred, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        self.current_predict_t = t_pred
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield x, y, std_x, std_y, int(is_meas)
                        # time.sleep(0.05)
                    except ValueError:
                        continue
            else:  # realtime mode
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(pos)
                        # time.sleep(0.01)
                        continue
                    try:
                        t_pred, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        self.current_predict_t = t_pred
                        x = r * np.cos(theta)
                        y = r * np.sin(theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield x, y, std_x, std_y, int(is_meas)
                    except ValueError:
                        continue

    def load_data(self):
        pos_gen = self.read_next_position()
        pred_gen = self.read_next_prediction()

        while True:
            try:
                self.position_queue.put(next(pos_gen))
                self.prediction_queue.put(next(pred_gen))
            except StopIteration:
                pass
            time.sleep(0.00001)

    # Estimate local direction of motion from trajectory
    def direction_of_motion(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        norm = np.hypot(dx, dy)
        if norm == 0:
            return np.array([1, 0]), np.array([0, 1])  # fallback
        tangent = np.array([dx, dy]) / norm
        normal = np.array([-tangent[1], tangent[0]])
        return tangent, normal

    def update(self, frame):
        MAX_STEPS = 50000

        artists = [self.trajectory_line_full, self.satellite_dot_full,
                   self.trajectory_line_zoom, self.satellite_dot_zoom,
                   self.pred_line_full, self.pred_dot_full,
                   self.pred_line_zoom, self.pred_dot_zoom,
                   self.pred_measurements_zoom, self.altitude_text]

        try:
            # Drain all new position data
            while True:
                try:
                    current_pos = self.position_queue.get_nowait()
                    x, y = current_pos
                    self.trajectory.append((x, y))
                    if len(self.trajectory) > MAX_STEPS:
                        self.trajectory = self.trajectory[-MAX_STEPS:]
                except Empty:
                    break

            # Update actual trajectory
            if self.trajectory:
                xs, ys = zip(*self.trajectory)
                x, y = self.trajectory[-1]

                self.trajectory_line_full.set_data(xs, ys)
                self.satellite_dot_full.set_data([x], [y])
                self.trajectory_line_zoom.set_data(xs, ys)
                self.satellite_dot_zoom.set_data([x], [y])

                current_dist = np.hypot(x, y)
                focus = None
                if self.focus_target == 'true' and self.trajectory:
                    focus = self.trajectory[-1]
                elif self.focus_target == 'predicted' and self.predictions:
                    focus = (self.predictions[-1][0], self.predictions[-1][1])

                if focus:
                    fx, fy = focus
                    dist = np.hypot(fx, fy)
                    base_width = max(500000, dist / 3)
                    zoom_width = base_width * self.zoom_factor
                    self.ax_zoom.set_xlim(fx - zoom_width / 6, fx + zoom_width / 6)
                    self.ax_zoom.set_ylim(fy - zoom_width / 6, fy + zoom_width / 6)

                altitude = current_dist - self.earth_radius
                self.altitude_text.set_text(f"Altitude: {altitude / 1000:.1f} km\n"
                                            f"Distance: {current_dist / 1000:.1f} km")

                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(f"IMPACT!\nFinal altitude: {altitude:.1f} m")
                    return artists

            # Drain all new prediction data
            while True:
                try:
                    current_pred = self.prediction_queue.get_nowait()
                    self.predictions.append(current_pred)
                    if len(self.predictions) > MAX_STEPS:
                        self.predictions = self.predictions[-MAX_STEPS:]
                except Empty:
                    break

            # Update predicted trajectory with uncertainty
            if self.predictions:
                pred_xs, pred_ys, std_xs, std_ys, meas_flags = zip(*self.predictions)
                pred_x, pred_y = pred_xs[-1], pred_ys[-1]

                self.pred_line_full.set_data(pred_xs, pred_ys)
                self.pred_dot_full.set_data([pred_x], [pred_y])
                self.pred_line_zoom.set_data(pred_xs, pred_ys)
                self.pred_dot_zoom.set_data([pred_x], [pred_y])

                # Remove old uncertainty polygon
                if self.uncertainty_polygon is not None:
                    self.uncertainty_polygon.remove()

                # Create new rotated uncertainty polygon
                polygon_points = []

                def get_direction(i, xs, ys):
                    if 0 < i < len(xs) - 1:
                        dx = xs[i + 1] - xs[i - 1]
                        dy = ys[i + 1] - ys[i - 1]
                    elif i > 0:
                        dx = xs[i] - xs[i - 1]
                        dy = ys[i] - ys[i - 1]
                    elif i < len(xs) - 1:
                        dx = xs[i + 1] - xs[i]
                        dy = ys[i + 1] - ys[i]
                    else:
                        return np.array([1.0, 0.0]), np.array([0.0, 1.0])  # fallback

                    norm = np.hypot(dx, dy)
                    if norm == 0:
                        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
                    tangent = np.array([dx, dy]) / norm
                    normal = np.array([-tangent[1], tangent[0]])
                    return tangent, normal

                # Forward pass
                for i in range(len(pred_xs)):
                    x, y = pred_xs[i], pred_ys[i]
                    sx, sy = std_xs[i], std_ys[i]
                    t_hat, n_hat = get_direction(i, pred_xs, pred_ys)
                    offset = sx * t_hat + sy * n_hat
                    polygon_points.append((x - offset[0], y - offset[1]))

                # Reverse pass
                for i in reversed(range(len(pred_xs))):
                    x, y = pred_xs[i], pred_ys[i]
                    sx, sy = std_xs[i], std_ys[i]
                    t_hat, n_hat = get_direction(i, pred_xs, pred_ys)
                    offset = sx * t_hat + sy * n_hat
                    polygon_points.append((x + offset[0], y + offset[1]))

                self.uncertainty_polygon = Polygon(polygon_points, closed=True,
                                                   color='blue', alpha=0.15, zorder=2)
                self.ax_zoom.add_patch(self.uncertainty_polygon)
                artists.append(self.uncertainty_polygon)

                # Update measurement points
                meas_xs = [x for x, flag in zip(pred_xs, meas_flags) if flag]
                meas_ys = [y for y, flag in zip(pred_ys, meas_flags) if flag]
                self.pred_measurements_zoom.set_data(meas_xs, meas_ys)

            # Update crash-site heatmap
            if os.path.exists(self.HEATMAP_FILE):
                with open(self.HEATMAP_FILE, 'r') as f:
                    angles = []
                    for line in f:
                        parts = line.strip().split()
                        angles.extend(float(p) % (2 * np.pi) for p in parts[1:] if p)

                if float(parts[0]) == self.current_predict_t:
                    if angles:
                        bin_counts, bin_edges = np.histogram(angles, bins=self.num_heatmap_bins, range=(0, 2 * np.pi))
                        bin_angles = (bin_edges[:-1] + bin_edges[1:]) / 2
                        max_count = bin_counts.max() or 1

                        if self.crash_heatmap is not None:
                            for patch in self.crash_heatmap:
                                patch.remove()
                        self.crash_heatmap = []

                        for count, angle in zip(bin_counts, bin_angles):
                            alpha = count / max_count
                            wedge = Wedge(
                                center=(0, 0),
                                r=self.earth_radius * 1.05,
                                theta1=np.degrees(angle - np.pi / self.num_heatmap_bins),
                                theta2=np.degrees(angle + np.pi / self.num_heatmap_bins),
                                width=0.02 * self.earth_radius,
                                facecolor='orange',
                                alpha=alpha,
                                zorder=1.5
                            )
                            self.ax_full.add_patch(wedge)
                            self.crash_heatmap.append(wedge)
                            artists.append(wedge)

        except Exception as e:
            print(f"Animation error: {e}")

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=1000,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

class Visualiser3D:
    def __init__(self, trajectory_file_path, prediction_file_path, break_point=0, mode='prewritten', MAX_STEPS=50000):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt
        self.user_controlled = False
        self.last_azim = None
        self.last_elev = None
        self.focus_on = 'true'  # Options: 'true' or 'predicted'
        self.zoom_scale = 1.0  # This is used in the update method
        self.zoom_factor = 1.0  # This is what your key handler modifies
        self.mode = mode

        def on_draw(event):
            if self.last_azim is not None and self.last_elev is not None:
                current_azim = self.ax_zoom.azim
                current_elev = self.ax_zoom.elev
                if abs(current_azim - self.last_azim) > 1 or abs(current_elev - self.last_elev) > 1:
                    self.user_controlled = True

            self.last_azim = self.ax_zoom.azim
            self.last_elev = self.ax_zoom.elev

        # File paths
        self.TRAJECTORY_FILE = trajectory_file_path
        self.PREDICTION_FILE = prediction_file_path

        # Data storage
        self.trajectory = deque(maxlen=MAX_STEPS)
        self.predictions = deque(maxlen=MAX_STEPS)
        self.data_lock = threading.Lock()
        self.new_position = None
        self.new_prediction = None
        self.uncertainty_polygon = None

        # Setup figure
        self.setup_plots()
        self.ax_zoom.figure.canvas.mpl_connect('draw_event', on_draw)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def setup_plots(self):
        # Create 3D figure with two subplots
        self.fig = plt.figure(figsize=(14, 7))
        self.fig.suptitle('3D Satellite Trajectory with Prediction Uncertainty', fontsize=14, y=0.98)

        # Full view 3D plot
        self.ax_full = self.fig.add_subplot(121, projection='3d')
        self.ax_full.set_title('Full 3D Trajectory View')
        self.ax_full.set_xlabel('X position (m)')
        self.ax_full.set_ylabel('Y position (m)')
        self.ax_full.set_zlabel('Z position (m)')

        # Zoomed view 3D plot
        self.ax_zoom = self.fig.add_subplot(122, projection='3d')
        self.ax_zoom.set_title('Zoomed 3D View with Uncertainty')
        self.ax_zoom.set_xlabel('X position (m)')
        self.ax_zoom.set_ylabel('Y position (m)')
        self.ax_zoom.set_zlabel('Z position (m)')

        # Create Earth sphere for both views
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot Earth in both views
        self.ax_full.plot_surface(x, y, z, color='gray', alpha=0.3)
        self.ax_zoom.plot_surface(x, y, z, color='gray', alpha=0.3)

        # Add longitude and latitude lines
        self.add_earth_grid(self.ax_full)
        self.add_earth_grid(self.ax_zoom)

        # Set full view limits
        plot_radius = 7E+6
        self.ax_full.set_xlim(-plot_radius, plot_radius)
        self.ax_full.set_ylim(-plot_radius, plot_radius)
        self.ax_full.set_zlim(-plot_radius, plot_radius)

        # Initialise true elements
        self.trajectory_line_full, = self.ax_full.plot([], [], [], 'r-', lw=1.5, zorder=3, label='Actual')
        self.satellite_dot_full, = self.ax_full.plot([], [], [], 'ro', markersize=5, zorder=4)
        self.trajectory_line_zoom, = self.ax_zoom.plot([], [], [], 'r-', lw=1.5, zorder=3)
        self.satellite_dot_zoom, = self.ax_zoom.plot([], [], [], 'ro', markersize=5, zorder=4)

        # Prediction elements
        self.pred_line_full, = self.ax_full.plot([], [], [], 'b-', lw=1.2, alpha=0.9, zorder=5, label='Predicted')
        self.pred_dot_full, = self.ax_full.plot([], [], [], 'bo', markersize=5, zorder=4)
        self.pred_line_zoom, = self.ax_zoom.plot([], [], [], 'b-', lw=1.2, alpha=0.9, zorder=5)
        self.pred_dot_zoom, = self.ax_zoom.plot([], [], [], 'bo', markersize=5, zorder=4)
        self.pred_measurements_zoom, = self.ax_zoom.plot([], [], [], 'go', markersize=6, zorder=6, label='Measurements')

        # Altitude display
        self.altitude_text = self.fig.text(0.7, 0.05, "Altitude: Initializing...",
                                           fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

        # Add legends
        self.ax_full.legend(loc='upper right')
        self.ax_zoom.legend(loc='upper right')

    def add_earth_grid(self, ax):
        # Longitude lines
        theta = np.linspace(0, 2 * np.pi, 13)
        phi = np.linspace(0, np.pi, 100)

        for t in theta[:-1]:
            x = self.earth_radius * np.cos(t) * np.sin(phi)
            y = self.earth_radius * np.sin(t) * np.sin(phi)
            z = self.earth_radius * np.cos(phi)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Latitude lines
        phi = np.linspace(0, np.pi, 7)
        theta = np.linspace(0, 2 * np.pi, 100)

        for p in phi[1:-1]:
            x = self.earth_radius * np.cos(theta) * np.sin(p)
            y = self.earth_radius * np.sin(theta) * np.sin(p)
            z = self.earth_radius * np.cos(p) * np.ones_like(theta)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Labels for longitudinal lines
        label_angles = [0, 90, 180, 270]
        for angle in label_angles:
            rad = np.deg2rad(angle)
            x = self.earth_radius * 1.05 * np.cos(rad)
            y = self.earth_radius * 1.05 * np.sin(rad)
            z = 0
            ax.text(x, y, z, f'{angle}°', color='black', fontsize=8, ha='center', va='center')

        # Latitude labels
        label_lats = [0, 30, 60, -30, -60]
        for lat in label_lats:
            rad = np.deg2rad(lat)
            z = self.earth_radius * np.sin(rad)
            r = self.earth_radius * np.cos(rad)
            x = r * 1.05
            y = 0
            ax.text(x, y, z, f'{lat}°', color='black', fontsize=8, ha='center', va='center')

    def on_key_press(self, event):
        if event.key == 't':
            self.focus_on = 'true'  # Fixed variable name (was focus_target)
            print("Focusing on true trajectory")
        elif event.key == 'p':
            self.focus_on = 'predicted'  # Fixed variable name (was focus_target)
            print("Focusing on predicted trajectory")
        elif event.key in ['+', 'up']:
            self.zoom_factor *= 0.9  # Zoom in
            print(f"Zoom factor: {self.zoom_factor:.2f}")
        elif event.key in ['-', 'down']:
            self.zoom_factor *= 1.1  # Zoom out
            print(f"Zoom factor: {self.zoom_factor:.2f}")
        # Update zoom_scale with zoom_factor
        self.zoom_scale = self.zoom_factor
        self.fig.canvas.draw_idle()  # Force redraw

    def read_next_position(self):
        with open(self.TRAJECTORY_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        _, r, theta, phi = map(float, line.strip().split())
                        x = r * np.sin(phi) * np.cos(theta)  # phi is polar, theta is azimuthal
                        y = r * np.sin(phi) * np.sin(theta)
                        z = r * np.cos(phi)
                        yield x, y, z
                        # time.sleep(0.05)  # Simulate streaming delay
                    except ValueError:
                        continue
            else:  # 'realtime'
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(pos)
                        # time.sleep(0.01)
                        continue
                    try:
                        _, r, theta, phi = map(float, line.strip().split())
                        x = r * np.sin(theta) * np.cos(phi)
                        y = r * np.sin(theta) * np.sin(phi)
                        z = r * np.cos(theta)
                        yield x, y, z
                    except ValueError:
                        continue

    def read_next_prediction(self):
        def spherical_to_cartesian(r, theta, phi):
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            return x, y, z

        def spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi):
            # Jacobian-based approximation of standard deviations in Cartesian coords
            sx = np.sqrt(
                (np.sin(phi) * np.cos(theta) * dr) ** 2 +
                (r * np.cos(phi) * np.cos(theta) * dphi) ** 2 +
                (r * np.sin(phi) * np.sin(theta) * dtheta) ** 2
            )
            sy = np.sqrt(
                (np.sin(phi) * np.sin(theta) * dr) ** 2 +
                (r * np.cos(phi) * np.sin(theta) * dphi) ** 2 +
                (r * np.sin(phi) * np.cos(theta) * dtheta) ** 2
            )
            sz = np.sqrt(
                (np.cos(phi) * dr) ** 2 +
                (r * np.sin(phi) * dphi) ** 2
            )
            return sx, sy, sz

        with open(self.PREDICTION_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        _, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield x, y, z, std_x, std_y, std_z, int(is_meas)
                    except ValueError:
                        continue
            else:  # realtime mode
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(pos)
                        # time.sleep(0.01)
                        continue
                    try:
                        _, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield x, y, z, std_x, std_y, std_z, int(is_meas)
                    except ValueError:
                        continue

    def load_data(self):
        pos_gen = self.read_next_position()
        pred_gen = self.read_next_prediction()

        while True:
            try:
                with self.data_lock:
                    self.new_position = next(pos_gen)
                    self.new_prediction = next(pred_gen)
            except StopIteration:
                pass
            time.sleep(0.00001)

    def create_uncertainty_tube(self, points, std_devs):
        vertices = []
        faces = []

        # Vertices for each cross-section
        for i, ((x, y, z), (sx, sy, sz)) in enumerate(zip(points, std_devs)):
            theta = np.linspace(0, 2 * np.pi, 16)
            for angle in theta:
                px = x + sx * np.cos(angle)
                py = y + sy * np.sin(angle)
                pz = z
                vertices.append([px, py, pz])

        # Faces
        n_circle = 16
        n_points = len(points)
        for i in range(n_points - 1):
            for j in range(n_circle):
                j_next = (j + 1) % n_circle
                v1 = i * n_circle + j
                v2 = i * n_circle + j_next
                v3 = (i + 1) * n_circle + j_next
                v4 = (i + 1) * n_circle + j
                faces.append([v1, v2, v3, v4])

        return vertices, faces

    def update(self, frame):
        artists = [self.trajectory_line_full, self.satellite_dot_full,
                   self.trajectory_line_zoom, self.satellite_dot_zoom,
                   self.pred_line_full, self.pred_dot_full,
                   self.pred_line_zoom, self.pred_dot_zoom,
                   self.pred_measurements_zoom, self.altitude_text]

        try:
            with self.data_lock:
                current_pos = self.new_position
                current_pred = self.new_prediction

            # Update true trajectory
            if current_pos:
                x, y, z = current_pos
                self.trajectory.append((x, y, z))

                # Convert trajectory to plottable arrays
                if len(self.trajectory) > 1:
                    xs, ys, zs = zip(*self.trajectory)
                else:
                    xs, ys, zs = [x], [y], [z]

                # Update true trajectory plots
                self.trajectory_line_full.set_data_3d(xs, ys, zs)
                self.satellite_dot_full.set_data_3d([x], [y], [z])
                self.trajectory_line_zoom.set_data_3d(xs, ys, zs)
                self.satellite_dot_zoom.set_data_3d([x], [y], [z])

                current_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                altitude = current_dist - self.earth_radius

                # Update camera view if not user-controlled
                if not self.user_controlled and len(self.trajectory) >= 2:
                    # Get last two positions to determine direction
                    (x_prev, y_prev, z_prev), (x_curr, y_curr, z_curr) = self.trajectory[-2], self.trajectory[-1]
                    dx, dy, dz = x_curr - x_prev, y_curr - y_prev, z_curr - z_prev

                    # Calculate azimuth and elevation from velocity vector
                    azim = np.degrees(np.arctan2(dy, dx))
                    elev = np.degrees(np.arctan2(dz, np.sqrt(dx ** 2 + dy ** 2)))
                    self.ax_zoom.view_init(elev=elev, azim=azim)

                # Update prediction visualization
                if current_pred:
                    pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas = current_pred
                    self.predictions.append((pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas))

                    if len(self.predictions) > 1:
                        # Extract prediction points and uncertainties
                        pred_points = np.array([(x, y, z) for x, y, z, _, _, _, _ in self.predictions])
                        pred_xs, pred_ys, pred_zs = pred_points.T
                        std_devs = [(sx, sy, sz) for _, _, _, sx, sy, sz, _ in self.predictions]
                        meas_flags = [m for _, _, _, _, _, _, m in self.predictions]

                        # Update prediction lines and dots
                        self.pred_line_full.set_data_3d(pred_xs, pred_ys, pred_zs)
                        self.pred_dot_full.set_data_3d([pred_x], [pred_y], [pred_z])
                        self.pred_line_zoom.set_data_3d(pred_xs, pred_ys, pred_zs)
                        self.pred_dot_zoom.set_data_3d([pred_x], [pred_y], [pred_z])

                        # Update uncertainty visualization
                        if hasattr(self, 'uncertainty_tube'):
                            self.uncertainty_tube.remove()

                        vertices, faces = self.create_uncertainty_tube(pred_points, std_devs)
                        vertices = np.array(vertices)
                        self.uncertainty_tube = Poly3DCollection(
                            [vertices[face] for face in faces],
                            alpha=0.15,
                            color='blue',
                            linewidths=0.5,
                            edgecolor='blue'
                        )
                        self.ax_zoom.add_collection3d(self.uncertainty_tube)
                        artists.append(self.uncertainty_tube)

                        # Update measurement points
                        if any(meas_flags):
                            meas_points = pred_points[np.array(meas_flags, dtype=bool)]
                            meas_xs, meas_ys, meas_zs = meas_points.T
                            self.pred_measurements_zoom.set_data_3d(meas_xs, meas_ys, meas_zs)

                # Update zoomed view limits
                focus_point = None
                if self.focus_on == 'true' and self.trajectory:
                    focus_point = self.trajectory[-1]
                elif self.focus_on == 'predicted' and self.predictions:
                    focus_point = (self.predictions[-1][0], self.predictions[-1][1], self.predictions[-1][2])

                if focus_point:
                    fx, fy, fz = focus_point
                    base_width = max(500000, np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) / 3)
                    zoom_width = base_width * self.zoom_scale

                    self.ax_zoom.set_xlim(fx - zoom_width / 2, fx + zoom_width / 2)
                    self.ax_zoom.set_ylim(fy - zoom_width / 2, fy + zoom_width / 2)
                    self.ax_zoom.set_zlim(fz - zoom_width / 2, fz + zoom_width / 2)

                # Update altitude text
                self.altitude_text.set_text(
                    f"Altitude: {altitude / 1000:.1f} km\n"
                    f"Distance: {current_dist / 1000:.1f} km\n"
                    f"Position: ({x / 1000:.1f}, {y / 1000:.1f}, {z / 1000:.1f}) km"
                )

                # Check for impact
                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(
                        f"IMPACT!\n"
                        f"Final altitude: {altitude:.1f} m\n"
                        f"Final Position: ({x / 1000:.1f}, {y / 1000:.1f}, {z / 1000:.1f}) km"
                    )
                    return artists

        except Exception as e:
            print(f"Animation error: {e}")
            import traceback
            traceback.print_exc()

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=1000,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# vis = Visualiser2D('trajectory.txt', 'predicted_trajectory.txt', break_point=100)
# vis.visualise()

# vis = Visualiser3D('trajectory_3d.txt', 'predicted_trajectory_3d.txt', break_point=0, MAX_STEPS=100)
# vis.visualise()
