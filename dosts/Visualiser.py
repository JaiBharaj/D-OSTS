import time
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon, Wedge
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import threading
from queue import Queue, Empty
from collections import deque
from .CrudeInitialConditions import InitialConditions
from .CoordinateTransformations import polar_to_cartesian, spherical_to_cartesian

class Visualiser2D:
    def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_path=None, measurement_times=None, break_point=0, mode='prewritten'):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt
        self.focus_target = 'true'  # or 'predicted'
        self.zoom_factor = 1.0
        self.mode = mode
        self.measurement_times = measurement_times

        # File paths
        self.TRAJECTORY_FILE = trajectory_file_path
        self.PREDICTION_FILE = prediction_file_path
        self.HEATMAP_FILE = heatmap_file_path

        # Data storage
        self.trajectory = []
        self.predictions = []
        self.crash_angles_history = []
        self.data_lock = threading.Lock()
        self.position_queue = Queue()
        self.prediction_queue = Queue()
        self.uncertainty_polygon = None

        # Heatmap data storage
        self.heatmap_data = []  # List of (time, [angles]) tuples
        self.current_heatmap_time = 0
        self.heatmap_artists = []
        self.heatmap_cmap = plt.cm.viridis
        self.heatmap_cbar = None
        self.heatmap_resolution = 360
        self.heatmap_alpha = 0.7

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

            # Add heatmap to the full view
            self.heatmap_artists = []

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

        self.heatmap_cbar = None  # Ensure this is set
        self.init_colorbar()

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

    def init_colorbar(self):
        if self.heatmap_cbar is None:
            sm = plt.cm.ScalarMappable(cmap=self.heatmap_cmap,
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            self.heatmap_cbar = self.fig.colorbar(
                sm, ax=self.ax_full,
                orientation='vertical',
                pad=0.05,
                label='Crash Probability (relative)'
            )
            # Don't add to heatmap_artists so it persists

    def add_radial_ticks_and_labels(self, ax):
        num_ticks = 12

        for i in range(num_ticks):
            angle_deg = i * (360 / num_ticks)
            angle_rad = np.deg2rad(angle_deg)

            # Coordinates for the tick position on the edge of the circle
            x_tick, y_tick = polar_to_cartesian(self.earth_radius, angle_rad)

            # Draw tick
            tick_length = - 0.01 * self.earth_radius  # -1.0 for insisde
            tick_mult_x, tick_mult_y = polar_to_cartesian(tick_length, angle_rad)
            tick_x = [x_tick, x_tick + tick_mult_x]
            tick_y = [y_tick, y_tick + tick_mult_y]
            ax.plot(tick_x, tick_y, color='black', lw=2, zorder=2)

            # Add angle label
            label_x, label_y = polar_to_cartesian(self.earth_radius * 0.85, angle_rad)
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
                        x, y = polar_to_cartesian(r, theta)
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
                        x, y = polar_to_cartesian(r, theta)
                        yield x, y
                    except ValueError:
                        continue

    def read_next_prediction(self):
        with open(self.PREDICTION_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        time, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        x, y = polar_to_cartesian(r, theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield time, x, y, std_x, std_y, int(is_meas)
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
                        time, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        x, y = polar_to_cartesian(r, theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield time, x, y, std_x, std_y, int(is_meas)
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

    def load_heatmap_data(self):
        if not self.HEATMAP_FILE:
            return

        try:
            with open(self.HEATMAP_FILE, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    timestamp = float(parts[0])
                    angles = list(map(float, parts[1:]))
                    self.heatmap_data.append((timestamp, angles))
        except FileNotFoundError:
            print(f"Heatmap file {self.HEATMAP_FILE} not found")

    def update_heatmap(self, current_time):
        # Clear previous heatmap wedges (but keep colorbar)
        for artist in self.heatmap_artists:
            try:
                artist.remove()
            except ValueError:
                # Already removed or not in current figure
                pass
        self.heatmap_artists = []

        # Find all predictions made before current_time
        current_angles = []
        for t, angles in self.heatmap_data:
            if t <= current_time:
                current_angles.extend(angles)

        if not current_angles:
            return

        # Create histogram
        angles_rad = np.array(current_angles) % (2 * np.pi)
        hist, bin_edges = np.histogram(angles_rad, bins=self.heatmap_resolution, range=(0, 2 * np.pi))
        max_count = max(1, hist.max())

        # Create colored wedges
        for i in range(self.heatmap_resolution):
            if hist[i] == 0:
                continue

            start_angle = np.degrees(bin_edges[i])
            end_angle = np.degrees(bin_edges[i + 1])
            intensity = hist[i] / max_count

            wedge = Wedge(
                (0, 0),
                self.earth_radius,
                start_angle,
                end_angle,
                width=self.earth_radius * 0.05,
                color=self.heatmap_cmap(intensity),
                alpha=self.heatmap_alpha * intensity
            )
            self.ax_full.add_patch(wedge)
            self.heatmap_artists.append(wedge)

        # Update colorbar
        if self.heatmap_cbar:
            sm = plt.cm.ScalarMappable(cmap=self.heatmap_cmap,
                                       norm=plt.Normalize(vmin=0, vmax=max_count))
            sm.set_array([])
            self.heatmap_cbar.update_normal(sm)
        else:
            self.init_colorbar()  # Initialize if not exists

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
                    focus = (self.predictions[-1][1], self.predictions[-1][2])

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
                pred_ts, pred_xs, pred_ys, std_xs, std_ys, meas_flags = zip(*self.predictions)
                pred_x, pred_y = pred_xs[-1], pred_ys[-1]
                current_time = pred_ts[-1]

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
                    polygon_points.append((x - sx, y - sy))

                # Reverse pass
                for i in reversed(range(len(pred_xs))):
                    x, y = pred_xs[i], pred_ys[i]
                    sx, sy = std_xs[i], std_ys[i]
                    polygon_points.append((x + sx, y + sy))

                self.uncertainty_polygon = Polygon(polygon_points, closed=True,
                                                   color='blue', alpha=0.15, zorder=2)
                self.ax_zoom.add_patch(self.uncertainty_polygon)
                artists.append(self.uncertainty_polygon)

                # Update measurement points
                meas_xs = [x for x, flag in zip(pred_xs, meas_flags) if flag]
                meas_ys = [y for y, flag in zip(pred_ys, meas_flags) if flag]
                self.pred_measurements_zoom.set_data(meas_xs, meas_ys)

                # Update heatmap for current time
                self.update_heatmap(current_time)
                artists.extend(self.heatmap_artists)

                return artists

        except Exception as e:
            print(f"Animation error: {e}")

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Load heatmap data properly
        self.load_heatmap_data()

        # Load heatmap data if file exists
        if self.HEATMAP_FILE:
            try:
                with open(self.HEATMAP_FILE, 'r') as f:
                    for line in f:
                        samples = map(float, line.strip().split())
                        self.crash_angles_history.extend(samples)
            except FileNotFoundError:
                print(f"Heatmap file {self.HEATMAP_FILE} not found")

        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=1000,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        # Save as GIF using Pillow
        self.ani.save("visualiser2D.gif", writer=animation.PillowWriter(fps=20))

        print("GIF saved as 'visualiser2D.gif'")

        plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.92])
        plt.show()

class Visualiser2DExtra:
    def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_path=None, measurement_times=None, break_point=0, mode='prewritten'):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt
        self.focus_target = 'true'  # or 'predicted'
        self.zoom_factor = 1.0
        self.mode = mode
        self.measurement_times = measurement_times

        # File paths
        self.TRAJECTORY_FILE = trajectory_file_path
        self.PREDICTION_FILE = prediction_file_path
        self.HEATMAP_FILE = heatmap_file_path

        # Data storage
        self.trajectory = []
        self.predictions = []
        self.crash_angles_history = []
        self.data_lock = threading.Lock()
        self.position_queue = Queue()
        self.prediction_queue = Queue()
        self.uncertainty_polygon = None

        # Heatmap data storage
        self.heatmap_data = []  # List of (time, [angles]) tuples
        self.current_heatmap_time = 0
        self.heatmap_artists = []
        self.heatmap_cmap = plt.cm.viridis
        self.heatmap_cbar = None
        self.heatmap_resolution = 360
        self.heatmap_alpha = 0.7

        # Setup figure
        self.setup_plots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def setup_plots(self):
        # Plot layout
        self.fig = plt.figure(figsize=(18, 10))
        gs = self.fig.add_gridspec(2, 2)

        self.ax_full = self.fig.add_subplot(gs[:, 0])
        self.ax_zoom = self.fig.add_subplot(gs[0, 1])
        self.ax_hist = self.fig.add_subplot(gs[1, 1])

        self.fig.suptitle('Satellite Trajectory with Prediction Uncertainty', fontsize=14, y=0.98)

        # Configure axes
        for ax in (self.ax_full, self.ax_zoom):
            ax.set_aspect('equal')
            earth_circle = plt.Circle((0, 0), self.earth_radius, color='blue', alpha=0.3, zorder=1)
            ax.add_patch(earth_circle)
            ax.grid(True, alpha=0.3)

            # Add heatmap to the full view
            self.heatmap_artists = []

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

        self.heatmap_cbar = None  # Ensure this is set
        self.init_colorbar()

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

        self.ax_hist.set_title("Crash Angle Histogram (Monte Carlo Estimate)")
        self.ax_hist.set_xlabel("Angle (degrees)")
        self.ax_hist.set_ylabel("Frequency")
        self.ax_hist.set_xlim(0, 360)
        self.ax_hist.set_xticks(np.arange(0, 361, 60))
        self.ax_hist.set_ylim(0, 1)  # Temporary, will be updated dynamically
        self.hist_bar_container = None  # For tracking histogram plot handles

    def init_colorbar(self):
        if self.heatmap_cbar is None:
            sm = plt.cm.ScalarMappable(cmap=self.heatmap_cmap,
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            self.heatmap_cbar = self.fig.colorbar(
                sm, ax=self.ax_full,
                orientation='vertical',
                pad=0.05,
                label='Crash Probability (relative)'
            )
            # Don't add to heatmap_artists so it persists

    def add_radial_ticks_and_labels(self, ax):
        num_ticks = 12

        for i in range(num_ticks):
            angle_deg = i * (360 / num_ticks)
            angle_rad = np.deg2rad(angle_deg)

            # Coordinates for the tick position on the edge of the circle
            x_tick, y_tick = polar_to_cartesian(self.earth_radius, angle_rad)

            # Draw tick
            tick_length = - 0.01 * self.earth_radius  # -1.0 for insisde
            tick_mult_x, tick_mult_y = polar_to_cartesian(tick_length, angle_rad)
            tick_x = [x_tick, x_tick + tick_mult_x]
            tick_y = [y_tick, y_tick + tick_mult_y]
            ax.plot(tick_x, tick_y, color='black', lw=2, zorder=2)

            # Add angle label
            label_x, label_y = polar_to_cartesian(self.earth_radius * 0.85, angle_rad)
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
                        x, y = polar_to_cartesian(r, theta)
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
                        x, y = polar_to_cartesian(r, theta)
                        yield x, y
                    except ValueError:
                        continue

    def read_next_prediction(self):
        with open(self.PREDICTION_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        time, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        x, y = polar_to_cartesian(r, theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield time, x, y, std_x, std_y, int(is_meas)
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
                        time, r, theta, dr, dtheta, is_meas = map(float, line.strip().split())
                        x, y = polar_to_cartesian(r, theta)
                        std_x = np.sqrt((dr * np.cos(theta)) ** 2 + (r * dtheta * np.sin(theta)) ** 2)
                        std_y = np.sqrt((dr * np.sin(theta)) ** 2 + (r * dtheta * np.cos(theta)) ** 2)
                        yield time, x, y, std_x, std_y, int(is_meas)
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

    def load_heatmap_data(self):
        if not self.HEATMAP_FILE:
            return

        try:
            with open(self.HEATMAP_FILE, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    timestamp = float(parts[0])
                    angles = list(map(float, parts[1:]))
                    self.heatmap_data.append((timestamp, angles))
        except FileNotFoundError:
            print(f"Heatmap file {self.HEATMAP_FILE} not found")

    def update_histogram(self, angles_rad):
        angles_deg = np.degrees(angles_rad) % 360

        self.ax_hist.cla()
        self.ax_hist.set_title("Crash Angle Histogram (Monte Carlo Estimate)")
        self.ax_hist.set_xlabel("Angle (degrees)")
        self.ax_hist.set_ylabel("Frequency")
        self.ax_hist.set_xlim(0, 360)
        self.ax_hist.set_xticks(np.arange(0, 361, 5))

        plot_min = 0.95 * np.min(angles_deg)
        plot_max = 1.05 * np.max(angles_deg)

        hist_vals, bin_edges, self.hist_bar_container = self.ax_hist.hist(
            angles_deg,
            bins=self.heatmap_resolution,
            range=(0, 360),
            color='red',
            alpha=0.7
        )
        self.ax_hist.set_ylim(0, max(5, hist_vals.max() * 1.2))
        self.ax_hist.set_xlim(plot_min, plot_max)

    def update_heatmap(self, current_time):
        # Clear previous heatmap wedges
        for artist in self.heatmap_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.heatmap_artists = []

        if not self.heatmap_data:
            return

        # Find most recent prediction ≤ current_time
        latest_angles = None
        for t, angles in reversed(self.heatmap_data):
            if t <= current_time:
                latest_angles = angles
                break

        if not latest_angles:
            return

        # Create histogram
        angles_rad = np.array(latest_angles) % (2 * np.pi)
        hist, bin_edges = np.histogram(angles_rad, bins=self.heatmap_resolution, range=(0, 2 * np.pi))
        max_count = max(1, hist.max())

        # Create colored wedges
        for i in range(self.heatmap_resolution):
            if hist[i] == 0:
                continue
            start_angle = np.degrees(bin_edges[i])
            end_angle = np.degrees(bin_edges[i + 1])
            intensity = hist[i] / max_count

            wedge = Wedge(
                (0, 0),
                self.earth_radius,
                start_angle,
                end_angle,
                width=self.earth_radius * 0.05,
                color=self.heatmap_cmap(intensity),
                alpha=self.heatmap_alpha * intensity
            )
            self.ax_full.add_patch(wedge)
            self.heatmap_artists.append(wedge)

        # Update colorbar
        if self.heatmap_cbar:
            sm = plt.cm.ScalarMappable(cmap=self.heatmap_cmap,
                                       norm=plt.Normalize(vmin=0, vmax=max_count))
            sm.set_array([])
            self.heatmap_cbar.update_normal(sm)
        else:
            self.init_colorbar()

        # Update the crash angle histogram
        self.update_histogram(angles_rad)

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
                    focus = (self.predictions[-1][1], self.predictions[-1][2])

                if focus:
                    fx, fy = focus
                    dist = np.hypot(fx, fy)
                    base_width = max(500000, dist / 3)
                    zoom_width = base_width * self.zoom_factor
                    self.ax_zoom.set_xlim(fx - zoom_width / 2, fx + zoom_width / 2)
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
                pred_ts, pred_xs, pred_ys, std_xs, std_ys, meas_flags = zip(*self.predictions)
                pred_x, pred_y = pred_xs[-1], pred_ys[-1]
                current_time = pred_ts[-1]

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
                    polygon_points.append((x - sx, y - sy))

                # Reverse pass
                for i in reversed(range(len(pred_xs))):
                    x, y = pred_xs[i], pred_ys[i]
                    sx, sy = std_xs[i], std_ys[i]
                    polygon_points.append((x + sx, y + sy))

                self.uncertainty_polygon = Polygon(polygon_points, closed=True,
                                                   color='blue', alpha=0.15, zorder=2)
                self.ax_zoom.add_patch(self.uncertainty_polygon)
                artists.append(self.uncertainty_polygon)

                # Update measurement points
                meas_xs = [x for x, flag in zip(pred_xs, meas_flags) if flag]
                meas_ys = [y for y, flag in zip(pred_ys, meas_flags) if flag]
                self.pred_measurements_zoom.set_data(meas_xs, meas_ys)

                # Update heatmap for current time
                self.update_heatmap(current_time)
                artists.extend(self.heatmap_artists)

                return artists

        except Exception as e:
            print(f"Animation error: {e}")

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Load heatmap data properly
        self.load_heatmap_data()

        # Load heatmap data if file exists
        if self.HEATMAP_FILE:
            try:
                with open(self.HEATMAP_FILE, 'r') as f:
                    for line in f:
                        samples = map(float, line.strip().split())
                        self.crash_angles_history.extend(samples)
            except FileNotFoundError:
                print(f"Heatmap file {self.HEATMAP_FILE} not found")

        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update,
            frames=1000,
            interval=50,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.92])
        plt.show()

class Visualiser3D:
    def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_path=None, thrust_heatmap_file_path=None, break_point=0, mode='prewritten', MAX_STEPS=50000):
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
        self.crash_toggle = 1

        # Heatmap parameters
        self.heatmap_data = []  # List of (time, [(theta, phi, intensity)]) tuples
        self.current_heatmap_time = 0
        self.heatmap_cmap = plt.cm.Reds
        self.heatmap_alpha = 0.8  # Increase base alpha
        self.heatmap_resolution = 50  # Number of divisions in theta and phi
        self.heatmap_artists = []
        self.heatmap_cbar = None

        # Thrust Heatmap parameters
        self.thrust_heatmap_data = []  # List of (time, [(theta, phi, intensity)]) tuples
        self.thrust_current_heatmap_time = 0
        self.thrust_heatmap_cmap = plt.cm.Greens
        self.thrust_heatmap_artists = []
        self.thrust_heatmap_cbar = None

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
        self.HEATMAP_FILE = heatmap_file_path
        self.THRUST_HEATMAP_FILE = thrust_heatmap_file_path

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
        self.fixed_limits = (-7E+6, 7E+6)  # Match initial plot_radius

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
        self.plot_earth(self.ax_full)
        self.plot_earth(self.ax_zoom)

        # Set full view limits
        plot_radius = 7E+6
        self.ax_full.set_xlim(-plot_radius, plot_radius)
        self.ax_full.set_ylim(-plot_radius, plot_radius)
        self.ax_full.set_zlim(-plot_radius, plot_radius)
        self.set_axes_equal(self.ax_full)

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

    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    def plot_earth(self, ax):
        # Create a sphere for Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(u, v)
        x, y, z = spherical_to_cartesian(self.earth_radius, theta, phi)

        # Plot Earth surface
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

        # Add longitude and latitude lines
        self.add_earth_grid(ax)

    def add_earth_grid(self, ax):
        # Longitude lines (meridians)
        theta = np.linspace(0, 2 * np.pi, 13)
        phi = np.linspace(0, np.pi, 100)

        for t in theta[:-1]:
            x, y, z = spherical_to_cartesian(self.earth_radius, t, phi)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Latitude lines (parallels)
        phi = np.linspace(0, np.pi, 7)
        theta = np.linspace(0, 2 * np.pi, 100)

        for p in phi[1:-1]:
            x, y, z = spherical_to_cartesian(self.earth_radius, theta, p)
            z *= np.ones_like(theta)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Labels for longitudinal lines
        label_angles = [0, 90, 180, 270]
        for angle in label_angles:
            rad = np.deg2rad(angle)
            x, y = polar_to_cartesian(self.earth_radius * 1.05, rad)
            z = 0
            ax.text(x, y, z, f'{angle}°', color='black', fontsize=8, ha='center', va='center')

        # Latitude labels
        label_lats = [0, 30, 60, -30, -60]
        for lat in label_lats:
            rad = np.deg2rad(lat)
            z, r = polar_to_cartesian(self.earth_radius, rad)
            x = r * 1.05
            y = 0
            ax.text(x, y, z, f'{lat}°', color='black', fontsize=8, ha='center', va='center')

    def load_heatmap_data(self):
        for file_path in [self.HEATMAP_FILE, self.THRUST_HEATMAP_FILE]:
            if not file_path:
                continue
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split()
                        if len(parts) < 3:  # Need at least timestamp, theta, phi
                            continue
                        timestamp = float(parts[0])
                        points = []
                        # Read triplets of (theta, phi, intensity)
                        for i in range(1, len(parts), 2):
                            try:
                                theta = float(parts[i])
                                phi = float(parts[i+1])
                                points.append((theta, phi))
                            except (IndexError, ValueError):
                                continue
                        if file_path == self.HEATMAP_FILE:
                            self.heatmap_data.append((timestamp, points))
                        else:
                            self.thrust_heatmap_data.append((timestamp, points))
            except FileNotFoundError:
                print(f"Heatmap file {file_path} not found")

    def update_heatmap(self, current_time):
        # Clear previous heatmaps
        for heat_artists in [self.heatmap_artists, self.thrust_heatmap_artists]:
            for artist in heat_artists:
                artist.remove()
            heat_artists.clear()

        # Process both heatmap datasets
        for dataset, cmap, artist_list in [
            (self.heatmap_data, self.heatmap_cmap, self.heatmap_artists),
            (self.thrust_heatmap_data, self.thrust_heatmap_cmap, self.thrust_heatmap_artists)
        ]:
            if not dataset:
                continue

            # Collect all points up to current_time
            current_points = []
            for t, points in dataset:
                if t <= current_time:
                    current_points.extend(points)

            if not current_points:
                continue

            # Convert to numpy array and fix coordinate ranges
            points_array = np.array(current_points)
            thetas = points_array[:, 0] % (2 * np.pi)  # Wrap theta to [0, 2π]
            phis = - np.abs(points_array[:, 1]) % np.pi  # Wrap phi to [0, π]

            # Create grid for density calculation
            theta_grid = np.linspace(0, 2 * np.pi, self.heatmap_resolution)
            phi_grid = np.linspace(0, np.pi, self.heatmap_resolution)
            theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)

            # Calculate point density
            H, _, _ = np.histogram2d(phis, thetas, bins=[phi_grid, theta_grid])

            # Apply smoothing if available
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=1)
            except ImportError:
                print("Note: scipy.ndimage not available, skipping smoothing")
                pass

            # Normalize
            if H.max() > 0:
                grid_z = H / H.max()
            else:
                grid_z = H

            # Offset radius slightly if this is the thrust heatmap
            r = self.earth_radius
            if dataset is self.thrust_heatmap_data:
                if self.crash_toggle == 1:
                    r *= 0.98
                else:
                    r *= 1.02  # 2% lift to float above nominal heatmap

            # Create surface coordinates
            x, y, z = spherical_to_cartesian(r, theta_mesh, phi_mesh)

            # Apply colormap with consistent alpha
            norm = Normalize(vmin=0, vmax=1)
            mapped_colors = cmap(norm(grid_z))
            alpha = 0.3 + 0.7 * np.power(grid_z, 0.3)
            mapped_colors[..., -1] = alpha * self.heatmap_alpha

            # Plot the heatmap
            heatmap = self.ax_full.plot_surface(
                x, y, z,
                facecolors=mapped_colors,
                rstride=1,
                cstride=1,
                shade=False,
                zorder=2,
                antialiased=True,
                linewidth=0.0
            )
            artist_list.append(heatmap)

            # Force redraw
            self.fig.canvas.draw_idle()

        # Update combined colorbar
        self.update_colorbar()

    def update_colorbar(self):
        # Only proceed if we have data to show
        has_nominal = len(self.heatmap_data) > 0
        has_thrust = hasattr(self, 'thrust_heatmap_data') and len(self.thrust_heatmap_data) > 0

        if not has_nominal and not has_thrust:
            # Remove colorbar if it exists but we have no data
            if hasattr(self, 'heatmap_cbar') and self.heatmap_cbar is not None:
                self.heatmap_cbar.remove()
                self.heatmap_cbar = None
            if hasattr(self, 'thrust_heatmap_cbar') and self.thrust_heatmap_cbar is not None:
                self.thrust_heatmap_cbar.remove()
                self.thrust_heatmap_cbar = None
            return

        # Get current axis limits to preserve them
        xlim = self.ax_full.get_xlim()
        ylim = self.ax_full.get_ylim()
        zlim = self.ax_full.get_zlim()

        # Set layout parameters once at initialization
        if not hasattr(self, '_layout_initialized'):
            # self.fig.subplots_adjust(right=0.80)  # Permanent space for colorbars
            self._layout_initialized = True

        # Create/update colorbar without changing layout engine
        if not hasattr(self, 'heatmap_cbar') or self.heatmap_cbar is None:
            if has_nominal:
                sm_nominal = plt.cm.ScalarMappable(cmap=self.heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                self.heatmap_cbar = self.fig.colorbar(
                    sm_nominal,
                    ax=self.ax_full,
                    orientation='horizontal',
                    pad=0.1,
                    label='Nominal Crash Probability'
                )
                # self.heatmap_cbar.ax.set_position([0.0, 0.0, 0.0, 0.0])

            if has_thrust:
                sm_thrust = plt.cm.ScalarMappable(cmap=self.thrust_heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                self.thrust_heatmap_cbar = self.fig.colorbar(
                    sm_thrust,
                    ax=self.ax_full,
                    orientation='vertical',
                    pad=0.15,
                    label='Thruster Crash Probability'
                )
                # self.thrust_heatmap_cbar.ax.set_position([0.85, 0.05, 0.03, 0.40])
        else:
            # Update existing colorbars
            if has_nominal:
                self.heatmap_cbar.mappable.set_cmap(self.heatmap_cmap)
                self.heatmap_cbar.set_label('Nominal Crash Probability')
                # self.heatmap_cbar.ax.set_position([0.85, 0.50, 0.03, 0.45])
            elif hasattr(self, 'heatmap_cbar') and self.heatmap_cbar is not None:
                self.heatmap_cbar.remove()
                self.heatmap_cbar = None

            if has_thrust:
                if not hasattr(self, 'thrust_heatmap_cbar') or self.thrust_heatmap_cbar is None:
                    sm_thrust = plt.cm.ScalarMappable(cmap=self.thrust_heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    self.thrust_heatmap_cbar = self.fig.colorbar(
                        sm_thrust,
                        ax=self.ax_full,
                        orientation='vertical',
                        pad=0.05,
                        label='Thruster Crash Probability'
                    )
                self.thrust_heatmap_cbar.mappable.set_cmap(self.thrust_heatmap_cmap)
                self.thrust_heatmap_cbar.set_label('Thruster Crash Probability')
                # self.thrust_heatmap_cbar.ax.set_position([0.85, 0.05, 0.03, 0.40])
            elif hasattr(self, 'thrust_heatmap_cbar') and self.thrust_heatmap_cbar is not None:
                self.thrust_heatmap_cbar.remove()
                self.thrust_heatmap_cbar = None

        # Restore original axis limits
        self.ax_full.set_xlim(xlim)
        self.ax_full.set_ylim(ylim)
        self.ax_full.set_zlim(zlim)

        # Manual adjustment instead of layout engine
        # self.fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on right for colorbars

    def on_key_press(self, event):
        if event.key == 't':
            self.focus_on = 'true'  # Fixed variable name (was focus_target)
            print("Focusing on true trajectory")
        elif event.key == 'p':
            self.focus_on = 'predicted'  # Fixed variable name (was focus_target)
            print("Focusing on predicted trajectory")
        elif event.key == 'c':
            self.crash_toggle = - self.crash_toggle
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
                        x, y, z = spherical_to_cartesian(r, theta, phi)
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
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        yield x, y, z
                    except ValueError:
                        continue

    def read_next_prediction(self):
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
                        time, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield time, x, y, z, std_x, std_y, std_z, int(is_meas)
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
                        time, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield time, x, y, z, std_x, std_y, std_z, int(is_meas)
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

                current_dist = np.sqrt(x**2 + y**2 + z**2)
                altitude = current_dist - self.earth_radius

                # Update camera view if not user-controlled
                if not self.user_controlled and len(self.trajectory) >= 2:
                    # Get last two positions to determine direction
                    (x_prev, y_prev, z_prev), (x_curr, y_curr, z_curr) = self.trajectory[-2], self.trajectory[-1]
                    dx, dy, dz = x_curr - x_prev, y_curr - y_prev, z_curr - z_prev

                    # Calculate azimuth and elevation from velocity vector
                    azim = np.degrees(np.arctan2(dy, dx))
                    elev = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
                    self.ax_zoom.view_init(elev=elev, azim=azim)

                # Update prediction visualisation
                if current_pred:
                    pred_t, pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas = current_pred
                    self.predictions.append((pred_t, pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas))

                    if len(self.predictions) > 1:
                        # Extract prediction points and uncertainties
                        pred_t = np.array([t for t, _, _, _, _, _, _, _ in self.predictions])
                        pred_points = np.array([(x, y, z) for _, x, y, z, _, _, _, _ in self.predictions])
                        pred_xs, pred_ys, pred_zs = pred_points.T
                        std_devs = [(sx, sy, sz) for _, _, _, _, sx, sy, sz, _ in self.predictions]
                        meas_flags = [m for _, _, _, _, _, _, _, m in self.predictions]

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
                        rgba_blue = mcolors.to_rgba('blue', alpha=0.15)

                        self.uncertainty_tube = Poly3DCollection(
                            [vertices[face] for face in faces],
                            facecolors=rgba_blue,
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

                        # Update heatmap for current time (use prediction time)
                        current_time = pred_t.T[-1]  # Or use actual time if available
                        self.update_heatmap(current_time)
                        # print(self.fixed_limits)
                        artists.extend(self.heatmap_artists)

                # Update zoomed view limits
                focus_point = None
                if self.focus_on == 'true' and self.trajectory:
                    focus_point = self.trajectory[-1]
                elif self.focus_on == 'predicted' and self.predictions:
                    focus_point = (self.predictions[-1][0], self.predictions[-1][1], self.predictions[-1][2])

                if focus_point:
                    fx, fy, fz = focus_point
                    base_width = max(500000, np.sqrt(fx**2 + fy**2 + fz**2) / 3)
                    zoom_width = base_width * self.zoom_scale

                    self.ax_zoom.set_xlim(fx - zoom_width/2, fx + zoom_width/2)
                    self.ax_zoom.set_ylim(fy - zoom_width/2, fy + zoom_width/2)
                    self.ax_zoom.set_zlim(fz - zoom_width/2, fz + zoom_width/2)

                # Update altitude text
                self.altitude_text.set_text(
                    f"Altitude: {altitude/1000:.1f} km\n"
                    f"Distance: {current_dist/1000:.1f} km\n"
                    f"Position: ({x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f}) km"
                )

                # Check for impact
                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(
                        f"IMPACT!\n"
                        f"Final altitude: {altitude:.1f} m\n"
                        f"Final Position: ({x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f}) km"
                    )
                    return artists

        except Exception as e:
            print(f"Animation error: {e}")
            import traceback
            traceback.print_exc()

        if not self.user_controlled:
            self.ax_full.set_xlim(self.fixed_limits)
            self.ax_full.set_ylim(self.fixed_limits)
            self.ax_full.set_zlim(self.fixed_limits)
            self.ax_full.set_aspect('auto')

        self.set_axes_equal(self.ax_full)

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Load heatmap data
        self.load_heatmap_data()

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

class Visualiser3DExtra:
    def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_path=None, thrust_heatmap_file_path=None, break_point=0, mode='prewritten', MAX_STEPS=50000):
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

        # Heatmap parameters
        self.heatmap_data = []  # List of (time, [(theta, phi, intensity)]) tuples
        self.current_heatmap_time = 0
        self.heatmap_cmap = plt.cm.Reds
        self.heatmap_alpha = 0.8  # Increase base alpha
        self.heatmap_resolution = 50  # Number of divisions in theta and phi
        self.heatmap_artists = []
        self.heatmap_cbar = None

        # Thrust Heatmap parameters
        self.thrust_heatmap_data = []  # List of (time, [(theta, phi, intensity)]) tuples
        self.thrust_current_heatmap_time = 0
        self.thrust_heatmap_cmap = plt.cm.Greens
        self.thrust_heatmap_artists = []
        self.thrust_heatmap_cbar = None

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
        self.HEATMAP_FILE = heatmap_file_path
        self.THRUST_HEATMAP_FILE = thrust_heatmap_file_path

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
        self.fixed_limits = (-7E+6, 7E+6)  # Match initial plot_radius

    def setup_plots(self):
        # Create 3D figure with two subplots
        self.fig = plt.figure(figsize=(14, 9))  # More vertical space
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1])
        self.ax_full = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_zoom = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax_heatmap = self.fig.add_subplot(gs[1, 1])  # Bottom right

        self.fig.suptitle('3D Satellite Trajectory with Prediction Uncertainty', fontsize=14, y=0.98)

        # Full view 3D plot
        # self.ax_full = self.fig.add_subplot(121, projection='3d')
        self.ax_full.set_title('Full 3D Trajectory View')
        self.ax_full.set_xlabel('X position (m)')
        self.ax_full.set_ylabel('Y position (m)')
        self.ax_full.set_zlabel('Z position (m)')

        # Zoomed view 3D plot
        # self.ax_zoom = self.fig.add_subplot(122, projection='3d')
        self.ax_zoom.set_title('Zoomed 3D View with Uncertainty')
        self.ax_zoom.set_xlabel('X position (m)')
        self.ax_zoom.set_ylabel('Y position (m)')
        self.ax_zoom.set_zlabel('Z position (m)')

        # Create Earth sphere for both views
        self.plot_earth(self.ax_full)
        self.plot_earth(self.ax_zoom)

        # Set full view limits
        plot_radius = 7E+6
        self.ax_full.set_xlim(-plot_radius, plot_radius)
        self.ax_full.set_ylim(-plot_radius, plot_radius)
        self.ax_full.set_zlim(-plot_radius, plot_radius)
        self.set_axes_equal(self.ax_full)

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
        self.altitude_text = self.fig.text(0.3, 0.025, "Altitude: Initializing...",
                                         fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

        # Add legends
        self.ax_full.legend(loc='upper right')
        self.ax_zoom.legend(loc='upper right')

    def set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    def plot_earth(self, ax):
        # Create a sphere for Earth
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        theta, phi = np.meshgrid(u, v)
        x, y, z = spherical_to_cartesian(self.earth_radius, theta, phi)

        # Plot Earth surface
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)

        # Add longitude and latitude lines
        self.add_earth_grid(ax)

    def add_earth_grid(self, ax):
        # Longitude lines (meridians)
        theta = np.linspace(0, 2 * np.pi, 13)
        phi = np.linspace(0, np.pi, 100)

        for t in theta[:-1]:
            x, y, z = spherical_to_cartesian(self.earth_radius, t, phi)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Latitude lines (parallels)
        phi = np.linspace(0, np.pi, 7)
        theta = np.linspace(0, 2 * np.pi, 100)

        for p in phi[1:-1]:
            x, y, z = spherical_to_cartesian(self.earth_radius, theta, p)
            z *= np.ones_like(theta)
            ax.plot(x, y, z, color='black', alpha=0.6, linewidth=0.5)

        # Labels for longitudinal lines
        label_angles = [0, 90, 180, 270]
        for angle in label_angles:
            rad = np.deg2rad(angle)
            x, y = polar_to_cartesian(self.earth_radius * 1.05, rad)
            z = 0
            ax.text(x, y, z, f'{angle}°', color='black', fontsize=8, ha='center', va='center')

        # Latitude labels
        label_lats = [0, 30, 60, -30, -60]
        for lat in label_lats:
            rad = np.deg2rad(lat)
            z, r = polar_to_cartesian(self.earth_radius, rad)
            x = r * 1.05
            y = 0
            ax.text(x, y, z, f'{lat}°', color='black', fontsize=8, ha='center', va='center')

    def load_heatmap_data(self):
        for file_path in [self.HEATMAP_FILE, self.THRUST_HEATMAP_FILE]:
            if not file_path:
                continue
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split()
                        if len(parts) < 3:  # Need at least timestamp, theta, phi
                            continue
                        timestamp = float(parts[0])
                        points = []
                        # Read triplets of (theta, phi, intensity)
                        for i in range(1, len(parts), 2):
                            try:
                                theta = float(parts[i])
                                phi = float(parts[i+1])
                                points.append((theta, phi))
                            except (IndexError, ValueError):
                                continue
                        if file_path == self.HEATMAP_FILE:
                            self.heatmap_data.append((timestamp, points))
                        else:
                            self.thrust_heatmap_data.append((timestamp, points))
            except FileNotFoundError:
                print(f"Heatmap file {file_path} not found")

    def update_theta_phi_heatmap(self, current_time):
        # Clear previous image
        self.ax_heatmap.clear()
        self.ax_heatmap.set_title("Latitude–Longitude Heatmap (Crash Distribution)")
        self.ax_heatmap.set_xlabel("Longitude (°)")
        self.ax_heatmap.set_ylabel("Latitude (°)")

        # Set resolution and matching bin edges in degrees
        resolution = self.heatmap_resolution
        lon_bins = np.linspace(0, 360, resolution)
        lat_bins = np.linspace(-90, 90, resolution)

        # Create empty histograms
        nominal_hist = np.zeros((resolution - 1, resolution - 1))
        thrust_hist = np.zeros((resolution - 1, resolution - 1))

        # Store all lat/lon for axis range calculation
        all_lons = []
        all_lats = []

        # Helper to fill histogram and collect transformed coordinates
        def accumulate_hist(data, hist):
            current_points = []
            for t, points in data:
                if t <= current_time:
                    current_points.extend(points)
            if current_points:
                arr = np.array(current_points)
                thetas = arr[:, 0] % (2 * np.pi)  # [0, 2π)
                phis = arr[:, 1] % np.pi  # [0, π)

                lons = np.degrees(thetas)  # [0°, 360°)
                lats = 90 - np.degrees(phis)  # [90°, -90°] -> flip to latitude

                all_lons.extend(lons)
                all_lats.extend(lats)

                H, _, _ = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])
                return H
            return hist

        # Accumulate data
        if self.heatmap_data:
            nominal_hist = accumulate_hist(self.heatmap_data, nominal_hist)
        if self.thrust_heatmap_data:
            thrust_hist = accumulate_hist(self.thrust_heatmap_data, thrust_hist)

        # Normalize histograms
        nominal_hist /= (nominal_hist.max() or 1)
        thrust_hist /= (thrust_hist.max() or 1)

        # Combine in RGB: R = nominal, G = thrust
        combined_rgb = np.zeros((nominal_hist.shape[0], nominal_hist.shape[1], 3))
        combined_rgb[..., 0] = nominal_hist  # Red channel
        combined_rgb[..., 1] = thrust_hist  # Green channel

        # Determine extent and axis limits
        extent = [lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]]

        if all_lons and all_lats:
            lon_min = max(0, 0.95 * min(all_lons))
            lon_max = min(360, 1.05 * max(all_lons))
            lat_min = max(-90, 0.95 * min(all_lats))
            lat_max = min(90, 1.05 * max(all_lats))
        else:
            lon_min, lon_max = 0, 360
            lat_min, lat_max = -90, 90

        # Display the heatmap
        self.ax_heatmap.imshow(
            np.flipud(combined_rgb),
            extent=extent,
            aspect='auto',
            origin='lower'
        )
        self.ax_heatmap.set_xlim(lon_min, lon_max)
        self.ax_heatmap.set_ylim(lat_min, lat_max)

    def update_heatmap(self, current_time):
        # Clear previous heatmaps
        for heat_artists in [self.heatmap_artists, self.thrust_heatmap_artists]:
            for artist in heat_artists:
                artist.remove()
            heat_artists.clear()

        # Process both heatmap datasets
        for dataset, cmap, artist_list in [
            (self.heatmap_data, self.heatmap_cmap, self.heatmap_artists),
            (self.thrust_heatmap_data, self.thrust_heatmap_cmap, self.thrust_heatmap_artists)
        ]:
            if not dataset:
                continue

            # Collect all points up to current_time
            current_points = []
            for t, points in dataset:
                if t <= current_time:
                    current_points.extend(points)

            if not current_points:
                continue

            # Convert to numpy array and fix coordinate ranges
            points_array = np.array(current_points)
            thetas = points_array[:, 0] % (2 * np.pi)  # Wrap theta to [0, 2π]
            phis = - np.abs(points_array[:, 1]) % np.pi  # Wrap phi to [0, π]

            # Create grid for density calculation
            theta_grid = np.linspace(0, 2 * np.pi, self.heatmap_resolution)
            phi_grid = np.linspace(0, np.pi, self.heatmap_resolution)
            theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)

            # Calculate point density
            H, _, _ = np.histogram2d(phis, thetas, bins=[phi_grid, theta_grid])

            # Apply smoothing if available
            try:
                from scipy.ndimage import gaussian_filter
                H = gaussian_filter(H, sigma=1)
            except ImportError:
                print("Note: scipy.ndimage not available, skipping smoothing")
                pass

            # Normalize
            if H.max() > 0:
                grid_z = H / H.max()
            else:
                grid_z = H

            # Offset radius slightly if this is the thrust heatmap
            r = self.earth_radius
            if dataset is self.thrust_heatmap_data:
                r *= 1.02  # 2% lift to float above nominal heatmap

            # Create surface coordinates
            x, y, z = spherical_to_cartesian(r, theta_mesh, phi_mesh)

            # Apply colormap with consistent alpha
            norm = Normalize(vmin=0, vmax=1)
            mapped_colors = cmap(norm(grid_z))
            alpha = 0.3 + 0.7 * np.power(grid_z, 0.3)
            mapped_colors[..., -1] = alpha * self.heatmap_alpha

            # Plot the heatmap
            heatmap = self.ax_full.plot_surface(
                x, y, z,
                facecolors=mapped_colors,
                rstride=1,
                cstride=1,
                shade=False,
                zorder=2,
                antialiased=True,
                linewidth=0.0
            )
            artist_list.append(heatmap)

            # Force redraw
            self.fig.canvas.draw_idle()

        # Update combined colorbar
        self.update_colorbar()

        # Update flat heatmap
        self.update_theta_phi_heatmap(current_time)

    def update_colorbar(self):
        # Only proceed if we have data to show
        has_nominal = len(self.heatmap_data) > 0
        has_thrust = hasattr(self, 'thrust_heatmap_data') and len(self.thrust_heatmap_data) > 0

        if not has_nominal and not has_thrust:
            # Remove colorbar if it exists but we have no data
            if hasattr(self, 'heatmap_cbar') and self.heatmap_cbar is not None:
                self.heatmap_cbar.remove()
                self.heatmap_cbar = None
            if hasattr(self, 'thrust_heatmap_cbar') and self.thrust_heatmap_cbar is not None:
                self.thrust_heatmap_cbar.remove()
                self.thrust_heatmap_cbar = None
            return

        # Get current axis limits to preserve them
        xlim = self.ax_full.get_xlim()
        ylim = self.ax_full.get_ylim()
        zlim = self.ax_full.get_zlim()

        # Set layout parameters once at initialization
        if not hasattr(self, '_layout_initialized'):
            # self.fig.subplots_adjust(right=0.80)  # Permanent space for colorbars
            self._layout_initialized = True

            # Create/update colorbar without changing layout engine
            if not hasattr(self, 'heatmap_cbar') or self.heatmap_cbar is None:
                if has_nominal:
                    sm_nominal = plt.cm.ScalarMappable(cmap=self.heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    self.heatmap_cbar = self.fig.colorbar(
                        sm_nominal,
                        ax=self.ax_full,
                        orientation='horizontal',
                        pad=0.1,
                        label='Nominal Crash Probability'
                    )
                    # Adjust the horizontal colorbar size and position
                    pos = self.heatmap_cbar.ax.get_position()
                    new_width = pos.width * 0.7  # Make it 30% narrower
                    new_height = pos.height * 0.7  # Make it 30% shorter
                    new_left = pos.x0 + (pos.width - new_width) / 2  # Center horizontally
                    new_bottom = pos.y0 + (pos.height - new_height)  # Keep at bottom
                    self.heatmap_cbar.ax.set_position([new_left, new_bottom, new_width, new_height])

            if has_thrust:
                sm_thrust = plt.cm.ScalarMappable(cmap=self.thrust_heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                self.thrust_heatmap_cbar = self.fig.colorbar(
                    sm_thrust,
                    ax=self.ax_full,
                    orientation='vertical',
                    pad=0.15,
                    label='Thruster Crash Probability'
                )
                # self.thrust_heatmap_cbar.ax.set_position([0.85, 0.05, 0.03, 0.40])
        else:
            # Update existing colorbars
            if has_nominal:
                self.heatmap_cbar.mappable.set_cmap(self.heatmap_cmap)
                self.heatmap_cbar.set_label('Nominal Crash Probability')
                # self.heatmap_cbar.ax.set_position([0.85, 0.50, 0.03, 0.45])
            elif hasattr(self, 'heatmap_cbar') and self.heatmap_cbar is not None:
                self.heatmap_cbar.remove()
                self.heatmap_cbar = None

            if has_thrust:
                if not hasattr(self, 'thrust_heatmap_cbar') or self.thrust_heatmap_cbar is None:
                    sm_thrust = plt.cm.ScalarMappable(cmap=self.thrust_heatmap_cmap, norm=plt.Normalize(vmin=0, vmax=1))
                    self.thrust_heatmap_cbar = self.fig.colorbar(
                        sm_thrust,
                        ax=self.ax_full,
                        orientation='vertical',
                        pad=0.05,
                        label='Thruster Crash Probability'
                    )
                self.thrust_heatmap_cbar.mappable.set_cmap(self.thrust_heatmap_cmap)
                self.thrust_heatmap_cbar.set_label('Thruster Crash Probability')
                # self.thrust_heatmap_cbar.ax.set_position([0.85, 0.05, 0.03, 0.40])
            elif hasattr(self, 'thrust_heatmap_cbar') and self.thrust_heatmap_cbar is not None:
                self.thrust_heatmap_cbar.remove()
                self.thrust_heatmap_cbar = None

        # Restore original axis limits
        self.ax_full.set_xlim(xlim)
        self.ax_full.set_ylim(ylim)
        self.ax_full.set_zlim(zlim)

        # Manual adjustment instead of layout engine
        # self.fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave 15% space on right for colorbars

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
                        x, y, z = spherical_to_cartesian(r, theta, phi)
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
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        yield x, y, z
                    except ValueError:
                        continue

    def read_next_prediction(self):
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
                        time, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield time, x, y, z, std_x, std_y, std_z, int(is_meas)
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
                        time, r, theta, phi, dr, dtheta, dphi, is_meas = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        std_x, std_y, std_z = spherical_uncertainty_to_cartesian(r, theta, phi, dr, dtheta, dphi)
                        yield time, x, y, z, std_x, std_y, std_z, int(is_meas)
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

                current_dist = np.sqrt(x**2 + y**2 + z**2)
                altitude = current_dist - self.earth_radius

                # Update camera view if not user-controlled
                if not self.user_controlled and len(self.trajectory) >= 2:
                    # Get last two positions to determine direction
                    (x_prev, y_prev, z_prev), (x_curr, y_curr, z_curr) = self.trajectory[-2], self.trajectory[-1]
                    dx, dy, dz = x_curr - x_prev, y_curr - y_prev, z_curr - z_prev

                    # Calculate azimuth and elevation from velocity vector
                    azim = np.degrees(np.arctan2(dy, dx))
                    elev = np.degrees(np.arctan2(dz, np.sqrt(dx**2 + dy**2)))
                    self.ax_zoom.view_init(elev=elev, azim=azim)

                # Update prediction visualisation
                if current_pred:
                    pred_t, pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas = current_pred
                    self.predictions.append((pred_t, pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas))

                    if len(self.predictions) > 1:
                        # Extract prediction points and uncertainties
                        pred_t = np.array([t for t, _, _, _, _, _, _, _ in self.predictions])
                        pred_points = np.array([(x, y, z) for _, x, y, z, _, _, _, _ in self.predictions])
                        pred_xs, pred_ys, pred_zs = pred_points.T
                        std_devs = [(sx, sy, sz) for _, _, _, _, sx, sy, sz, _ in self.predictions]
                        meas_flags = [m for _, _, _, _, _, _, _, m in self.predictions]

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
                        rgba_blue = mcolors.to_rgba('blue', alpha=0.15)

                        self.uncertainty_tube = Poly3DCollection(
                            [vertices[face] for face in faces],
                            facecolors=rgba_blue,
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

                        # Update heatmap for current time (use prediction time)
                        current_time = pred_t.T[-1]  # Or use actual time if available
                        self.update_heatmap(current_time)
                        # print(self.fixed_limits)
                        artists.extend(self.heatmap_artists)

                # Update zoomed view limits
                focus_point = None
                if self.focus_on == 'true' and self.trajectory:
                    focus_point = self.trajectory[-1]
                elif self.focus_on == 'predicted' and self.predictions:
                    focus_point = (self.predictions[-1][0], self.predictions[-1][1], self.predictions[-1][2])

                if focus_point:
                    fx, fy, fz = focus_point
                    base_width = max(500000, np.sqrt(fx**2 + fy**2 + fz**2) / 3)
                    zoom_width = base_width * self.zoom_scale

                    self.ax_zoom.set_xlim(fx - zoom_width/2, fx + zoom_width/2)
                    self.ax_zoom.set_ylim(fy - zoom_width/2, fy + zoom_width/2)
                    self.ax_zoom.set_zlim(fz - zoom_width/2, fz + zoom_width/2)

                # Update altitude text
                self.altitude_text.set_text(
                    f"Altitude: {altitude/1000:.1f} km\n"
                    f"Distance: {current_dist/1000:.1f} km\n"
                    f"Position: ({x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f}) km"
                )

                # Check for impact
                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(
                        f"IMPACT!\n"
                        f"Final altitude: {altitude:.1f} m\n"
                        f"Final Position: ({x/1000:.1f}, {y/1000:.1f}, {z/1000:.1f}) km"
                    )
                    return artists

        except Exception as e:
            print(f"Animation error: {e}")
            import traceback
            traceback.print_exc()

        if not self.user_controlled:
            self.ax_full.set_xlim(self.fixed_limits)
            self.ax_full.set_ylim(self.fixed_limits)
            self.ax_full.set_zlim(self.fixed_limits)
            self.ax_full.set_aspect('auto')

        self.set_axes_equal(self.ax_full)

        return artists

    def visualise(self):
        # Start data loading thread
        data_thread = threading.Thread(target=self.load_data, daemon=True)
        data_thread.start()

        # Load heatmap data
        self.load_heatmap_data()

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
