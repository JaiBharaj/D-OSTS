import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from CrudeInitialConditions import InitialConditions
import threading
from queue import Queue, Empty
from collections import deque

class Visualiser2D:
    def __init__(self, trajectory_file_path, prediction_file_path, break_point=0):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt

        # File paths
        self.TRAJECTORY_FILE = trajectory_file_path
        self.PREDICTION_FILE = prediction_file_path

        # Data storage
        self.trajectory = []
        self.predictions = []
        self.data_lock = threading.Lock()
        self.position_queue = Queue()
        self.prediction_queue = Queue()
        self.uncertainty_polygon = None

        # Setup figure
        self.setup_plots()

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

    def read_next_position(self):
        with open(self.TRAJECTORY_FILE, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    f.seek(pos)
                    #time.sleep(0.005)
                    continue
                try:
                    x, y = map(float, line.strip().split())
                    yield x, y
                except ValueError:
                    continue

    def read_next_prediction(self):
        with open(self.PREDICTION_FILE, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    f.seek(pos)
                    #time.sleep(0.0001)
                    continue
                try:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        x, y, std_x, std_y, is_meas = map(float, parts)
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

    def update(self, frame):
        MAX_STEPS = 500

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
                zoom_width = max(500000, current_dist / 3)
                self.ax_zoom.set_xlim(x - zoom_width / 6, x + zoom_width / 6)
                self.ax_zoom.set_ylim(y - zoom_width / 6, y + zoom_width / 6)

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

                # Create new uncertainty polygon
                polygon_points = []
                for x, y, sx, sy in zip(pred_xs, pred_ys, std_xs, std_ys):
                    polygon_points.append((x - sx, y - sy))
                for x, y, sx, sy in reversed(list(zip(pred_xs, pred_ys, std_xs, std_ys))):
                    polygon_points.append((x + sx, y + sy))

                self.uncertainty_polygon = Polygon(polygon_points, closed=True,
                                                   color='blue', alpha=0.15, zorder=2)
                self.ax_zoom.add_patch(self.uncertainty_polygon)
                artists.append(self.uncertainty_polygon)

                # Update measurement points
                meas_xs = [x for x, flag in zip(pred_xs, meas_flags) if flag]
                meas_ys = [y for y, flag in zip(pred_ys, meas_flags) if flag]
                self.pred_measurements_zoom.set_data(meas_xs, meas_ys)

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
    def __init__(self, trajectory_file_path, prediction_file_path, break_point=0, MAX_STEPS=500):
        # Initialise parameters
        self.earth_radius = InitialConditions.earthRadius
        self.stop_distance = self.earth_radius + break_point
        self.initial_altitude = InitialConditions.initSatAlt

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
        self.ax_full.plot_surface(x, y, z, color='blue', alpha=0.3)
        self.ax_zoom.plot_surface(x, y, z, color='blue', alpha=0.3)

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

    def read_next_position(self):
        with open(self.TRAJECTORY_FILE, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    f.seek(pos)
                    # time.sleep(0.05)
                    continue
                try:
                    x, y, z = map(float, line.strip().split())
                    yield x, y, z
                except ValueError:
                    continue

    def read_next_prediction(self):
        with open(self.PREDICTION_FILE, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    f.seek(pos)
                    # time.sleep(0.05)
                    continue
                try:
                    parts = line.strip().split()
                    if len(parts) == 7:
                        x, y, z, std_x, std_y, std_z, is_meas = map(float, parts)
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
                   self.pred_line_full, self.pred_line_zoom,
                   self.pred_measurements_zoom, self.altitude_text]

        try:
            # Get current data
            with self.data_lock:
                current_pos = self.new_position
                current_pred = self.new_prediction

            # Update actual trajectory
            if current_pos:
                x, y, z = current_pos
                self.trajectory.append((x, y, z))

                xs, ys, zs = zip(*self.trajectory) if len(self.trajectory) > 1 else ([x], [y], [z])

                self.trajectory_line_full.set_data_3d(xs, ys, zs)
                self.satellite_dot_full.set_data_3d([x], [y], [z])
                self.trajectory_line_zoom.set_data_3d(xs, ys, zs)
                self.satellite_dot_zoom.set_data_3d([x], [y], [z])

                # Dynamic zoom window
                current_dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                zoom_width = max(500000, current_dist / 3)
                elev_angle = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
                self.ax_zoom.view_init(elev=elev_angle, azim=np.degrees(np.arctan2(y, x)))

                if len(self.trajectory) >= 2:
                    (x0, y0, z0), (x1, y1, z1) = self.trajectory[-2], self.trajectory[-1]
                    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

                    norm = np.linalg.norm([dx, dy, dz])
                    if norm > 0:
                        dx, dy, dz = dx / norm, dy / norm, dz / norm

                        # Compute trailing view
                        azim = (np.degrees(np.arctan2(dy, dx)) + 180) % 360
                        elev = 20

                        self.ax_zoom.view_init(elev=elev, azim=azim)

                self.ax_zoom.set_xlim(x - zoom_width / 4, x + zoom_width / 4)
                self.ax_zoom.set_ylim(y - zoom_width / 4, y + zoom_width / 4)
                self.ax_zoom.set_zlim(z - zoom_width / 4, z + zoom_width / 4)

                # Update altitude display
                altitude = current_dist - self.earth_radius
                self.altitude_text.set_text(f"Altitude: {altitude / 1000:.1f} km\n"
                                            f"Distance: {current_dist / 1000:.1f} km\n"
                                            f"Position: ({x / 1000:.1f}, {y / 1000:.1f}, {z / 1000:.1f}) km")

                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(f"IMPACT!\n"
                                                f"Final altitude: {altitude:.1f} m\n"
                                                f"Final Position: ({x / 1000:.1f}, {y / 1000:.1f}, {z / 1000:.1f}) km")
                    return artists

            # Update predicted trajectory with uncertainty
            if current_pred:
                pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas = current_pred
                self.predictions.append((pred_x, pred_y, pred_z, std_x, std_y, std_z, is_meas))

                if len(self.predictions) > 1:
                    # Extract components from predictions
                    pred_points = np.array([(x, y, z) for x, y, z, _, _, _, _ in self.predictions])
                    pred_xs, pred_ys, pred_zs = pred_points.T
                    std_devs = [(sx, sy, sz) for _, _, _, sx, sy, sz, _ in self.predictions]
                    meas_flags = [m for _, _, _, _, _, _, m in self.predictions]

                    # Update predicted trajectory lines
                    self.pred_line_full.set_data_3d(pred_xs, pred_ys, pred_zs)
                    self.pred_dot_full.set_data_3d([pred_x], [pred_y], [pred_z])
                    self.pred_line_zoom.set_data_3d(pred_xs, pred_ys, pred_zs)
                    self.pred_dot_zoom.set_data_3d([pred_x], [pred_y], [pred_z])

                    # Remove old uncertainty tube
                    if hasattr(self, 'uncertainty_tube'):
                        self.uncertainty_tube.remove()

                    # Create new uncertainty tube
                    vertices, faces = self.create_uncertainty_tube(pred_points, std_devs)
                    vertices = np.array(vertices)
                    self.uncertainty_tube = Poly3DCollection(
                        [vertices[face_idx] for face_idx in faces],
                        alpha=0.15,
                        color='blue',
                        linewidths=0.5,
                        edgecolor='blue'
                    )

                    self.ax_zoom.add_collection3d(self.uncertainty_tube)
                    artists.append(self.uncertainty_tube)

                    # Update measurement points
                    meas_points = pred_points[np.array(meas_flags, dtype=bool)]
                    if len(meas_points) > 0:
                        meas_xs, meas_ys, meas_zs = meas_points.T
                        self.pred_measurements_zoom.set_data_3d(meas_xs, meas_ys, meas_zs)

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

# vis = Visualiser2D('trajectory.txt', 'predicted_trajectory.txt', break_point=100)
# vis.visualise()

vis = Visualiser3D('trajectory_3d.txt', 'predicted_trajectory_3d.txt', break_point=0, MAX_STEPS=100)
vis.visualise()