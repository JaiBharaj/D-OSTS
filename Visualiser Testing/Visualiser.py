import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from CrudeInitialConditions import InitialConditions
import threading

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
        self.new_position = None
        self.new_prediction = None
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

        # Set full view limits
        plot_radius = 7E+6
        self.ax_full.set_xlim(-plot_radius, plot_radius)
        self.ax_full.set_ylim(-plot_radius, plot_radius)
        self.ax_full.set_title('Full Trajectory View')
        self.ax_full.set_xlabel('X position (m)')
        self.ax_full.set_ylabel('Y position (m)')

        # Initialise plot elements
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

    def read_next_position(self):
        with open(self.TRAJECTORY_FILE, 'r') as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    f.seek(pos)
                    time.sleep(0.05)
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
                    time.sleep(0.05)
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
                with self.data_lock:
                    self.new_position = next(pos_gen)
                    self.new_prediction = next(pred_gen)
            except StopIteration:
                pass
            time.sleep(0.05)

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
                x, y = current_pos
                self.trajectory.append((x, y))

                xs, ys = zip(*self.trajectory) if len(self.trajectory) > 1 else ([x], [y])

                self.trajectory_line_full.set_data(xs, ys)
                self.satellite_dot_full.set_data([x], [y])
                self.trajectory_line_zoom.set_data(xs, ys)
                self.satellite_dot_zoom.set_data([x], [y])

                # Dynamic zoom window
                current_dist = np.hypot(x, y)
                zoom_width = max(500000, current_dist / 3)
                self.ax_zoom.set_xlim(x - zoom_width / 6, x + zoom_width / 6)
                self.ax_zoom.set_ylim(y - zoom_width / 6, y + zoom_width / 6)

                # Update altitude display
                altitude = current_dist - self.earth_radius
                self.altitude_text.set_text(f"Altitude: {altitude / 1000:.1f} km\n"
                                            f"Distance: {current_dist / 1000:.1f} km")

                if current_dist <= self.stop_distance:
                    self.ani.event_source.stop()
                    self.altitude_text.set_text(f"IMPACT!\nFinal altitude: {altitude:.1f} m")
                    return artists

            # Update predicted trajectory with uncertainty
            if current_pred:
                pred_x, pred_y, std_x, std_y, is_meas = current_pred
                self.predictions.append((pred_x, pred_y, std_x, std_y, is_meas))

                if self.predictions:
                    pred_xs, pred_ys, std_xs, std_ys, meas_flags = zip(*self.predictions)

                    # Update prediction lines
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
            interval=100,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Usage example:
vis = Visualiser2D('trajectory.txt', 'predicted_trajectory.txt', break_point=100)
vis.visualise()