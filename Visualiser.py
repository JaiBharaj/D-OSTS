import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from CoordinateTransformations import InitialConditions
import threading

class BasicOperations:

    @staticmethod
    def cart_to_polar(x, y):
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(y / x)
        return r, theta

    @staticmethod
    def polar_to_cart(r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    @staticmethod
    def cart_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y / x)
        phi = np.arccos(z / r)
        return r, theta, phi

    @staticmethod
    def spherical_to_cart(r, theta, phi):
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z


# Earth parameters
earth_radius = InitialConditions.earthRadius
stop_distance = earth_radius + 100  # 100m above surface
initial_altitude = InitialConditions.initSatAlt

# File paths
TRAJECTORY_FILE = 'trajectory.txt'
PREDICTION_FILE = 'predicted_trajectory.txt'

# Plot layout
fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Satellite Trajectory with Prediction Uncertainty', fontsize=14, y=0.98)

# Configure axes
for ax in (ax_full, ax_zoom):
    ax.set_aspect('equal')
    earth_circle = plt.Circle((0, 0), earth_radius, color='blue', alpha=0.3, zorder=1)
    ax.add_patch(earth_circle)
    ax.grid(True, alpha=0.3)

# Set full view limits
plot_radius = 7E+6
ax_full.set_xlim(-plot_radius, plot_radius)
ax_full.set_ylim(-plot_radius, plot_radius)
ax_full.set_title('Full Trajectory View')
ax_full.set_xlabel('X position (m)')
ax_full.set_ylabel('Y position (m)')

# Initialise satellite elements
trajectory_line_full, = ax_full.plot([], [], 'r-', lw=1.5, zorder=3, label='Actual Trajectory')
satellite_dot_full, = ax_full.plot([], [], 'ro', markersize=5, zorder=4)
trajectory_line_zoom, = ax_zoom.plot([], [], 'r-', lw=1.5, zorder=3)
satellite_dot_zoom, = ax_zoom.plot([], [], 'ro', markersize=5, zorder=4)

# Initialise predicted satellite elements
pred_line_full, = ax_full.plot([], [], 'b-', lw=1.2, alpha=0.9, zorder=5, label='Predicted')
pred_line_zoom, = ax_zoom.plot([], [], 'b-', lw=1.2, alpha=0.9, zorder=5)

# Initialise uncertainty region
pred_uncertainty_zoom = ax_zoom.fill_between([], [], [], color='blue', alpha=0.15, zorder=2)
pred_measurements_zoom, = ax_zoom.plot([], [], 'go', markersize=6, zorder=6, label='Measurements')

ax_zoom.set_title('Zoomed View with Uncertainty')
ax_zoom.legend(loc='upper right')

# Altitude display
altitude_text = ax_zoom.text(0.02, 0.98, "Altitude: Initializing...",
                             transform=ax_zoom.transAxes, ha='left', va='top',
                             fontsize=11, bbox=dict(facecolor='white', alpha=0.7))

# Data storage
trajectory = []
predictions = []
data_lock = threading.Lock()
new_position = None
new_prediction = None

def read_next_position():
    with open(TRAJECTORY_FILE, 'r') as f:
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


def read_next_prediction():
    with open(PREDICTION_FILE, 'r') as f:
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


def load_data():
    global new_position, new_prediction
    pos_gen = read_next_position()
    pred_gen = read_next_prediction()

    while True:
        try:
            with data_lock:
                new_position = next(pos_gen)
                new_prediction = next(pred_gen)
        except StopIteration:
            pass
        time.sleep(0.05)


def update(frame):
    global trajectory, predictions, new_position, new_prediction

    artists = [trajectory_line_full, satellite_dot_full,
               trajectory_line_zoom, satellite_dot_zoom,
               pred_line_full, pred_line_zoom,
               pred_measurements_zoom, altitude_text]

    try:
        # Get current data
        with data_lock:
            current_pos = new_position
            current_pred = new_prediction

        # Update actual trajectory
        if current_pos:
            x, y = current_pos
            trajectory.append((x, y))

            xs, ys = zip(*trajectory) if len(trajectory) > 1 else ([x], [y])

            trajectory_line_full.set_data(xs, ys)
            satellite_dot_full.set_data([x], [y])
            trajectory_line_zoom.set_data(xs, ys)
            satellite_dot_zoom.set_data([x], [y])

            # Dynamic zoom window
            current_dist = np.hypot(x, y)
            zoom_width = max(500000, current_dist / 3)
            ax_zoom.set_xlim(x - zoom_width / 6, x + zoom_width / 6)
            ax_zoom.set_ylim(y - zoom_width / 6, y + zoom_width / 6)

            # Update altitude display
            altitude = current_dist - earth_radius
            altitude_text.set_text(f"Altitude: {altitude / 1000:.1f} km\n"
                                   f"Distance: {current_dist / 1000:.1f} km")

            # Check for impact
            if current_dist <= stop_distance:
                ani.event_source.stop()
                altitude_text.set_text(f"IMPACT!\nFinal altitude: {altitude:.1f} m")
                return artists

        # Update predicted trajectory with uncertainty
        if current_pred:
            pred_x, pred_y, std_x, std_y, is_meas = current_pred
            predictions.append((pred_x, pred_y, std_x, std_y, is_meas))

            # Keep only recent predictions (for performance)
            if len(predictions) > 75:
                predictions = predictions[-75:]

            if predictions:
                pred_xs, pred_ys, std_xs, std_ys, meas_flags = zip(*predictions)

                # Update prediction lines
                pred_line_full.set_data(pred_xs, pred_ys)
                pred_line_zoom.set_data(pred_xs, pred_ys)

                # Calculate uncertainty bounds
                upper = [y + std for y, std in zip(pred_ys, std_ys)]
                lower = [y - std for y, std in zip(pred_ys, std_ys)]

                # Update uncertainty region
                for coll in ax_zoom.collections[:]:
                    if coll not in artists:
                        coll.remove()
                ax_zoom.fill_between(pred_xs, lower, upper, color='blue', alpha=0.15, zorder=2)

                # Update measurement points
                meas_xs = [x for x, flag in zip(pred_xs, meas_flags) if flag]
                meas_ys = [y for y, flag in zip(pred_ys, meas_flags) if flag]
                pred_measurements_zoom.set_data(meas_xs, meas_ys)

    except Exception as e:
        print(f"Animation error: {e}")

    return artists


# Start data loading thread
data_thread = threading.Thread(target=load_data, daemon=True)
data_thread.start()

# Create animation
ani = animation.FuncAnimation(
    fig, update,
    frames=1000,
    interval=100,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()