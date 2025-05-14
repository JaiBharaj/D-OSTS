# Visualiser

This Visualiser is a highly visual tool for displaying, analyzing, and interacting with satellite trajectories, predictions, uncertainties, and crash risk heatmaps in real-time or playback modes using both 2D and 3D animated views.

## Class: `Visualiser2Dr`

This module defines the `Visualiser2D` class used to visualize the **true** and **predicted** trajectories of a satellite in 2D space.
It supports uncertainty rendering, heatmap overlays for crash probabilities, and live/replay modes.
The visualizer is typically used during simulation or prediction evaluation of atmospheric re-entry scenarios.

---

```python
def __init__(trajectory_file_path, prediction_file_path, heatmap_file_path=None, measurement_times=None, break_point=0, mode='prewritten')
```

- **Purpose**: Initializes the visualisation engine with required file paths and simulation configuration.
- **Parameters**:

  * `trajectory_file_path`: Path to the file containing the true trajectory.
  * `prediction_file_path`: Path to the file containing predicted trajectory and uncertainty.
  * `heatmap_file_path`: (optional) File with angular crash data for heatmap rendering.
  * `measurement_times`: (optional) Times when measurements are expected.
  * `break_point`: Distance above Earth's radius to define a "crash".
  * `mode`: `'prewritten'` or `'realtime'` to control file reading mode.

---

```python
def setup_plots(self)
    ...
```

- **Purpose**: Initializes the full and zoomed-in plots, Earth circle, legends, and layout.

---

```python
def init_colorbar(self)
...
```

- **Purpose**: Adds a vertical colorbar to indicate heatmap crash likelihood scale.

---

```python
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
```

- **Purpose**: Captures key presses for controlling the view.
- **Keys**:

  * `'t'`: Focus on true trajectory.
  * `'p'`: Focus on predicted trajectory.
  * `'+'`, `'up'`: Zoom in.
  * `'-'`, `'down'`: Zoom out.

---

```python
def read_next_prediction(self)
...
```

- **Purpose**: Reads predicted position and uncertainty, returning:

  ```python
  (time, x, y, std_x, std_y, is_meas)
  ```
- **Implementation**:

  * Converts from polar + uncertainty to Cartesian + error ellipse.
  * Marks whether the point is a measurement.

---

```python
def load_data(self)
...
```

- **Purpose**: Continuously feeds data into queues from the trajectory and prediction files.
- **Implementation**: Runs in a background thread.

---

```python
def direction_of_motion(self, x1, y1, x2, y2)
    dx = x2 - x1
    dy = y2 - y1
    norm = np.hypot(dx, dy)
    if norm == 0:
        return np.array([1, 0]), np.array([0, 1])  
        tangent = np.array([dx, dy]) / norm
        normal = np.array([-tangent[1], tangent[0]])
        return tangent, normal
```

- **Purpose**: Calculates tangent and normal vectors between two points.
- **Returns**:

  * `tangent`: Normalized direction vector of motion.
  * `normal`: Perpendicular to tangent (used in uncertainty rendering).

---

```python
def load_heatmap_data(self)
...
```

- **Purpose**: Parses `HEATMAP_FILE` to build a time-annotated list of angular crash samples.
- **File format**:

  ```
  <time> <angle1> <angle2> ...
  ```

---

```python
def update_heatmap(self, current_time)
...
```

- **Purpose**: Generates a radial histogram (360 wedges) showing the density of predicted crash angles up to `current_time`.
- **Implementation**:

  * Computes histogram over `[0, 2π]`.
  * Draws each bin as a colored wedge.
  * Updates the heatmap colorbar scale.

---

```python
def update(self, frame):
    ...
```

- **Purpose**
  Updates the satellite trajectory animation for a given frame. This includes:

  * Visualizing new actual and predicted positions
  * Dynamically adjusting zoom view
  * Updating uncertainty regions
  * Rendering measurement points
  * Displaying live crash heatmaps (if enabled)

- **Parameters**

  * `frame` (`int`):
    Current animation frame index (required by `matplotlib.animation.FuncAnimation`, though unused explicitly).

- **Procedure**

  1. **Update Actual Trajectory**

     * Dequeues new position data from `position_queue`.
     * Updates full and zoom views.
     * Tracks current altitude.
     * Stops animation if satellite altitude drops below impact threshold.

  2. **Update Predicted Trajectory**

     * Dequeues new prediction data from `prediction_queue`.
     * Updates position of predicted path and marker.
     * Computes and redraws the uncertainty region as a closed polygon based on local standard deviations (`std_x`, `std_y`).
     * Displays measurement points if available.

  3. **Update Heatmap Overlay (if enabled)**

     * Aggregates crash angles up to the current prediction time.
     * Re-renders a circular histogram (wedge-style heatmap) on the full trajectory view.

- **Returns**

  * `artists` (`list[matplotlib.Artist]`):
    A list of updated artists (lines, patches, markers) to be redrawn by the animation system.

- **Error Handling**

  * Catches and logs any runtime exceptions without halting the animation engine.

- **Internal State Modified**

  * `self.trajectory`: Appended with new actual positions.
  * `self.predictions`: Appended with new predicted points.
  * `self.uncertainty_polygon`: Updated polygon showing EKF prediction uncertainty.
  * `self.altitude_text`: Updated with live altitude and distance info.
  * `self.heatmap_artists`: Redrawn if a heatmap is active.


---

```python
def visualise(self)
...
```

- **Purpose**: Launches the full visualisation system.
- **Implementations**:

  * Starts data loader thread.
  * Loads heatmap data.
  * Creates a Matplotlib animation with update frames.
  * Displays the interactive plot window.

---

# : `Visualiser3D`

This module defines a `Visualiser3D` class for real-time or scripted visualization of satellite trajectories and predicted paths, including uncertainty volumes and crash probability heatmaps. It is primarily designed for debugging and analyzing satellite orbit simulations in 3D using `matplotlib`.

---

## `class Visualiser3D`

```python
def __init__(self, trajectory_file_path, prediction_file_path, heatmap_file_path=None, thrust_heatmap_file_path=None, break_point=0, mode='prewritten', MAX_STEPS=50000)
```

- **Purpose**:

Initializes a 3D visualization environment, sets up plot elements, reads external data sources, and prepares the animation loop.

- **Parameters**:

* `trajectory_file_path`: Path to file containing the satellite’s actual trajectory (in spherical coordinates).
* `prediction_file_path`: Path to file with predicted positions and uncertainties.
* `heatmap_file_path` *(optional)*: Path to nominal crash probability heatmap file.
* `thrust_heatmap_file_path` *(optional)*: Path to thrust-adjusted crash heatmap.
* `break_point`: Altitude threshold (in meters) below which the simulation stops.
* `mode`: `'prewritten'` for static file loading or `'realtime'` for streaming updates.
* `MAX_STEPS`: Maximum number of stored points for trajectories/predictions.

---


```python
def setup_plots(self)
...
```

- **Purpose**:

Creates two 3D subplots: a global view and a zoomed view with uncertainty overlays. Draws Earth, initializes plot lines and legends.

---

```python
def plot_earth(self, ax)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(u, v)
    x, y, z = spherical_to_cartesian(self.earth_radius, theta, phi)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)


    self.add_earth_grid(ax)
```

- **Purpose**:

Plots a semi-transparent 3D Earth sphere on the given axis using a spherical mesh.

---

```python
def add_earth_grid(self, ax)
...

```

- **Purpose**:

Overlays meridians and parallels on the Earth’s surface for geographic reference, including labeled degree markers.

---

```python
def set_axes_equal(self, ax)
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
```

- **Purpose**:

Ensures all 3D axes have equal scaling for correct spatial perception.

---


```python
def load_heatmap_data(self)
```

- **Purpose**:Parses and loads both nominal and thrust heatmap data from specified files. Each dataset includes timestamped θ-φ coordinate pairs.

---

```python
def update_heatmap(self, current_time)
...
```

- **Purpose**:Updates both heatmaps to reflect density up to the current time. Applies optional smoothing and uses color intensity to represent probability density.

---

```python
def update_colorbar(self)
...
```

- **Purpose**:Adds or updates colorbars for both nominal and thrust heatmaps, dynamically depending on data availability.

---


```python
def on_key_press(self, event)
```

- The same as above.
---


```python
def read_next_position(self):
    with open(self.TRAJECTORY_FILE, 'r') as f:
            if self.mode == 'prewritten':
                for line in f:
                    try:
                        _, r, theta, phi = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        yield x, y, z
                     
                    except ValueError:
                        continue
            else: '
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        f.seek(pos)
                        continue
                    try:
                        _, r, theta, phi = map(float, line.strip().split())
                        x, y, z = spherical_to_cartesian(r, theta, phi)
                        yield x, y, z
                    except ValueError:
                        continue

```

- **Purpose**:Generator yielding the next position `(x, y, z)` from the trajectory file, converting from spherical to Cartesian coordinates.

---

```python
def read_next_prediction(self)
...
```

- **Purpose**:Generator yielding predictions and associated uncertainties, converting them into Cartesian coordinates and standard deviations for visualization.

---

```python
def load_data(self)
```

- **Purpose**:Runs in a background thread to continuously fetch the latest position and prediction data and stores them safely for animation updates.

---


```python
def create_uncertainty_tube(self, points, std_devs)
```

- **Purpose**: Creates a 3D polygonal mesh representing a tube around the predicted trajectory, visualizing spatial uncertainty via standard deviation radii.

---

```python
def update(self, frame)
```

- **Purpose**:Per animation frame:

* Updates actual and predicted trajectory lines.
* Redraws the uncertainty tube.
* Updates heatmaps.
* Repositions the zoomed view camera.
* Detects and signals satellite impact.
* Dynamically updates on-screen text (altitude, coordinates).

Returns a list of updated `matplotlib` artists.

---

```python
def visualise(self)
```

- The same as above 2DVisualiser

---
