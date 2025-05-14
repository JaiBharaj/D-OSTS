# Visualiser

This Visualiser is a visual tool for displaying, analyzing, and interacting with satellite trajectories, predictions, uncertainties, and crash risk heatmaps in real-time using both 2D and 3D animated views. Consists of two classes:
- `Visualiser2D`: visualize the **true** and **predicted** trajectories of a satellite in 2D space.
- `Visualiser3D`: visualize the **true** and **predicted** trajectories of a satellite in 3D space.

---

```python
def __init__(trajectory_file_path, 
            prediction_file_path, 
            heatmap_file_path=None, 
            measurement_times=None, 
            break_point=0, mode='prewritten'):
  ...
```
- **Parameters**:
  * `trajectory_file_path`: Path to the file containing the true trajectory.
  * `prediction_file_path`: Path to the file containing predicted trajectory and uncertainty.
  * `heatmap_file_path`: (optional) File with angular crash data for heatmap rendering.
  * `measurement_times`: (optional) Times when measurements are expected.
  * `break_point`: Distance above Earth's radius to define a "crash".
  * `mode`: `'prewritten'` or `'realtime'` to control file reading mode.

---

```python
def setup_plots(self):
  ...
```
- **Purpose**: Initializes the full and zoomed-in plots, Earth circle, legends, and layout.

---

```python
def init_colorbar(self):
  ...
```
- **Purpose**: Adds a vertical colorbar to indicate heatmap crash likelihood scale.

---

```python
def on_key_press(self, event):
  ...
```
- **Purpose**: Captures key presses for controlling the view.
- **Keys**:
  * `'t'`: Focus on true trajectory.
  * `'p'`: Focus on predicted trajectory.
  * `'+'`, `'up'`: Zoom in.
  * `'-'`, `'down'`: Zoom out.

---

```python
def read_next_position(self):
  ...
```
- **Purpose**:Generator yielding the next position `(x, y, z) from the trajectory file, converting from spherical to Cartesian coordinates.

---

```python
def read_next_prediction(self):
  ...
```
- **Purpose**: Reads predicted position and uncertainty from file.

---

```python
def load_data(self):
  ...
```
- **Purpose**: Continuously feeds data into queues from the trajectory and prediction files.

---

```python
def direction_of_motion(self, x1, y1, x2, y2):
  ...
  return tangent, normal
```
- **Purpose**: Calculates tangent and normal vectors between two points.
- **Parameters**: 
  - `x1, y1`: Position of point 1.
  - `x2, y2`: Position of point 2.
- **Returns**:
  * `tangent`: Normalized direction vector of motion.
  * `normal`: Perpendicular to tangent (used in uncertainty rendering).

---

```python
def load_heatmap_data(self):
  ...
```
- **Purpose**: Parses `HEATMAP_FILE` to build a time-annotated list of angular crash samples.

---

```python
def update_heatmap(self, current_time):
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
- **Returns**: A list of updated artists (lines, patches, markers) to be redrawn by the animation system.

---

```python
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
```
- **Purpose**: Launches the full visualisation system.
- **Implementations**:
  * Starts data loader thread.
  * Loads heatmap data.
  * Creates a Matplotlib animation with update frames.
  * Displays the interactive plot window.

---

```python
def plot_earth(self, ax):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(u, v)
    x, y, z = spherical_to_cartesian(self.earth_radius, theta, phi)
    ax.plot_surface(x, y, z, color='gray', alpha=0.3)
    self.add_earth_grid(ax)
```
- **Purpose**: Plots a semi-transparent 3D Earth sphere on the given axis using a spherical mesh.

---

```python
def add_earth_grid(self, ax):
  ...
```
- **Purpose**: Overlays meridians and parallels on the Earth’s surface for geographic reference, including labeled degree markers.

---

```python
def set_axes_equal(self, ax):
  ...
```
- **Purpose**: Ensures all 3D axes have equal scaling for correct spatial perception.

---

```python
def create_uncertainty_tube(self, points, std_devs):
  ...
```
- **Purpose**: Creates a 3D polygonal mesh representing a tube around the predicted trajectory, visualizing spatial uncertainty via standard deviation radii.
