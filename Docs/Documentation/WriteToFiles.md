# File Writer

This module provides utility functions to export satellite trajectory data (either in 2D or 3D) to text files.

---

```python
def write_to_file_2d(file_name, traj):
    with open(file_name, 'w') as file:
        for (t, r, theta) in zip(traj[:, 0], traj[:, 1], traj[:, 2]):
            file.write(f"{t} {r} {theta}\n")
    print(f"Wrote {traj.shape[0]} points to {file_name}")
```
- **Purpose**: Writes a 2D satellite trajectory to a text file, with each row containing time, radius, and angle values.
- **Parameters**:
  - `file_name`: Name of the output file.
  - `traj`: NumPy array of shape `(N, 3)` containing `[t, r, θ]` values.
- **Implementation**:
  - Opens the file for writing.
  - Iterates over trajectory entries and writes each row to file.
  - Each line format: `t r θ`
- **Output**: Saves a text file and prints the number of points written.

---

```python
def write_to_file_3d(file_name, traj):
    with open(file_name, 'w') as file:
        for (t, r, theta, phi) in 
        zip(traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]):
            file.write(f"{t} {r} {theta} {phi}\n")
    print(f"Wrote {traj.shape[0]} points to {file_name}")
```
- **Purpose**: Writes a 3D satellite trajectory to a text file, with each row containing time, radius, azimuthal angle, and polar angle.
- **Parameters**:
  - `file_name`: Name of the output file.
  - `traj`: NumPy array of shape `(N, 4)` containing `[t, r, θ, φ]` values.
- **Implementation**:
  - Similar to the 2D function, but includes the additional φ (polar angle) column.
  - Each line format: `t r θ φ`.
- **Output**: Saves a text file and prints the number of points written.
