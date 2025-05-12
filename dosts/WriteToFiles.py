# function to write 2D trajectory to file
def write_to_file_2d(file_name, traj):
    with open(file_name, 'w') as file:
        for (t, r, theta) in zip(traj[:, 0], traj[:, 1], traj[:, 2]):
            file.write(f"{t} {r} {theta}\n")
    print(f"Wrote {traj.shape[0]} points to {file_name}")

# function to write 3D trajectory to file
def write_to_file_3d(file_name, traj):
    with open(file_name, 'w') as file:
        for (t, r, theta, phi) in zip(traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]):
            file.write(f"{t} {r} {theta} {phi}\n")
    print(f"Wrote {traj.shape[0]} points to {file_name}")