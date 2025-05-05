import numpy as np

def weighted_average(x, noise):
    weights = 1 / noise ** 2
    return np.sum(weights * x) / np.sum(weights)

def combine_radar_measurements(radars, true_traj):
    times = true_traj[:, 0]
    theta_arr = true_traj[:, 2]
    r_arr = np.zeros(len(times))

    for i, time in enumerate(times):
        seen_radius = []
        noise = []
        for rad in radars:
            is_visible = rad.satellite_measurements['visibility'][i]
            if not is_visible:
                continue
            seen_radius.append(rad.satellite_measurements['r'][i])
            noise.append(rad.get_noise()[i])

        if len(seen_radius) == 0:
            r_arr[i] = np.nan  # <-- key fix: no visible radars
        else:
            seen_radius = np.array(seen_radius)
            noise = np.array(noise)
            r_arr[i] = weighted_average(seen_radius, noise)

    return np.array([times, r_arr, theta_arr]).T


def write_to_file(file_name, traj):
    with open(file_name, 'w') as file:
        for (t, r, theta) in zip(traj[:, 0], traj[:, 1], traj[:, 2]):
            file.write(f"{t} {r} {theta}\n")

'''
# example use
true_traj = np.array([t, r, theta]).T
noisy_traj = combine_radar_measurements(radars, true_traj)
file = 'noisy_traj.txt'
write_to_file(file, noisy_traj)
'''