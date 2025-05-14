from dosts.Visualiser import Visualiser2D

true_traj_file = f"Trajectories/2d_true_trajectory.txt"
pred_traj_file = f"Trajectories/2d_pred_trajectory.txt"
crash_heat_file = f"Trajectories/2d_crash_heatmap_data.txt"

vis = Visualiser2D(true_traj_file, pred_traj_file, crash_heat_file, mode='prewritten')
vis.visualise()