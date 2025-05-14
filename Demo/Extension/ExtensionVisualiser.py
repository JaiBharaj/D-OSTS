from dosts.Visualiser import Visualiser3D

true_traj_file = f"Trajectories/3d_true_trajectory.txt"
pred_traj_file = f"Trajectories/3d_pred_trajectory.txt"

vis = Visualiser3D(true_traj_file, pred_traj_file)
vis.visualise()