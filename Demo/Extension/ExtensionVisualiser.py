from dosts.Visualiser import Visualiser3D, Visualiser3DSimple

true_traj_file = f"Trajectories/3d_true_trajectory.txt"
pred_traj_file = f"Trajectories/3d_pred_trajectory.txt"
crash_heatmap_file = f"Trajectories/3d_crash_heatmap_data.txt"
thrust_crash_heatmap_file = f"Trajectories/3d_thrust_crash_heatmap_data.txt"
thrust_f = f"Trajectories/3d_1kt_true_trajectory.txt"

# vis = Visualiser3D(true_traj_file, pred_traj_file, crash_heatmap_file, thrust_crash_heatmap_file)
# vis.visualise()

vis = Visualiser3DSimple(true_traj_file, thrust_f)
vis.visualise()