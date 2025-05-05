rs, vis_ang = distribute_radars2D(100, 6371)
radars = []

t = np.linspace(0,6*np.pi,100)
r = np.linspace(10000, 6371, 100)
traj = r*np.array([np.cos(t),np.sin(t)])

r = np.sqrt(traj[0]**2 + traj[1]**2)
theta = np.angle(traj[0]+traj[1]*1j)

for i in range(len(rs)):
    radars.append(Radar(i, rs[i]))


for rad in radars:
    for j,tau in enumerate(t):
        #print(f'{tau:.1f}, {r[j]:.1f}, {theta[j]:.1f}')
        rad.record_satellite(tau, [r[j], theta[j]])
    rad.add_noise()

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(traj[0], traj[1], '-o')
plot_2D(ax, rs)
plt.show()


true_traj = np.array([t,r,theta]).T

noisy_traj = combine_radar_measurements(radars, true_traj)
file = 'noisy_traj.txt'
write_to_file(file, noisy_traj)

noisy_r = noisy_traj[:,1]
noisy_theta = noisy_traj[:,2]
noisy_x = noisy_r*np.cos(noisy_theta)
noisy_y = noisy_r*np.sin(noisy_theta)
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(noisy_x, noisy_y, '-o')
plot_2D(ax, rs)
plt.show()
