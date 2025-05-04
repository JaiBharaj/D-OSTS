import numpy as np
import matplotlib.pyplot as plt

def distribute_radars2D(Hdark, R_earth):
    distribution_angle = np.arccos(R_earth/(R_earth + Hdark))
    num_stations = int(np.ceil(np.pi/distribution_angle))
    interval_points = np.linspace(0, 2 * np.pi, num_stations, endpoint=False)
    radar_coors = np.column_stack((np.full(num_stations,R_earth), interval_points))
    return radar_coors

def fibonacci_sphere_grid(N):
    gold= 0.5*(1+np.sqrt(5))
    ind = np.array([i for i in range(0,N)])
    x = (ind/gold)%gold
    y = ind/(N-1)
    theta = 2*np.pi*x
    phi = np.arccos(1-2*y)
    return theta, phi

def distribute_radars3D(H, R=6371):
    theta = np.arccos(R/(R+H)) # angle of visibility
    area_cap = 0.5*(1-np.cos(theta))
    num_stations = int(np.ceil( 1/area_cap ))
    theta, phi = fibonacci_sphere_grid(num_stations)
    return np.column_stack((np.full(num_stations,R), theta, phi))

# optional plotting
def plot2D(rs):
  R=6371
  r, theta = rs.T
  x = r * np.cos(theta)
  y = r * np.sin(theta)
  circle_theta = np.linspace(0, 2 * np.pi, 200)
  circle_x = R * np.cos(circle_theta)
  circle_y = R * np.sin(circle_theta)
  
  fig = plt.figure(figsize=(8,8))
  
  plt.plot(circle_x,circle_y)
  plt.scatter(x,y, color='r', s=60)
  plt.show()

def plot3D(rs, interactive=False):
  R=6371
  r, theta, phi = rs3D.T
  x = R * np.cos(theta)*np.sin(phi)
  y = R * np.sin(theta)*np.sin(phi)
  z = R * np.cos(phi)
  
  u = np.linspace(0, 2 * np.pi, 40)
  v = np.linspace(0, np.pi, 40)
  U,V = np.meshgrid(u,v)
  X = R * np.cos(U)*np.sin(V)
  Y = R * np.sin(U)*np.sin(V)
  Z = R * np.cos(V)

  from matplotlib import cm
  from mpl_toolkits.mplot3d import Axes3D
  if interactive:
    %matplotlib qt
  
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(projection='3d')
  
  ax.plot_surface(X,Y,Z, cmap=cm.Blues, alpha=0.95)
  ax.scatter(x,y,z, color='r', s=5)
  ax.set_aspect('equal')
  #ax.set_axis_off() #optional
  plt.show()
