import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

pose = pd.read_csv("./runs_mars/movingcars/pose.txt", delim_whitespace=True)
x = pose.iloc[:,7]
y = pose.iloc[:,8]
z = pose.iloc[:,9]
trackID = pose.iloc[:,2]

condition = trackID.astype(int) == trackID[0]
x_filtered = x[condition]
#x_filtered = x_filtered - 3.0
y_filtered = y[condition]
z_filtered = z[condition]

condition = trackID.astype(int) == trackID[1]
x1_filtered = x[condition]
#x1_filtered = x1_filtered - 3.0
y1_filtered = y[condition]
z1_filtered = z[condition]

ext = pd.read_csv("./runs_mars/movingcars/extrinsic.txt", delim_whitespace=True)
ego_x = ext.iloc[1:,5]
ego_y = ext.iloc[1:,9]
ego_z = ext.iloc[1:,13]
camID = ext.iloc[1:,1]
condition = camID.astype(int) == 0
ego_x = ego_x[condition]
ego_y = ego_y[condition]
ego_z = ego_z[condition]

fig=plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()

ax.scatter(x_filtered,y_filtered,z_filtered, c='gray', label='White Van')
ax.scatter(x1_filtered,y1_filtered,z1_filtered, c='r', label='Red Car')
ax.scatter(ego_x, ego_y, ego_z, c='b', label='Camera')

ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

ax.legend()
plt.show()


