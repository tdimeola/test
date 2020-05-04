#My 3d graph

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import csv
import pandas

figure = plt.figure()
axis = figure.add_subplot(111, projection = '3d')
	

colnames = ['x', 'y', 'z']
data = pandas.read_csv('trajectory_1.csv', names=colnames)

x = data.x.tolist()
y = data.y.tolist()
z = data.z.tolist()
z = np.array([z,z])

axis.plot_wireframe(x, y, z)

axis.set_xlabel('x-axis')
axis.set_ylabel('y-axis')
axis.set_zlabel('z-axis')

plt.show()
