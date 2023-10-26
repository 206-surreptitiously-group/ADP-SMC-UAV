import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


traj = pd.read_csv('./uniform_mc_ref_traj1.csv', header=0).to_numpy()
r1 = pd.read_csv('./uniform_mc_test_smc.csv', header=0).to_numpy()
r2 = pd.read_csv('./uniform_mc_test_rl.csv', header=0).to_numpy()
X = traj[:, 0] * 180 / np.pi	# 所有的振幅
Y = traj[:, 3]					# 所有的周期
Z1 = r1[:, 0]
Z2 = r2[:, 0]

print(X.shape, Y.shape, Z1.shape, Z2.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(X, Y, Z1, label='')
plt.show()
