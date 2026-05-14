import numpy as np
import matplotlib.pyplot as plt

data = np.load('./data/latent_data.npz')
z_all = data['z_all']
y_all = data['y_all']

X = z_all[:, :3]
y = y_all
y_int = y.astype(int)


alphas = np.linspace(0.35, 3.55, 700)  # ind = 96,389,692
z0 = z_all[96]
z1 = z_all[389]
z2 = z_all[692]

# generate a linear interpolation between z0 and z1, z1 and z2
n = 50
alpha = np.linspace(0, 1, n)[:, None]  # Shape: (n, 1)
z_interp_01 = (1 - alpha) * z0 + alpha * z1  # Broadcasting
z_interp_12 = (1 - alpha) * z1 + alpha * z2  # Broadcasting
z_interp = np.concatenate((z_interp_01, z_interp_12), axis=0)

z_cdw = np.array([-6.0038,  0.9378, -0.0133])
z_interp_cdw = (1 - alpha) * z1 + alpha * z_cdw


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(z_all[:700, 0], z_all[:700, 1], z_all[:700, 2], c=y_all[:700], cmap='RdBu',
#                 edgecolors='none',linewidths=0)
# ax.plot(z_interp[:, 0], z_interp[:, 1], z_interp[:, 2], color='black', linewidth=2, label='Interpolation Path',linestyle='--')
# # set limit
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])
# ax.set_zlim([-3, 3])
# ax.set_xlabel(r'$z_1$', fontsize=12)
# ax.set_ylabel(r'$z_2$', fontsize=12)
# ax.set_zlabel(r'$z_3$', fontsize=12)
# plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(z_all[:, 0], z_all[:, 1], z_all[:, 2], c=y_all[:], cmap='RdBu',
                edgecolors='none',linewidths=0)
ax.scatter(z1[0], z1[1], z1[2], color='blue', s=300, marker='*', edgecolor='none', label='Start')
ax.scatter(z_cdw[0], z_cdw[1], z_cdw[2], color='red', s=300, marker='*', edgecolor='none', label='End')
ax.plot(z_interp_cdw[:, 0], z_interp_cdw[:, 1], z_interp_cdw[:, 2], color='black', linewidth=2, label='Interpolation Path',linestyle='--')
# set limit
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.set_xlabel(r'$z_1$', fontsize=12)
ax.set_ylabel(r'$z_2$', fontsize=12)
ax.set_zlabel(r'$z_3$', fontsize=12)
plt.show()