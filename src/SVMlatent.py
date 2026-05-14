import numpy as np
from sklearn.svm import SVC


data = np.load('./data/latent_data.npz')
z_all = data['z_all']
y_all = data['y_all']

X = z_all[:, :3]
y = y_all
y_int = y.astype(int)

# Train an SVM classifier
clf = SVC(kernel='rbf', C=10).fit(X, y)

accuracy = clf.score(X, y)
print(f"Training accuracy: {accuracy:.4f}")


# Create a 3D grid of points
n_grid = 100  # Increase for higher resolution, but will be slower
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, n_grid),
    np.linspace(y_min, y_max, n_grid),
    np.linspace(z_min, z_max, n_grid)
)

grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
decision_values = clf.decision_function(grid)

# Find points near the decision boundary (decision function ~ 0)
boundary_points = grid[np.abs(decision_values) < 0.05]  # Adjust threshold for thickness

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='RdBu', edgecolor='none', alpha=0.5,linewidths=0)
if len(boundary_points) > 0:
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
               color='black', s=1, alpha=0.2)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])
ax.set_xlabel(r'$z_1$', fontsize=12)
ax.set_ylabel(r'$z_2$', fontsize=12)
ax.set_zlabel(r'$z_3$', fontsize=12)

plt.savefig("/Users/angkunwu/Desktop/svm_latent.pdf")
plt.show()


# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "browser"
# fig = go.Figure()
# # Add latent points
# fig.add_trace(go.Scatter3d(
#     x=X[:, 0], y=X[:, 1], z=X[:, 2],
#     mode='markers',
#     marker=dict(size=3, color=y_int, colorscale='RdBu', opacity=0.5),
#     name='Latent Points'
# ))

# # Add boundary points if you have them
# if len(boundary_points) > 0:
#     fig.add_trace(go.Scatter3d(
#         x=boundary_points[:, 0], y=boundary_points[:, 1], z=boundary_points[:, 2],
#         mode='markers',
#         marker=dict(size=1, color='black', opacity=0.2),
#         name='SVM Boundary'
#     ))

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(title='z₁', range=[-5, 5]),
#         yaxis=dict(title='z₂', range=[-5, 5]),
#         zaxis=dict(title='z₃', range=[-5, 5])
#     ),
#     title='3D Latent Space with SVM Decision Boundary',
#     font=dict(size=12)
# )
# fig.show()
#fig.write_image("/Users/angkunwu/Desktop/svm_latent.pdf")
