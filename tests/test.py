import gmsh
import sys
import meshio
import numpy as np
import pyvista as pv
import dislocs as dl
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


# # initialize gmsh
# gmsh.initialize(sys.argv)
# gmsh.model.add("rectangle")

# # 4 coners of the rectangle
# p1 = gmsh.model.geo.addPoint(439.60017008978974, 3924.5362397775775, -5, meshSize=10)  # 设置不同网格尺寸
# p2 = gmsh.model.geo.addPoint(429.7520925596677, 3926.272721554247, -5, meshSize=10)
# p3 = gmsh.model.geo.addPoint(430.34600430580656, 3929.6409624425814, -14.396926207859083, meshSize=10)
# p4 = gmsh.model.geo.addPoint(440.1940818359286, 3927.904480665912, -14.396926207859083, meshSize=10)


# # define the edges of the rectangle
# l1 = gmsh.model.geo.addLine(p1, p2)
# l2 = gmsh.model.geo.addLine(p2, p3)
# l3 = gmsh.model.geo.addLine(p3, p4)
# l4 = gmsh.model.geo.addLine(p4, p1)

# # create a surface
# l_lines = [l1, l2, l3, l4]
# loop = gmsh.model.geo.addCurveLoop(l_lines)
# surface = gmsh.model.geo.addPlaneSurface([loop])

# # sync and generate mesh (without using Transfinite)
# gmsh.model.geo.synchronize()
# gmsh.model.mesh.generate(2)

# gmsh.write("rectangle_non_uniform.msh")

# gmsh.finalize()

# ============ read the file and visulize ============
mesh = meshio.read("rectangle_non_uniform.msh")

# get the triangles
triangles = np.array(mesh.cells_dict["triangle"])
points = np.array(mesh.points)

print(f"triangles: {triangles}")
print(f"points: {points}")

# plotter = pv.Plotter()
# mesh_pv = pv.PolyData(points, np.hstack([np.full((triangles.shape[0], 1), 3), triangles]))
# plotter.add_mesh(mesh_pv, show_edges=True, color="lightblue")

# plotter.view_isometric()
# plotter.show_axes()
# plotter.show()

#---------------------------------------------------------------------------------------------------- 
mu = 3.3e10
nu = 0.25

x = np.arange(400, 461, 2.5)
y = np.arange(3900, 3951, 2.5)

X, Y = np.meshgrid(x, y)

# Construct X, Y, 0
obs = np.zeros((X.size, 3)) 
obs[:, 0] = X.ravel()  
obs[:, 1] = Y.ravel() 
# obs[:, 2] = 0
obs[:, 2] = -0.0000000001  

# obs = np.array([
#     [450, 3916, 0],
#     [456, 4000, -1],
#     [434.6761313247287, 3935.404480665912, -5]
#     ])


uc = np.array([434.6761313247287, 3925.404480665912, -5])
model = np.array([[uc[0], uc[1], uc[2],   10, 10,   280, 70,    5, 5, 0]])
print(f"model: {model}")

[U, S, E, flags] = dl.rde(obs, model, mu, nu)


# Walk through all the tdes, get the vertexes 
triangle_vertices = np.array([points[triangle].flatten() for triangle in triangles])  #  (n, 9)

ss = np.full((len(triangles), 1), 5)      # (n, 1)
ds = np.full((len(triangles), 1), 5)      # (n, 1)
ts = np.full((len(triangles), 1), 0)      # (n, 1)
triangle_data = np.hstack([triangle_vertices, ss, ds, ts])
triangle_data2 = np.hstack([triangle_vertices, ss, ds, ts])


# print("(x1, y1, z1, x2, y2, z2, x3, y3, z3) :")
# print(triangle_vertices[:5])  # only print the first 5 rows to avoid too long output

[U2, S2, E2 ] = dl.tde(obs, triangle_data, mu, nu)
[U3, S3, E3 ] = dl.tde_meade(obs, triangle_data2, mu, nu)

#-------------------------------------------------------
U   = np.round(U, 12)
U2  = np.round(U2, 12)
U3  = np.round(U3, 12)
print(f"U = \n{U}")
print(f"U2 = \n{U2}")
print(f"U3 = \n{U3}")

S   = np.round(S, 12)
S2  = np.round(S2, 12)
S3  = np.round(S3, 12)
print(f"S = \n{S}")
print(f"S2 = \n{S2}")
print(f"S3 = \n{S3}")

U_2d  =  U.reshape(len(y), len(x), 3)
U2_2d = U2.reshape(len(y), len(x), 3)
U3_2d = U3.reshape(len(y), len(x), 3)

U_magnitude = np.sqrt(U_2d[:, :, 0]**2 + U_2d[:, :, 1]**2 + U_2d[:, :, 2]**2)
U2_magnitude = np.sqrt(U2_2d[:, :, 0]**2 + U2_2d[:, :, 1]**2 + U2_2d[:, :, 2]**2)
U3_magnitude = np.sqrt(U3_2d[:, :, 0]**2 + U3_2d[:, :, 1]**2 + U3_2d[:, :, 2]**2)

# create a triangulation object (using only x, y coordinates)
tri = Triangulation(points[:, 0], points[:, 1], triangles)


#  get the max value
max_value = np.max(U3_magnitude)
# get the index of the max value (flattened one-dimensional index)
max_index = np.argmax(U3_magnitude)

# one-dimensional index to two-dimensional index
row_index, col_index = np.unravel_index(max_index, U3_magnitude.shape)

# ---  plot_surface ---
fig = plt.figure(figsize=(20, 6))  

# U_magnitude (3D)
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_magnitude, cmap='rainbow', alpha=0.8)
ax1.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, 
                 color='lightblue', edgecolor='black', alpha=0.5)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('U from dl.rde')
ax1.view_init(elev=30, azim=45)

# U2_magnitude (3D)
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, U2_magnitude, cmap='rainbow', alpha=0.8)
ax2.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, 
                 color='lightblue', edgecolor='black', alpha=0.5)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('U2 from dl.tde')
ax2.view_init(elev=30, azim=45)

# U3_magnitude (3D)
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, U3_magnitude, cmap='rainbow', alpha=0.8)
ax3.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, 
                 color='lightblue', edgecolor='black', alpha=0.5)
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_title('U3 from dl.tde_meade')
ax3.view_init(elev=30, azim=45)

# colorbar
cbar = fig.colorbar(surf3, ax=[ax1, ax2, ax3], shrink=0.5, aspect=10, pad=0.15)
cbar.set_label('Displacement Magnitude (m)')

# adjust the colorbar layout to avoid overlap
plt.subplots_adjust(left=0.05, right=0.77, wspace=0.3)  

# save as PNG 
plt.savefig('displacement_field.png', dpi=300, bbox_inches='tight')

plt.show()



# fig = plt.figure(figsize=(24, 5))

# ax1 = fig.add_subplot(1, 4, 1)  
# cf1 = ax1.contourf(X, Y, U_magnitude, cmap='rainbow', levels=20)
# # fig.colorbar(cf1, ax=ax1, label='Displacement Magnitude (m)')
# ax1.set_xlabel('X (m)')
# ax1.set_ylabel('Y (m)')
# ax1.set_title('U from dl.rde')
# ax1.axis('equal')

# ax2 = fig.add_subplot(1, 4, 2)
# cf2 = ax2.contourf(X, Y, U2_magnitude, cmap='rainbow', levels=20)
# # fig.colorbar(cf2, ax=ax2, label='Displacement Magnitude (m)')
# ax2.set_xlabel('X (m)')
# ax2.set_ylabel('Y (m)')
# ax2.set_title('U2 from dl.tde')
# ax2.axis('equal')

# ax3 = fig.add_subplot(1, 4, 3)
# cf3 = ax3.contourf(X, Y, U3_magnitude, cmap='rainbow', levels=20)
# fig.colorbar(cf3, ax=ax3, label='Displacement Magnitude (m)')
# ax3.set_xlabel('X (m)')
# ax3.set_ylabel('Y (m)')
# ax3.set_title('U3 from dl.tde_meade')
# ax3.axis('equal')

# ax4 = fig.add_subplot(1, 4, 4, projection='3d') 
# ax4.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], triangles=triangles, 
#                  color='lightblue', edgecolor='black', linewidth=0.5)
# ax4.set_xlabel('X (m)')
# ax4.set_ylabel('Y (m)')
# ax4.set_zlabel('Z (m)')
# ax4.set_title('Triangular Mesh (3D)')

# ax4.view_init(elev=30, azim=45) 

# plt.tight_layout()
# plt.show()

# # ------------
# fig, axes = plt.subplots(1, 4, figsize=(24, 5))
#
# cf1 = axes[0].contourf(X, Y, U_magnitude, cmap='rainbow', levels=20)
# fig.colorbar(cf1, ax=axes[0], label='Displacement Magnitude (m)')
# axes[0].triplot(tri, 'k-', lw=1)
# axes[0].set_xlabel('X (m)')
# axes[0].set_ylabel('Y (m)')
# axes[0].set_title('U from dl.rde')
# axes[0].axis('equal')
#
# cf2 = axes[1].contourf(X, Y, U2_magnitude, cmap='rainbow', levels=20)
# fig.colorbar(cf2, ax=axes[1], label='Displacement Magnitude (m)')
# axes[1].triplot(tri, 'k-', lw=1)
# axes[1].set_xlabel('X (m)')
# axes[1].set_ylabel('Y (m)')
# axes[1].set_title('U2 from dl.tde')
# axes[1].axis('equal')
#
# cf3 = axes[2].contourf(X, Y, U3_magnitude, cmap='rainbow', levels=20)
# fig.colorbar(cf3, ax=axes[2], label='Displacement Magnitude (m)')
# axes[2].triplot(tri, 'k-', lw=1)
# axes[2].set_xlabel('X (m)')
# axes[2].set_ylabel('Y (m)')
# axes[2].set_title('U3 from dl.tde_meade')
# axes[2].axis('equal')
#
# axes[3].triplot(tri, 'k-', lw=1)
# axes[3].set_xlabel('X (m)')
# axes[3].set_ylabel('Y (m)')
# axes[3].set_title('Triangular Mesh')
# axes[3].axis('equal')
#
# plt.tight_layout()
# plt.show()


#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 

#9mu = 3.3e10
#9nu = 0.25
#9
#9# ----------------------------------------------------------------------------------------------------
#9# # ---------------  X axis ----------------
#9# model = np.array([
#9# [0, 0, -1, 4, 4,  90, 90,        5, 5, 5]  # -------- X axis
#9# ])
#9
#9# obs = np.array([
#9#     [1, -1, 0],
#9#     [1,  1, 0],
#9# ])
#9# model_tde = np.array([ [-2,  0, -1,    2,  0, -1,     2,  0, -5,    5, 5, 5],
#9#                        [-2,  0, -1,   -2,  0, -5,     2,  0, -5,    5, 5, 5],
#9# ])
#9# ----------- to be test!
#9
#9# # 1) test above adjust the order of vertes
#9# # 2) test different ss, ds, ts, respectively
#9
#9# model_tde = np.array([ [-2,  0, -1,    2,  0, -1,     2,  0, -5,    5, 5, 5],
#9#                        [-2,  0, -1,    2,  0, -5,    -2,  0, -5,    5, 5, 5],
#9# ])
#9# ----------------------------------------------------------------------------------------------------
#9
#9# # ---------------  Y axis ----------------
#9# model = np.array([
#9#   [0, 0, -1, 4, 4,   0, 90,        5, 5, 5] 
#9# ])
#9
#9# obs = np.array([
#9#     [-1,  1, 0],
#9#     [ 1,  1, 0],
#9# ])
#9
#9
#9# model_tde = np.array([ [0, -2, -1,    0,  2, -1,    0,   2, -5,    5, 5, 5],
#9#                        [0, -2, -1,    0,  -2, -5,    0,  2, -5,    5, 5, 5],
#9# ])
#9
#9# model_tde = np.array([ [0, -2, -1,    0,  2, -1,    0,   2, -5,    5, 5, 5],
#9#                        [0, -2, -1,    0,  -2, -5,   0,  2, -5,     5, 5, 5],
#9# ])
#9
#9# model_tde = np.array([ [0, -2, -1,    0,   2, -5,   0,  2, -1,     5, 5, 5],
#9#                        [0, -2, -1,    0,  2, -5,    0,  -2, -5,    5, 5, 5],
#9# ])
#9# ----------------------------------------------------------------------------------------------------
#9# ---------------  vertical fault but ohter dierections ----------------
#9model = np.array([
#9  [0, 0, 0, 2.8284, 2.8284,   45, 90,        0, 5, 0] 
#9])
#9
#9obs = np.array([
#9    [-1,  1, 0],
#9    [ 1, -1, 0],
#9    [ 0.5, 0.5, -0.5],
#9])
#9
#9
#9# model_tde = np.array([ [-1, -1, 0,  1,  1,  0,        1,  1, -2.8284,    5, 5, 5],
#9#                        [-1, -1, 0,  1, 1, -2.8284,   -1, -1, -2.8284,    5, 5, 5],
#9# ])
#9
#9model_tde = np.array([ [-1, -1, 0,  1,  1,  0,        1,  1, -2.8284,    0, 5, 0],
#9                       [-1, -1, 0,  -1, -1, -2.8284,  1, 1, -2.8284,     0, 5, 0],
#9])
#9# ----------------------------------------------------------------------------------------------------
#9# # # ---------------  horizontal fault ----------------
#9# model = np.array([
#9#   # [0, 0, -1, 2., 2.,   90, 0,        10, 0, 0] 
#9#   [0, 1, -1, 2., 2.,   90, 0,        5, 5, 5] 
#9# ])
#9
#9# obs = np.array([
#9#     [ 0.1,  0.1,  -0.9],
#9#     [ 0.1,  0.1,  -1.1],
#9#     [ 0.5,  0.5,  -1.0],
#9#     # [ 0.5,  0.5,  -0.9],
#9#     # [ 0.5,  0.5,  -1.1],
#9# ])
#9
#9# # model_tde = np.array([ [  1,  -1, -1,   -1,  1, -1,   1,  1,  -1,   5, 5, 5],
#9# #                        [ -1,  -1, -1,   -1,  1, -1,   1, -1,  -1,   5, 5, 5],
#9# # ])
#9
#9# model_tde = np.array([ [  1,  -1, -1,   1,  1,  -1,   -1,  1, -1,   5, 5, 5],
#9#                        [ -1,  -1, -1,   1, -1,  -1,   -1,  1, -1,   5, 5, 5],
#9# ])
#9# ----------------------------------------------------------------------------------------------------
#9# # ---------------  Others ----------------
#9
#9# model = np.array([
#9# [0, 0, -1, 4, 4,  90, 90,        5, 5, 5]  # 
#9# ])
#9
#9# obs = np.array([
#9#     # [1, -1, 0],
#9#     # [1,  1, 0],
#9#     [-1/3, -1/3, -3],
#9#     [-1/3, -1/3, -14/3],
#9#     [-1/3, -1/3, -6],
#9#     [7,     -1,  -5],
#9#     [-7,   -1,   -5],
#9# ])
#9# model_tde = np.array([ [-1, -1, -5,    1, -1, -5,    -1,  1, -4,    1, -1, 2],
#9#                        [-1, -1, -5,   -1,  1,  0,     1, -1, -5,    1, -1, 2],
#9# ])
#9
#9# # model_tde = np.array([ [-1, -1, -5,    1, -1, -5,    -1,  1, -4,    5, 5, 5],
#9# #                        [-1, -1, -5,    1, -1, -5,    -1,  1,  0,    5, 5, 5],
#9# # ])
#9
#9# ----------------------------------------------------------------------------------------------------
#9
#9# ----------------------------------------------------------------------------------------------------
#9# length, width, depth, dip, strike, easting, northing, str-slip, dip-selip. opening
#9[U, S, E, flags] = dl.rde(obs, model, mu, nu)
#9[Utede, Stede, Etede] = dl.tde(obs, model_tde, mu, nu)
#9[Utede2, Stede2, Etede2] = dl.tde_meade(obs, model_tde, mu, nu)
#9
#9print("------------------- Result OKADA, mehdi, meade ------------------------------")
#9print(f"OKADA flags =\n {flags}")
#9print(f"OKADA U =\n {U}")
#9print(f"mehdi Utede =\n {Utede}")
#9print(f"meade Utede2 =\n {Utede2}")
#9
#9print("-------------------  stress  ------------------------------")
#9print(f"Okada S =\n {S}")
#9print(f"mehdi Stede =\n {Stede}")
#9print(f"meade Stede2 =\n {Stede2}")
#9
#9# print(f"E =\n {E}")
#9# print(f"Etede =\n {Etede}")
#9# print(f"Etede2 =\n {Etede2}")
