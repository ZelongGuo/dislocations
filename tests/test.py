import gmsh
import sys
import meshio
import numpy as np
import pyvista as pv
import dislocations as dl
import _okada92


#8 # 初始化 gmsh
#8 # gmsh.initialize(sys.argv)
#8 # gmsh.model.add("rectangle")
#8 
#8 # # 定义矩形的四个角点
#8 # p1 = gmsh.model.geo.addPoint(439.60017008978974, 3924.5362397775775, -5, meshSize=10)  # 设置不同网格尺寸
#8 # p2 = gmsh.model.geo.addPoint(429.7520925596677, 3926.272721554247, -5, meshSize=10)
#8 # p3 = gmsh.model.geo.addPoint(430.34600430580656, 3929.6409624425814, -14.396926207859083, meshSize=10)
#8 # p4 = gmsh.model.geo.addPoint(440.1940818359286, 3927.904480665912, -14.396926207859083, meshSize=10)
#8 
#8 
#8 # # 定义矩形的边
#8 # l1 = gmsh.model.geo.addLine(p1, p2)
#8 # l2 = gmsh.model.geo.addLine(p2, p3)
#8 # l3 = gmsh.model.geo.addLine(p3, p4)
#8 # l4 = gmsh.model.geo.addLine(p4, p1)
#8 
#8 # # 创建一个面
#8 # l_lines = [l1, l2, l3, l4]
#8 # loop = gmsh.model.geo.addCurveLoop(l_lines)
#8 # surface = gmsh.model.geo.addPlaneSurface([loop])
#8 
#8 # # 同步并生成网格（不使用 Transfinite 设置）
#8 # gmsh.model.geo.synchronize()
#8 # gmsh.model.mesh.generate(2)
#8 
#8 # # 保存网格文件
#8 # gmsh.write("rectangle_non_uniform.msh")
#8 
#8 # # 关闭 gmsh
#8 # gmsh.finalize()
#8 
#8 # ============ 读取网格文件并可视化 ============
#8 mesh = meshio.read("rectangle_non_uniform.msh")
#8 
#8 # 获取三角形单元
#8 triangles = np.array(mesh.cells_dict["triangle"])
#8 points = np.array(mesh.points)
#8 
#8 print(f"triangles: {triangles}")
#8 print(f"points: {points}")
#8 
#8 # # 可视化
#8 # plotter = pv.Plotter()
#8 # mesh_pv = pv.PolyData(points, np.hstack([np.full((triangles.shape[0], 1), 3), triangles]))
#8 # plotter.add_mesh(mesh_pv, show_edges=True, color="lightblue")
#8 
#8 # # 设置视图和坐标轴
#8 # plotter.view_isometric()
#8 # plotter.show_axes()
#8 # plotter.show()
#8 
#8 #---------------------------------------------------------------------------------------------------- 
#8 mu = 3.3e10
#8 nu = 0.25
#8 obs = np.array([
#8     [450, 3916, 0],
#8     [456, 4000, -1],
#8     [434.6761313247287, 3935.404480665912, -5]
#8     ])
#8 
#8 uc = np.array([434.6761313247287, 3925.404480665912, -5])
#8 model = np.array([[uc[0], uc[1], uc[2],   10, 10,   280, 70,    5, 5, 5]])
#8 print(f"model: {model}")
#8 
#8 [U, S, E, flags] = dl.rde(obs, model, mu, nu)
#8 
#8 
#8 
#8 # 遍历所有三角形，获取顶点坐标
#8 triangle_vertices = np.array([points[triangle].flatten() for triangle in triangles])  # 形状 (n, 9)
#8 
#8 ss = np.full((len(triangles), 1), 5)      # (n, 1)
#8 ds = np.full((len(triangles), 1), 5)      # (n, 1)
#8 ts = np.full((len(triangles), 1), 5)      # (n, 1)
#8 triangle_data = np.hstack([triangle_vertices, ss, ds, ts])
#8 triangle_data2 = np.hstack([triangle_vertices, ss, ds, ts])
#8 
#8 
#8 # 打印前几个三角形的数据
#8 print("每个三角形的顶点坐标 (x1, y1, z1, x2, y2, z2, x3, y3, z3) :")
#8 print(triangle_vertices[:5])  # 只打印前 5 行，避免输出过长
#8 
#8 [U2, S2, E2 ] = dl.tde(obs, triangle_data, mu, nu)
#8 [U3, S3, E3 ] = dl.tde_meade(obs, triangle_data2, mu, nu)
#8 
#8 #-------------------------------------------------------
#8 # cc = np.array([447.3987425976274, 3931.9897116567467, -25.606601717798213])
#8 # [U4, E4, S4 ] = _okada92.okada92(obs[0, :], obs[1, :], obs[2, :],  np.array(cc[0]), np.array(cc[1]), np.array(-cc[2]), np.array(10), np.array(10), np.array(45), np.array(50),np.array(0), np.array(5), np.array(0), mu, nu, np.array([0,0,0]),)
#8 
#8 #-------------------------------------------------------
#8 #3 [U2, S2, E2 ] = dl.tde(obs, model2, mu, nu)
#8 #3 [U3, S3, E3 ] = dl.tde_meade(obs, model2, mu, nu)
#8 U   = np.round(U, 12)
#8 U2  = np.round(U2, 12)
#8 U3  = np.round(U3, 12)
#8 print(f"U = \n{U}")
#8 print(f"U2 = \n{U2}")
#8 print(f"U3 = \n{U3}")
#8 
#8 S   = np.round(S, 12)
#8 S2  = np.round(S2, 12)
#8 S3  = np.round(S3, 12)
#8 print(f"S = \n{S}")
#8 print(f"S2 = \n{S2}")
#8 print(f"S3 = \n{S3}")
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------- 

mu = 3.3e10
nu = 0.25

# ----------------------------------------------------------------------------------------------------
# # ---------------  X axis ----------------
# model = np.array([
# [0, 0, -1, 4, 4,  90, 90,        5, 5, 5]  # -------- X axis
# ])

# obs = np.array([
#     [1, -1, 0],
#     [1,  1, 0],
# ])
# model_tde = np.array([ [-2,  0, -1,    2,  0, -1,     2,  0, -5,    5, 5, 5],
#                        [-2,  0, -1,   -2,  0, -5,     2,  0, -5,    5, 5, 5],
# ])
# ----------- to be test!

# # 1) test above adjust the order of vertes
# # 2) test different ss, ds, ts, respectively

# model_tde = np.array([ [-2,  0, -1,    2,  0, -1,     2,  0, -5,    5, 5, 5],
#                        [-2,  0, -1,    2,  0, -5,    -2,  0, -5,    5, 5, 5],
# ])
# ----------------------------------------------------------------------------------------------------

# # ---------------  Y axis ----------------
# model = np.array([
#   [0, 0, -1, 4, 4,   0, 90,        5, 5, 5] 
# ])

# obs = np.array([
#     [-1,  1, 0],
#     [ 1,  1, 0],
# ])


# model_tde = np.array([ [0, -2, -1,    0,  2, -1,    0,   2, -5,    5, 5, 5],
#                        [0, -2, -1,    0,  -2, -5,    0,  2, -5,    5, 5, 5],
# ])

# model_tde = np.array([ [0, -2, -1,    0,  2, -1,    0,   2, -5,    5, 5, 5],
#                        [0, -2, -1,    0,  -2, -5,   0,  2, -5,     5, 5, 5],
# ])

# model_tde = np.array([ [0, -2, -1,    0,   2, -5,   0,  2, -1,     5, 5, 5],
#                        [0, -2, -1,    0,  2, -5,    0,  -2, -5,    5, 5, 5],
# ])
# ----------------------------------------------------------------------------------------------------
# ---------------  vertical fault but ohter dierections ----------------
model = np.array([
  [0, 0, 0, 2.8284, 2.8284,   45, 90,        0, 5, 0] 
])

obs = np.array([
    [-1,  1, 0],
    [ 1, -1, 0],
    [ 0.5, 0.5, -0.5],
])


# model_tde = np.array([ [-1, -1, 0,  1,  1,  0,        1,  1, -2.8284,    5, 5, 5],
#                        [-1, -1, 0,  1, 1, -2.8284,   -1, -1, -2.8284,    5, 5, 5],
# ])

model_tde = np.array([ [-1, -1, 0,  1,  1,  0,        1,  1, -2.8284,    0, 5, 0],
                       [-1, -1, 0,  -1, -1, -2.8284,  1, 1, -2.8284,     0, 5, 0],
])
# ----------------------------------------------------------------------------------------------------
# # # ---------------  horizontal fault ----------------
# model = np.array([
#   # [0, 0, -1, 2., 2.,   90, 0,        10, 0, 0] 
#   [0, 1, -1, 2., 2.,   90, 0,        5, 5, 5] 
# ])

# obs = np.array([
#     [ 0.1,  0.1,  -0.9],
#     [ 0.1,  0.1,  -1.1],
#     [ 0.5,  0.5,  -1.0],
#     # [ 0.5,  0.5,  -0.9],
#     # [ 0.5,  0.5,  -1.1],
# ])

# # model_tde = np.array([ [  1,  -1, -1,   -1,  1, -1,   1,  1,  -1,   5, 5, 5],
# #                        [ -1,  -1, -1,   -1,  1, -1,   1, -1,  -1,   5, 5, 5],
# # ])

# model_tde = np.array([ [  1,  -1, -1,   1,  1,  -1,   -1,  1, -1,   5, 5, 5],
#                        [ -1,  -1, -1,   1, -1,  -1,   -1,  1, -1,   5, 5, 5],
# ])
# ----------------------------------------------------------------------------------------------------
# # ---------------  Others ----------------

# model = np.array([
# [0, 0, -1, 4, 4,  90, 90,        5, 5, 5]  # 
# ])

# obs = np.array([
#     # [1, -1, 0],
#     # [1,  1, 0],
#     [-1/3, -1/3, -3],
#     [-1/3, -1/3, -14/3],
#     [-1/3, -1/3, -6],
#     [7,     -1,  -5],
#     [-7,   -1,   -5],
# ])
# model_tde = np.array([ [-1, -1, -5,    1, -1, -5,    -1,  1, -4,    1, -1, 2],
#                        [-1, -1, -5,   -1,  1,  0,     1, -1, -5,    1, -1, 2],
# ])

# # model_tde = np.array([ [-1, -1, -5,    1, -1, -5,    -1,  1, -4,    5, 5, 5],
# #                        [-1, -1, -5,    1, -1, -5,    -1,  1,  0,    5, 5, 5],
# # ])

# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# length, width, depth, dip, strike, easting, northing, str-slip, dip-selip. opening
[U, S, E, flags] = dl.rde(obs, model, mu, nu)
[Utede, Stede, Etede] = dl.tde(obs, model_tde, mu, nu)
[Utede2, Stede2, Etede2] = dl.tde_meade(obs, model_tde, mu, nu)

print("------------------- Result OKADA, mehdi, meade ------------------------------")
print(f"OKADA flags =\n {flags}")
print(f"OKADA U =\n {U}")
print(f"mehdi Utede =\n {Utede}")
print(f"meade Utede2 =\n {Utede2}")

print("-------------------  stress  ------------------------------")
print(f"Okada S =\n {S}")
print(f"mehdi Stede =\n {Stede}")
print(f"meade Stede2 =\n {Stede2}")

# print(f"E =\n {E}")
# print(f"Etede =\n {Etede}")
# print(f"Etede2 =\n {Etede2}")
