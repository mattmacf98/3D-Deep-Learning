import open3d
from pytorch3d.io import load_ply

mesh_file = "parrallel_planes.ply"
print("Visualizing ith open3d")
mesh = open3d.io.read_triangle_mesh(mesh_file)

open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

print("Loading with Pytorch3D")
vertices, faces = load_ply(mesh_file)

print('Vertex type, ', type(vertices))
print('Face type,', type(faces))
print('vertices - ', vertices)
print('faces - ', faces)