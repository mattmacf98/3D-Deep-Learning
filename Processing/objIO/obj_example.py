import open3d
from pytorch3d.io import load_obj

mesh_file = "cube_texture.obj"
print("visualiing using open3d")
mesh = open3d.io.read_triangle_mesh(mesh_file, True)
open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

print("Loading using pytorch3d")
vertices, faces, aux = load_obj(mesh_file)

print('Vertex type, ', type(vertices))
print('Face type,', type(faces))
print('Aux type,', type(aux))
print('vertices - ', vertices)
print('faces - ', faces)
print('aux - ', aux)

texture_images = getattr(aux, 'texture_images')
print('texture images type, ', type(texture_images))
for key in texture_images:
    print(key)
print(texture_images['Skin'].shape)