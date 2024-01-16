import open3d
import os
import torch

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures.meshes import join_meshes_as_batch

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mesh_names = ['cube.obj', 'diamond.obj', 'dodecahedron.obj']
data_path = "./data"
for mesh_name in mesh_names:
    mesh = open3d.io.read_triangle_mesh(os.path.join(data_path, mesh_name))
    open3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)

mesh_list = list()
for mesh_name in mesh_names:
    mesh = load_objs_as_meshes([os.path.join(data_path, mesh_name)], device=device)
    mesh_list.append(mesh)

mesh_batch = join_meshes_as_batch(mesh_list, include_textures=False)

# List Format
vertex_list = mesh_batch.verts_list()
print('vertex list = ', vertex_list)
face_list = mesh_batch.faces_list()
print('face list = ', face_list)

# Padded format
vertex_padded = mesh_batch.verts_padded()
print('verts padded = ', vertex_padded)
face_padded = mesh_batch.faces_padded()
print('faces padded = ', face_padded)

# Packed format
vertex_packed = mesh_batch.verts_packed()
print('verts packed =', vertex_packed)
face_packed = mesh_batch.faces_packed()
print('faces packed = ', face_packed)

num_vertices = vertex_packed.shape[0]
print('num vertices = ', num_vertices)

# do a ml experiment to learn the camera position
mesh_batch_noisy = mesh_batch.clone()

motion_gt = np.array([3, 4, 5])
motion_gt = torch.as_tensor(motion_gt)
print('motion ground truth = ', motion_gt)

motion_gt = motion_gt[None, :] # turns our 1 dimensional (3,) to a 2 dim (1,3)
motion_gt.to(device)

# make a noisy change of a noisy offset of all the verts by [3,4,5] (these are the truths) so average is [3,4,5]
noise = (0.1**0.5)*torch.randn(mesh_batch_noisy.verts_packed().shape).to(device)
noise = noise + motion_gt
mesh_batch_noisy = mesh_batch_noisy.offset_verts(noise).detach()

# starts at [0,0,0]
motion_estimate = torch.zeros(motion_gt.shape, device=device, requires_grad=True)
optimizer = torch.optim.SGD([motion_estimate], lr=0.1, momentum=0.9)

for i in range(0, 200):
    optimizer.zero_grad()
    current_mesh_batch = mesh_batch.offset_verts(motion_estimate.repeat(num_vertices, 1))

    # sample 5000 points from the ground truth mesh positions (sample_src) and the estimate displacement, loss is distance between them
    sample_trg = sample_points_from_meshes(current_mesh_batch, 5000)
    sample_src = sample_points_from_meshes(mesh_batch_noisy, 5000)
    loss, _ = chamfer_distance(sample_trg, sample_src)

    # learns to minimize the loss by picking an offset close to our ground truth [3,4,5]
    loss.backward()
    optimizer.step()
    print('i = ', i, ' motion_estimation = ', motion_estimate)