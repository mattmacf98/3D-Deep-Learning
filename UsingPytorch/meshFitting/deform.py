import os
import sys
import torch
from pytorch3d.io import load_ply, save_ply
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARN: using CPU only")

# get the verts and faces of our raw point cloud data
(verts, faces) = load_ply("data/pedestrian.ply")
verts = verts.to(device)
faces = faces.to(device)

# calculate the mean and scale of the ply and move our verts so that they are centered at (0,0) and are between [-1,1]
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale
verts = verts[None, :, :]

# we will start our mesh as a spehere and set our deform verts param (which we will be optimizing to be all 0's (i.e. guessed mesh is a sphere)
src_mesh = ico_sphere(4, device) # if we start with more segments we can get finer grained meshes but may require more training
src_vert = src_mesh.verts_list()
deform_verts = torch.full(src_vert[0].shape, 0.0, device=device, requires_grad=True)

optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

# we need a few losses, chamfer makes the guessed mesh points close to the source ply,
# edge penalizes large length edges,
# normal makes sure close points have similar normal vectors
# laplacian tries to make the mesh smooth by having close points on same plane

w_chamfer = 1.0
w_edge = 1.0
w_normal = 0.01
w_laplacian = 0.1

# TRAINING LOOP
for i in range(0, 2000):
    print("i = ", i)
    optimizer.zero_grad()

    # offset our sphere with our predicted deformVerts and get the source verts and the sampled verts from our mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    sample_trg = verts
    sample_src = sample_points_from_meshes(new_src_mesh, verts.shape[1])

    # compute all the losses and weighted sum them up
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    loss_edge = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

    #backprop and step
    loss.backward()
    optimizer.step()

# get the final verts after 2000th iter and move them back to original position and scale which we extracted at the beginning
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
final_verts = final_verts * scale + center

# export the final ply file
final_obj = os.path.join("./", "deform.ply")
save_ply(final_obj, final_verts, final_faces, ascii=True)

