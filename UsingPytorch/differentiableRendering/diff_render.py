import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
                                RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
                                SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    print("WARN: using CPU only")

output_dir = './result_teapot'

verts, faces_idx, _ = load_obj("./data/teapot.obj")
faces = faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = TexturesVertex(verts_features=verts_rgb.to(device))

teapot_mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures
)

cameras = FoVPerspectiveCameras(device=device)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# set up silhouette renderer (differentiable)
silhouette_raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=100
)

silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=silhouette_raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

# set up phong render for visualitzaion (non differentiable)
phong_raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1
)
lights = PointLights(device=device, location=((2.0, 2.0, -2.0,),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=phong_raster_settings,
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

# our ground truth that we are trying to estimate
distance = 3
elevation = 50.0
azimuth = 0.0

R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

# render images
silhouette = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)

silhouette = silhouette.cpu().numpy()
image_ref = image_ref.cpu().numpy()

# plot images
plt.figure(figsize=(10, 10))
plt.imshow(silhouette.squeeze()[..., 3]) # only plot alpha channel
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_silhouette.png'))
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_rgb.png'))
plt.close()


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        # makes a binary mask that is 0.0 (black) where the mesh is and 1.0 (white) in empty space
        # F.E. pixel -> extract r,g,b -> get the max pixel val -> if it is 1 then paint black if it is not 1 then paint white
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        self.camera_position = nn.Parameter(torch.from_numpy(np.array([3.0, 6.9, 2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        R, T = look_at_view_transform(self.camera_position[0], self.camera_position[1], self.camera_position[2], device=device)
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image


model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# plot starting image
_, image_init = model()
plt.figure(figsize=(10, 10))
plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3]) # only plot alpha channel
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'starting_silhouette.png'))
plt.close()

# train
for i in range(0, 200):
    print(f'i = {i}')
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()

    if loss.item() < 500:
        break

    if i % 10 == 0:
        R, T = look_at_view_transform(model.camera_position[0], model.camera_position[1], model.camera_position[2], device=device)
        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)

        plt.figure()
        plt.imshow(image[..., :3])
        plt.title("iter: %d, loss %0.2f (dist=%0.2f, elev=%0.2f, azim=%0.2f)" % (i, loss.data, model.camera_position[0], model.camera_position[1], model.camera_position[2]))
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f'fitting_{i}.png'))
        plt.close()
