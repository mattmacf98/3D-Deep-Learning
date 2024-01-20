import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

from pytorch3d.io import load_objs_as_meshes

from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, RasterizationSettings,
                                MeshRenderer, MeshRasterizer, BlendParams, SoftSilhouetteShader, HardPhongShader,
                                PointLights, SoftPhongShader)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARN: using cpu only")

output_dir = "./result_cow"

obj_filename = "./data/cow.obj"
cow_mesh = load_objs_as_meshes([obj_filename], device=device)

cameras = FoVPerspectiveCameras(device=device)
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=100
)
render_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)

sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
    faces_per_pixel=50
)
render_textured = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_soft
    ),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1
)
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)

dist = 3
elevation = 50.0
azimuth = 0.0
R, T = look_at_view_transform(dist=dist, elev=elevation, azim=azimuth, device=device)

silhouette = render_silhouette(meshes_world=cow_mesh, R=R, T=T)
silhouette = silhouette.cpu().numpy()
image_ref = phong_renderer(meshes_world=cow_mesh, R=R, T=T)
image_ref = image_ref.cpu().numpy()

# plot images
plt.figure(figsize=(10, 10))
plt.imshow(silhouette.squeeze()[..., 3])  # only plot alpha channel
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_silhouette.png'))
plt.close()

plt.figure(figsize=(10, 10))
plt.imshow(image_ref.squeeze())
plt.grid(False)
plt.savefig(os.path.join(output_dir, 'target_rgb.png'))
plt.close()


class Model(nn.Module):
    def __init__(self, meshes, renderer_silhouette, renderer_textured, image_ref, weight_silhouette, weight_texture):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer_silhouette = renderer_silhouette
        self.renderer_textured = renderer_textured
        self.weight_silhouette = weight_silhouette
        self.weight_texture = weight_texture

        image_ref_silhouette = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref_silhouette', image_ref_silhouette)

        image_ref_textured = torch.from_numpy((image_ref[..., :3]).astype(np.float32))
        self.register_buffer('image_ref_textured', image_ref_textured)

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([3.0, 6.9, 2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]  # (1, 3)

        image_silhouette = self.renderer_silhouette(meshes_world=self.meshes.clone(), R=R, T=T)
        image_textured = self.renderer_textured(meshes_world=self.meshes.clone(), R=R, T=T)

        loss_silhouette = torch.sum((image_silhouette[..., 3] - self.image_ref_silhouette) ** 2)
        loss_texture = torch.sum((image_textured[..., :3] - self.image_ref_textured) ** 2)

        loss = self.weight_silhouette * loss_silhouette + self.weight_texture * loss_texture
        return loss, image_silhouette, image_textured


# apparently having any weights on the texture makes the model shit to train
model = Model(meshes=cow_mesh, renderer_silhouette=render_silhouette, renderer_textured=render_textured,
              image_ref=image_ref, weight_silhouette=1.0, weight_texture=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


def print_current_rgb(name, model, i):
    R = look_at_rotation(model.camera_position[None, :], device=model.device)
    T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]  # (1, 3)
    image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
    image = image[0, ..., :3].detach().squeeze().cpu().numpy()
    image = img_as_ubyte(image)

    plt.figure()
    plt.imshow(image[..., :3])
    plt.title("iter: %d, loss %0.2f (x=%0.2f, y=%0.2f, z=%0.2f)" % (
        i, loss.data, model.camera_position[0], model.camera_position[1], model.camera_position[2]))
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, name))
    plt.close()


# train
for i in range(0, 200):
    optimizer.zero_grad()
    loss, _, _ = model()
    loss.backward()
    optimizer.step()
    print(f'i = {i} , Loss: {loss.item()}')

    if loss.item() < 500:
        print_current_rgb(f'fitting_{i}.png', model, i)
        break

    if i % 10 == 0:
        print_current_rgb(f'fitting_{i}.png', model, i)
