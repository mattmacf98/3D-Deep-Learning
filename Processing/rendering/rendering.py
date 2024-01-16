import open3d
import os
import sys
import torch

import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras, PointLights, Materials, RasterizationSettings, MeshRenderer, MeshRasterizer)
from pytorch3d.renderer.mesh.shader import HardPhongShader

# Setup
sys.path.append(os.path.abspath(''))

DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cowMesh/cow.obj")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load mesh, camera and light
mesh = load_objs_as_meshes([obj_filename], device=device)

R, T = look_at_view_transform(2.7, 0, 180)
cameras = PerspectiveCameras(device=device, R=R, T=T)

lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# set up rasterizer, shader and renderer
raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1)
rasterizer = MeshRasterizer(raster_settings=raster_settings, cameras=cameras)
shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

# render


def display_with_name(name, images):
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.savefig(f'{name}.png')
    plt.show()


#light in front
images = renderer(mesh)
display_with_name("light_at_front", images)

# light behind
lights.location = torch.tensor([[0.0, 0.0, 1.0]], device=device)
images = renderer(mesh, lights=lights)
display_with_name("light_at_back", images)

# no ambient light
materials = Materials(device=device, specular_color=[[0.0, 1.0, 0.0]], shininess=10.0, ambient_color=[[0.01, 0.01, 0.01]])
images = renderer(mesh, lights=lights, materials=materials)
display_with_name("dark", images)

# green specular (and move camera and lights)
R, T = look_at_view_transform(dist=2.7, elev=10, azim=150)
cameras = PerspectiveCameras(device=device, R=R, T=T)

lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

materials = Materials(device=device, specular_color=[[0.0, 1.0, 0.0]], shininess=10.0)

images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
display_with_name("green_specular", images)

# red specular
materials = Materials(device=device, specular_color=[[1.0, 0.0, 0.0]], shininess=20.0)

images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
display_with_name("red_specular", images)

#No Specular
materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)

images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)
display_with_name("no_specular", images)

