import matplotlib.pyplot as plt
import pyquaternion
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import quaternion

from PIL import Image
from set_up_model_for_inference import synsin_model


def inference(path_to_model, test_image, save_path = None, theta = -0.15, phi= -0.1, tx = 0, ty = 0, tz = 0.1):

    model_to_test = synsin_model(path_to_model)
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if isinstance(test_image, str):
        im = Image.open(test_image)
    else:
        im = test_image

    im = transform(im)

    RT = torch.eye(4).unsqueeze(0)

    RT[0, 0:3, 0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
    RT[0, 0:3, 3] = torch.Tensor([tx, ty, tz])

    batch = {
        'images': [im.unsqueeze(0)],
        'cameras': [{
            'K': torch.eye(4).unsqueeze(0),
            'kinv': torch.eye(4).unsqueeze(0)
        }]
    }

    with torch.no_grad():
        pred_imgs = model_to_test.model.module.forward_angle(batch, [RT])
        depth = nn.Sigmoid()(model_to_test.model.moudle.pts_regressor(batch['images'][0].cuda()))

    fig, axis = plt.subplots(1, 3, figsize=(10, 20))
    axis[0].axis('off')
    axis[1].axis('off')
    axis[2].axis('off')

    axis[0].imshow(im.permute(1, 2, 0) * 0.5 + 0.5)
    axis[0].set_title('Input Image')
    axis[1].imshow(pred_imgs[0].squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5)
    axis[1].set_title('Generated Image')
    axis[2].imshow(depth.squeeze().cpu().clamp(max=0.04))
    axis[2].set_title('Predicted Depth')

    if save_path:
        plt.savefig(save_path)
    else:
        return pred_imgs[0].squeeze().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5


if __name__ == '__main__':
    inference(path_to_model= './synsin/modelcheckpoints/realestate/zbufferpts.pth',
              test_image='apartment.JPG',
              save_path='output.png')