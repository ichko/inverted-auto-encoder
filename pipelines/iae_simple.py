import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision

import kornia

from datetime import datetime
from tqdm.auto import trange
import numpy as np
import os


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        bs = x.size(0)
        return x.reshape(bs, *self.shape)


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.structural_augmentation = nn.Sequential(
            kornia.augmentation.RandomAffine(
                degrees=20,
                translate=[0.1, 0.1],
                scale=[0.9, 1.1],
                shear=[-2, 2],
            ),
            kornia.augmentation.RandomPerspective(distortion_scale=0.5, p=1),
        )

    def forward(self, x):
        x = self.structural_augmentation(x)
        noise = torch.randn_like(x)[:, :] + 1
        x = x + noise

        return x


class DenseIAE(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, msg_size, img_shape):
        super().__init__()
        self.msg_size = msg_size
        img_size = np.prod(img_shape)
        self.expand = nn.Sequential(
            nn.Linear(msg_size, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, img_size),
            nn.Sigmoid(),
            Reshape(*img_shape),
        )
        self.squeeze = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(64, msg_size),
        )

        self.noise = Noise()

    def sample(self, bs):
        return torch.randn(bs, self.msg_size).to(self.device)

    def forward(self, msg):
        pred_img = self.expand(msg)
        noise_img = self.noise(pred_img)
        pred_msg = self.squeeze(noise_img)
        return pred_msg, pred_img

    def optim_step(self, bs, lr):
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        msg = self.sample(bs)
        pred_msg, pred_img = self.forward(msg)
        loss = F.mse_loss(pred_msg, msg)

        optim.zero_grad()
        loss.backward()
        optim.step()

        return loss, pred_img


if __name__ == "__main__":
    device = 'cuda'
    model = DenseIAE(256, img_shape=[3, 32, 32])
    model = model.to(device)

    its = 10000
    pbar = trange(its)
    history = []

    run_id = f'run_{datetime.now()}'
    os.makedirs(f'new_img/{run_id}', exist_ok=True)

    msg = model.sample(bs=16)
    for i in pbar:
        loss, pred_img = model.optim_step(bs=32, lr=0.001)
        loss = loss.detach().cpu().numpy()
        pbar.set_description(f"loss: {loss:.5f}")

        if i % 100 == 0:
            plt.scatter(i, loss, c="b")
            plt.pause(0.0001)

            pred_img = model.expand(msg)
            pred_img = torchvision.utils.make_grid(pred_img, nrows=4)
            pred_img = pred_img.permute(1, 2, 0).detach().cpu().numpy()

            plt.close()
            plt.imshow(pred_img)
            plt.savefig(
                f"new_img/{run_id}/screen_{i:05}.png",
                bbox_inches="tight",
                pad_inches=0.0,
                transparent=True,
            )

    print("making a gif...")
    os.system(f"magick convert -delay 20 -loop 0 new_img/\"{run_id}\"/* new_img/\"{run_id}\".gif")
    print("done!")
