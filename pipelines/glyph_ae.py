import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import utils.nn as tu
import utils.vis as vis
import utils.mp as mp
import utils as ut

from tqdm.auto import trange

import kornia
import random


class MsgEncoder(tu.Module):
    def __init__(self, msg_size, img_channels):
        super().__init__()

        self.net = nn.Sequential(
            tu.Reshape(-1, msg_size, 1, 1),
            tu.deconv_block(msg_size, 128, ks=5, s=2, p=1),
            # nn.Dropout(0.2),
            tu.deconv_block(128, 64, ks=5, s=1, p=2),
            # nn.Dropout(0.3),
            tu.deconv_block(64, 64, ks=5, s=1, p=2),
            tu.deconv_block(64, 64, ks=5, s=2, p=1),
            # nn.Dropout(0.3),
            tu.deconv_block(64, 32, ks=5, s=1, p=2),
            tu.deconv_block(32, 32, ks=5, s=1, p=2),
            # nn.Dropout(0.01),
            tu.deconv_block(32, 16, ks=5, s=2, p=2),
            tu.deconv_block(16, img_channels, ks=4, s=2, p=0, a=nn.Sigmoid()),
        )

    def forward(self, z):
        return self.net(z)


class MsgDecoder(tu.Module):
    def __init__(self, in_channels, msg_size):
        super().__init__()

        self.net = tu.ConvToFlat(
            [in_channels, 128, 128, 64, 32, 32],
            msg_size,
            ks=3,
            s=2,
        )

    def forward(self, x):
        return self.net(x)


class ReverseAE(tu.Module):
    def __init__(self, msg_size, img_channels):
        super().__init__()
        self.msg_size = msg_size
        self.encoder = MsgEncoder(msg_size, img_channels)
        self.decoder = MsgDecoder(img_channels, msg_size)

        # This is done so that te decoder parameters are initialized
        imgs = self.encoder(self.sample(bs=1))
        self.decoder(imgs)

        self.noise = nn.Sequential(
            # kornia.augmentation.RandomHorizontalFlip(0.5),
            # kornia.augmentation.RandomVerticalFlip(0.5),
            kornia.augmentation.RandomAffine(
                degrees=10,
                translate=[0.1, 0.1],
                scale=[0.8, 1.2],
                shear=[-15, 15],
            ),
            kornia.augmentation.RandomPerspective(0.6, p=0.5),

        )

    def get_data_gen(self, bs):
        while True:
            X = self.sample(bs)
            yield X, X

    def sample(self, bs):
        return T.randn(bs, self.msg_size).to(self.device)

    def forward(self, bs):
        msg = self.sample(bs)
        img = self.encoder(msg)
        return img

    def configure_optim(self, lr, noise_size):
        self.noise_size = noise_size
        self.optim = T.optim.Adam(self.parameters(), lr)

    def optim_forward(self, X):
        def apply_noise(t):
            noise = T.randn_like(t).to(self.device)

            t = self.noise(t)
            t = t + noise

            return t

        img = self.encoder(X)
        img = apply_noise(img)
        pred_msg = self.decoder(img)

        return pred_msg


if __name__ == "__main__":
    from datetime import datetime

    msg_size = 128

    model = ReverseAE(msg_size, img_channels=1)
    model = model.to('cuda')
    model.make_persisted('.models/glyph-ae.h5')

    model.configure_optim(lr=0.001, noise_size=1)

    print(model.summary())

    X_mnist, y_mnist = next(iter(ut.data.get_mnist_dl(bs=1024, train=True)))
    X_mnist = X_mnist.to('cuda')
    y_mnist = y_mnist.to('cuda')

    X = []
    for i in range(10):
        x = X_mnist[y_mnist == i]
        X.append(x[:10])
    X = T.cat(X).to('cuda')

    msgs = model.sample(100)
    run_id = f'img_{datetime.now()}'
    imgs = model.encoder(msgs)

    with vis.fig([15, 5]) as ctx, mp.fit(
        model=model,
        its=1,
        dataloader=model.get_data_gen(bs=128),
    ) as fit:
        for i in fit.wait:
            model.persist()

            mnist_msg = model.decoder(X)
            mnist_recon = model.encoder(mnist_msg)

            imgs = model.encoder(msgs)

            T.cat([
                imgs.view(1, 1, 10, 10, *imgs.shape[-3:]),
                X.view(1, 1, 10, 10, *X_mnist.shape[-3:]),
                mnist_recon.view(1, 1, 10, 10, *mnist_recon.shape[-3:])
            ], dim=1).imshow()

            plt.savefig(f'.imgs/screen_{run_id}.png', bbox_inches='tight')

    model.save()
