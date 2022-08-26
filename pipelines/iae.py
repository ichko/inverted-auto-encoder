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


class ConvExpand(tu.Module):
    def __init__(self, msg_size, img_channels):
        super().__init__()
        self.net = nn.Sequential(
            tu.FlatToConv(
                [msg_size, 128, 64, 64, 32, img_channels],
                ks=5,
                s=2,
                a=[nn.LeakyReLU(0.3)] * 4 + [nn.Sigmoid()],
                p=1,
            ),
        )

    def forward(self, z):
        return self.net(z)


class ConvShrink(tu.Module):
    def __init__(self, in_channels, msg_size):
        super().__init__()
        self.net = tu.ConvToFlat(
            [in_channels, 128, 64, 64, 32, 32],
            msg_size,
            ks=5,
            s=2,
        )

    def forward(self, x):
        return self.net(x)


class Augmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.structural_augmentation = nn.Sequential(
            # kornia.augmentation.RandomHorizontalFlip(0.5),
            # kornia.augmentation.RandomVerticalFlip(0.5),
            # kornia.augmentation.ColorJitter(0.0, 0.5, 0.5, 0),
            kornia.augmentation.RandomAffine(
                degrees=10,
                translate=[0.0, 0.0],
                scale=[0.9, 1.1],
                shear=[-2, 2],
            ),
            kornia.augmentation.RandomPerspective(distortion_scale=0.3, p=0.9),
        )

    def forward(self, x):
        x = self.structural_augmentation(x)
        noise = T.randn_like(x)[:, :] + 1
        x = x + noise

        return x


class InvertedAE(tu.Module):
    def __init__(self, msg_size, img_channels):
        super().__init__()
        self.msg_size = msg_size
        self.expand = ConvExpand(msg_size, img_channels)
        self.augment = Augmentation()
        self.shrink = ConvShrink(img_channels, msg_size)

        # This is done so that te decoder parameters are initialized
        imgs = self.expand(self.sample(bs=1))
        self.shrink(imgs)

    def get_data_gen(self, bs):
        while True:
            X = self.sample(bs)
            yield X, X

    def sample(self, bs):
        return T.randn(bs, self.msg_size).to(self.device)

    def forward(self, bs):
        msg = self.sample(bs)
        img = self.expand(msg)
        return img

    def optim_forward(self, x):
        x = self.expand(x)
        x = self.augment(x)
        x = self.shrink(x)
        return x


def get_model(preload=False):
    import sys

    try:
        msg_size = int(sys.argv[1])
    except:
        msg_size = 200

    print(f'MSG_SIZE = {msg_size}')

    # if preload:
    #     whole_model_name = f'../.models/glyph-ae-{msg_size}.h5_whole.h5'
    #     return T.load(whole_model_name)

    model = InvertedAE(msg_size, img_channels=3)
    model = model.to('cuda')

    model.make_persisted(f'models/glyph-ae-{msg_size}.h5')

    if preload:
        model.preload_weights()

    model.summary()

    return model


if __name__ == "__main__":
    from datetime import datetime

    model = get_model(preload=False)

    X_mnist, y_mnist = next(iter(ut.data.get_mnist_dl(bs=1024, train=True)))
    X_mnist = X_mnist.to('cuda')
    y_mnist = y_mnist.to('cuda')

    X = []
    for i in range(10):
        x = X_mnist[y_mnist == i]
        X.append(x[:10])
    X = T.cat(X).to('cuda')

    msgs = model.sample(32)
    run_id = f'img_{datetime.now()}'
    imgs = model.expand(msgs)
    print(f'IMG_SHAPE: {imgs.shape}')

    with vis.fig([8, 4]) as ctx, mp.fit(
        model=model,
        its=512 * 25,
        dataloader=model.get_data_gen(bs=64),
        optim_kw={'lr': 0.02}
    ) as fit:
        for i in fit.wait:
            model.persist()

            # mnist_msg = model.decoder(X)
            # mnist_recon = model.encoder(mnist_msg)

            imgs = model.expand(msgs)

            T.cat([
                imgs.view(1, 1, 8, 4, *imgs.shape[-3:]),
                # X.view(1, 1, 10, 10, *X_mnist.shape[-3:]),
                # mnist_recon.view(1, 1, 10, 10, *mnist_recon.shape[-3:])
            ], dim=1).imshow(cmap='gray')

            plt.savefig(
                f'imgs/screen_{run_id}.png',
                bbox_inches='tight',
                pad_inches=0.0,
                transparent=True,
            )

    model.save()
