import utils.nn as tu
import utils as ut

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms

from pipelines.iae import MsgEncoder, MsgDecoder, InvertedAE


class Model(tu.Module):
    """
    Class for optimizing inputs with gradient ascent.
    Trying to generate inputs that generate images from some dataset (probably MNIST):
        - we have fixed num of samples per class and fixed num classes
        - the representations for the inputs that we are optimizing 
          are in the self.embedding instance
    """

    def __init__(self, msg_size, pretrained, requires_grad):
        super().__init__()

        self.num_samples_per_class = 10
        self.num_classes = 10
        self.num_embeddings = self.num_classes * self.num_samples_per_class

        self.embedding = nn.Embedding(
            self.num_embeddings,
            msg_size
        )

        if pretrained:
            self.ae = T.load(f'.models/glyph-ae-{msg_size}.h5_whole.h5')
        else:
            self.ae = InvertedAE(msg_size, img_channels=1)

        self.ae.set_requires_grad(requires_grad)

    def forward(self, idx):
        msg = self.embedding(idx)
        img = self.ae.encoder(msg)
        return img

    def generate_all(self):
        idxs = T.arange(0, self.num_embeddings).to(self.device)
        imgs = self.forward(idxs)
        return self.data_imgs.to(self.device), imgs

    def get_data(self, bs):
        # Hard coded to work with the MNIST dataset
        X_mnist, y_mnist = next(
            iter(ut.data.get_mnist_dl(bs=1024, train=True)))

        imgs = []
        for i in range(self.num_classes):
            x = X_mnist[y_mnist == i]
            imgs.append(x[:self.num_samples_per_class])

        imgs = T.cat(imgs)
        device = self.device
        self.data_imgs = imgs

        class Dataset(T.utils.data.Dataset):
            def __len__(self):
                return len(imgs) * 999_999

            def __getitem__(self, idx):
                idx = idx % 100

                y = imgs[idx].to(device)
                X = T.tensor(idx).to(device)
                return X, y

        return T.utils.data.DataLoader(
            dataset=Dataset(),
            batch_size=bs,
            shuffle=True,
        )


if __name__ == '__main__':
    import sys
    import time

    from utils.logger import WAndBLogger
    from datetime import datetime
    import matplotlib.pyplot as plt

    run_id = f'img_{datetime.now()}'

    model = Model(
        msg_size=32,
        pretrained=False,
        requires_grad=False,
    ).to('cuda')

    with ut.mp.fit(
        model=model,
        dataloader=model.get_data(bs=512),
        its=20_000,
        optim_kw={'lr': 0.001},
    ) as fit:
        for i in fit.wait:
            data_imgs, reconstr_imgs = model.generate_all()

            T.cat([
                data_imgs.view(1, 1, 10, 10, *data_imgs.shape[-3:]),
                reconstr_imgs.view(1, 1, 10, 10, *reconstr_imgs.shape[-3:])
            ], dim=1).imshow(cmap='gray', figsize=[10, 5])

            plt.savefig(
                f'.imgs/screen_{run_id}.png',
                transparent=True,
            )

            time.sleep(1)
