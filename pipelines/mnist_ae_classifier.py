from pipelines.glyph_ae import ReverseAE
import utils.nn as tu
import utils as ut

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms

from pipelines.glyph_ae import MsgEncoder, MsgDecoder


class Classifier(tu.Module):
    def __init__(self):
        super().__init__()
        # self.ae = T.load('.models/glyph-ae.h5_whole.h5')

        msg_size = 128
        self.ae = ReverseAE(msg_size, img_channels=1)

        self.ae.requires_grad = False

    def metrics(self, loss, info):
        y_pred = T.argmax(info['y_pred'], dim=1)
        acc = (y_pred == info['y']).float().sum() / len(y_pred)
        return {'acc': f'{acc.item():0.5f}'}

    def forward(self, x):
        x = self.ae.decoder(x)
        in_size = x.size(1)

        if not hasattr(self, 'net'):
            self.criterion = nn.CrossEntropyLoss()

            self.net = nn.Sequential(
                nn.Flatten(),
                tu.dense(i=in_size, o=100),
                tu.dense(i=100, o=10),
            ).to(x.device)

        return self.net(x)


if __name__ == '__main__':
    data = ut.common.pipe(
        ut.data.get_mnist_dl,
        ut.data.map_it(lambda batch: [t.to('cuda') for t in batch]),
    )

    with ut.mp.fit(
        model=Classifier().to('cuda'),
        dataloader=data(bs=128, train=True),
        epochs=10,
        its=60_000 // 128,
        optim_kw={'lr': 0.001}
    ) as fit:
        fit.join()
