import utils.nn as tu
import utils as ut

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms

from pipelines.iae import MsgEncoder, MsgDecoder, InvertedAE


class Classifier(tu.Module):
    def __init__(self, pretrained, requires_grad):
        super().__init__()
        if pretrained:
            self.ae = T.load('.models/glyph-ae.h5_whole.h5')
        else:
            msg_size = 64
            self.ae = InvertedAE(msg_size, img_channels=1)

        self.ae.requires_grad = requires_grad

    def metrics(self, loss, info):
        y_pred = T.argmax(info['y_pred'], dim=1)
        acc = (y_pred == info['y']).float().sum() / len(y_pred)
        return {'acc': acc.item()}

    def forward(self, x):
        x = self.ae.decoder(x)
        in_size = x.size(1)

        if not hasattr(self, 'net'):
            self.criterion = nn.CrossEntropyLoss()

            self.net = nn.Sequential(
                nn.Flatten(),
                # tu.dense(i=in_size, o=100),
                tu.dense(i=in_size, o=10),
            ).to(x.device)

        return self.net(x)


if __name__ == '__main__':
    from utils.logger import WAndBLogger
    import sys

    data = ut.common.pipe(
        ut.data.get_mnist_dl,
        ut.data.map_it(lambda batch: [t.to('cuda') for t in batch]),
    )

    if len(sys.argv) == 3:
        pretrained = sys.argv[1] == 'True'
        requires_grad = sys.argv[2] == 'True'
    else:
        pretrained = False
        requires_grad = False

    print(f'''
        =======================
        pretrained = {pretrained}
        requires_grad = {requires_grad}
        =======================
    ''')

    model = Classifier(pretrained=pretrained,
                       requires_grad=requires_grad).to('cuda')

    logger = WAndBLogger(
        project='zero_shot_structure',
        name=f'mnist_classifier_pretrained={pretrained}_grad={requires_grad}',
        model=model,
        hparams={},
        type='video',
    )

    with ut.mp.fit(
        model=model,
        dataloader={
            'train': data(bs=128, train=True),
            'val': data(bs=128, train=False)
        },
        epochs=10,
        optim_kw={'lr': 0.001},
        logger=logger,
    ) as fit:
        fit.model.summary(input_size=(1, 28, 28))
        fit.join()
