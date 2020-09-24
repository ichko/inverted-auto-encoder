from pipelines.glyph_ae import ReverseAE
import utils.nn as tu

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms


def get_mnist_iterator(bs, train=False):
    dataset = torchvision.datasets.MNIST(
        root='.data',
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]),
        download=True
    )
    dl = T.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        # shuffle=True
    )
    it = iter(dl)

    while True:
        try:
            yield it.next()
        except StopIteration:
            it = iter(dl)


class Classifier(tu.Module):
    def __init__(self, in_size):
        super().__init__()
        self.net = tu.dense(i=in_size, o=10, a=nn.Softmax(dim=1))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier

    X_train, y_train = next(get_mnist_iterator(bs=2 ** 15, train=True))
    X_test, y_test = next(get_mnist_iterator(bs=2 ** 12, train=False))

    msg_size = 48
    model = ReverseAE(msg_size, img_channels=1)
    model = model.to('cuda')
    model.make_persisted('.models/glyph-ae.h5')
    model.preload_weights()

    # 0.80877685546875
    # 0.7255859375

    # 0.972381591796875
    # 0.91845703125

    clf = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        hidden_layer_sizes=(100, 10),
        max_iter=500,
        random_state=1,
        shuffle=True,
        verbose=1,
        n_iter_no_change=9999999999,
    )

    clf.fit(model.decoder(X_train.to('cuda')).np, y_train.np)

    score_train = clf.score(model.decoder(X_train.to('cuda')).np, y_train.np)
    score_test = clf.score(model.decoder(X_test.to('cuda')).np, y_test.np)
    print(score_train)
    print(score_test)
