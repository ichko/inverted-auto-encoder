from pipelines.glyph_ae import ReverseAE
import utils.nn as tu

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms


def get_mnist_iterator(bs):
    dataset = torchvision.datasets.MNIST(
        root='.data',
        train=True,
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


X, y = next(get_mnist_iterator(bs=2 ** 12))
print(X.shape)

X_flat = X.reshape(-1, 28 * 28)


class Classifier(tu.Module):
    def __init__(self, in_size):
        super().__init__()
        self.net = tu.dense(i=in_size, o=10, a=nn.Softmax(dim=1))

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier

    msg_size = 32
    model = ReverseAE(msg_size, img_channels=1)
    model = model.to('cuda')
    model.make_persisted('.models/glyph-ae.h5')
    model.preload_weights()

    X_repr = model.decoder(X.to('cuda'))

    X_repr = X_repr.np
    # X_repr = X_flat.np

    y = y.np

    clf = MLPClassifier(
        solver='adam',
        alpha=1e-5,
        hidden_layer_sizes=(100, 10),
        max_iter=5000,
        random_state=1,
        shuffle=True,
        verbose=1,
        n_iter_no_change=9999999999,
    )

    clf.fit(X_repr, y)

    print(clf.score(X_repr, y))

    print(X_repr.shape)
