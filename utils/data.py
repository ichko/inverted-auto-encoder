import torch as T
import torchvision
import torchvision.transforms as transforms

from utils.common import partial
from functools import wraps


@partial
def map_it(mapper, dl):
    class Wrapper:
        def __getattr__(self, name):
            return getattr(dl, name)

        def __len__(self):
            return len(dl)

        def __iter__(self):
            return map(mapper, dl)

    return wraps(dl)(Wrapper())


def get_mnist_dl(bs, train=False):
    dataset = torchvision.datasets.MNIST(
        root='.data',
        train=train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]),
        download=True
    )
    dl = T.utils.data.DataLoader(
        dataset=dataset,
        batch_size=bs,
        shuffle=True,
    )

    return dl
