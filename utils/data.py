import torch as T
import torchvision
import torchvision.transforms as transforms


def get_mnist_iterator(bs, train=False, repeat=False):
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
        except StopIteration as e:
            if repeat:
                it = iter(dl)
            else:
                raise e
