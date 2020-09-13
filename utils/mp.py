import multiprocessing as mp
import threading

from collections import deque
import atexit
from tqdm.auto import trange


def fit(model, data_gen, its, optim_kw={}):
    class FitCTX:
        def __enter__(self):
            self.loss = 999
            self.info = {}
            self.should_terminate = False
            self.it = 0

            self.thread = threading.Thread(target=self._step)
            self.thread.start()
            atexit.register(self.terminate)

            return self

        @property
        def done(self):
            return not self.thread.is_alive()

        def _step(self):
            self.model = model

            tr = trange(its)
            for i in tr:
                if self.should_terminate:
                    return

                batch = next(data_gen)
                loss, info = model.optim_step(batch, optim_kw)
                tr.set_description(f'Loss: {loss:0.6f}')
                self.loss, self.info = loss, info
                self.it = i

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.terminate()

        def join(self):
            self.thread.join()

        def terminate(self):
            self.should_terminate = True

        @property
        def wait(self):
            i = 0
            while not self.done:
                yield i
                i += 1

    return FitCTX()


class MiltiprocessBuffer:
    def __init__(self, buffer_size, generator_func, num_processes):
        self.buffer = mp.Queue(maxsize=buffer_size)
        self.generator_func = generator_func
        self.lock = mp.Lock()
        self.processes = [
            mp.Process(target=self._run, args=(i, ))
            for i in range(num_processes)
        ]

        atexit.register(self.terminate)

    def start(self):
        for p in self.processes:
            p.start()

    def terminate(self):
        for p in self.processes:
            if p.is_alive:
                p.terminate()

    def try_pop(self):
        # with self.lock:
        if self.buffer.empty():
            return None
        return self.buffer.get()

    def pop(self):
        while True:
            if not self.buffer.empty():
                return self.buffer.get()

    def get(self, n):
        result = []
        while len(result) < n:
            if not self.buffer.empty():
                value = self.buffer.get()
                result.append(value)

        return result

    def _run(self, proc_id):
        generator = self.generator_func()
        while True:
            if not self.buffer.full():
                value = next(generator)
                with self.lock:
                    self.buffer.put(value)


def sanity_check():
    import random
    import time

    def gen_init():
        val = random.randint(0, 100)
        while True:
            time.sleep(1)
            val += 1
            yield val

    mpb = MiltiprocessBuffer(
        buffer_size=1000,
        generator_func=gen_init,
        num_processes=16,
    )

    mpb.start()

    while True:
        values = mpb.get(5)
        print(values)


if __name__ == '__main__':
    import nn as tu
    import vis
    import torch as T
    import torchvision
    from torchvision import transforms
    import matplotlib.pyplot as plt

    def get_mnist_iterator(bs):
        dataset = torchvision.datasets.MNIST(
            root='./data',
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

    X, y = next(get_mnist_iterator(bs=128))
    ae = tu.DenseAE().to('cuda')

    with vis.fig(figsize=[10, 10]) as ctx, fit(
        model=ae,
        its=512,
        data_gen=([x, x] for x, y in get_mnist_iterator(bs=128)),
        optim_kw={'lr': 0.001}
    ) as fit:
        while not fit.done:
            imgs = X[:64]
            pred_imgs = ae(imgs)

            # ctx.clear()
            T.cat([
                imgs.reshape(1, 1, 8, 8, 1, 28, 28),
                pred_imgs.reshape(1, 1, 8, 8, 1, 28, 28),
            ], dim=1).imshow()
