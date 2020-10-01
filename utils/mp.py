import multiprocessing as mp
import threading

from collections import deque, defaultdict
import atexit
from tqdm.auto import tqdm, trange

import torch as T


def dataloader_top(dataloader):
    for batch in dataloader:
        return batch


def sanity_check_model(model, dataloader, optim_kw):
    top_batch = dataloader_top(dataloader)
    with T.no_grad():
        model.optim_step(top_batch)


def get_train_val(dataloader):
    if hasattr(dataloader, 'keys'):
        return dataloader['train'], dataloader['val']
    else:
        return dataloader, None


def try_get_its(its, dataloader):
    if its is None:
        try:
            return len(dataloader)
        except Exception:
            pass

    return its


def fit(model, dataloader, epochs=1, its=None, optim_kw={}, logger=None):
    train, val = get_train_val(dataloader)
    its = try_get_its(its, train)

    # Sanity check should be last!!!
    sanity_check_model(model, train, optim_kw)

    class FitCTX:
        def __enter__(self):
            self.model = model
            self.loss = 999
            self.info = {}
            self.should_terminate = False
            self.it = 0

            self.thread = threading.Thread(target=self._step)
            self.thread.start()
            self.history = defaultdict(lambda: defaultdict(lambda: []))
            atexit.register(self.terminate)

            return self

        @property
        def done(self):
            return not self.thread.is_alive()

        def _step(self):
            self.model = model

            for e in range(epochs):
                tr = tqdm(iter(train), total=its)

                epoch_str = f'[{(e + 1):03}/{epochs}]'
                metrics_sums = defaultdict(lambda: 0)

                for i, batch in enumerate(tr):
                    if self.should_terminate:
                        return

                    if i >= its:
                        break

                    loss, info = model.optim_step(batch, optim_kw)
                    for k, v in {'loss': loss, **info['metrics']}.items():
                        metrics_sums[k] += v

                    avg_metric = {
                        k: v / (i + 1)
                        for k, v in metrics_sums.items()
                    }
                    for k, v in avg_metric.items():
                        self.history['train_metrics'][k].append(v)

                    description = {
                        k: f'{v:0.5f}'
                        for k, v in avg_metric.items()
                    }
                    tr.set_description(
                        f'T {epoch_str} | {description}'
                    )

                    self.loss, self.info = loss, info
                    self.it = i

                    if logger is not None:
                        to_log = {f'train_{k}': v[-1]
                                  for k, v in self.history['train_metrics'].items()}
                        logger.log(to_log)

                if val is not None:
                    with T.no_grad():
                        tr = tqdm(val)
                        metrics_sums = defaultdict(lambda: 0)
                        for i, batch in enumerate(iter(tr)):
                            loss, info = model.optim_step(batch, optim_kw)
                            for k, v in {'loss': loss, **info['metrics']}.items():
                                metrics_sums[k] += v

                            avg_metric = {
                                k: v / (i + 1)
                                for k, v in metrics_sums.items()
                            }
                            for k, v in avg_metric.items():
                                self.history['val_metrics'][k].append(v)

                            description = {
                                k: f'{v:0.5f}'
                                for k, v in avg_metric.items()
                            }
                            tr.set_description(
                                f'V {epoch_str} | {description}'
                            )

                        if logger is not None:
                            to_log = {f'val_{k}': v[-1]
                                      for k, v in self.history['val_metrics'].items()}
                            logger.log(to_log)

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


def run_mp_generator(
    generator_ctor,
    buffer_size=32,
    num_processes=1,
):
    class MiltiprocessBuffer:
        def __init__(self):
            ctx = mp.get_context('fork')  # spawn
            self.buffer = ctx.Queue(maxsize=buffer_size)
            self.generator_ctor = generator_ctor
            self.lock = ctx.Lock()
            self.processes = [
                ctx.Process(target=self._run, args=(i, ))
                for i in range(num_processes)
            ]
            self.started = False

            atexit.register(self.terminate)

        def __next__(self):
            if not self.started:
                self.start()

            return self.pop()

        def __iter__(self):
            return self

        def __del__(self):
            self.terminate()

        def start(self):
            self.started = True
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
            generator = self.generator_ctor()
            for value in generator:
                while self.buffer.full():
                    pass
                with self.lock:
                    self.buffer.put(value)

    return MiltiprocessBuffer()


def sanity_check():
    import random
    import time

    def gen_init():
        val = random.randint(0, 100)
        while True:
            time.sleep(1)
            val += 1
            yield val

    mpb = run_mp_generator(
        buffer_size=1000,
        generator_ctor=gen_init,
        num_processes=16,
    )

    mpb.start()

    while True:
        values = mpb.get(5)
        print(values)


if __name__ == '__main__':
    sanity_check()

    # import nn as tu
    # import vis
    # import torch as T
    # import torchvision
    # from torchvision import transforms
    # import matplotlib.pyplot as plt

    # def get_mnist_iterator(bs):
    #     dataset = torchvision.datasets.MNIST(
    #         root='.data',
    #         train=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             torchvision.transforms.Normalize((0.1307,), (0.3081,))
    #         ]),
    #         download=True
    #     )
    #     dl = T.utils.data.DataLoader(
    #         dataset=dataset,
    #         batch_size=bs,
    #         # shuffle=True
    #     )
    #     it = iter(dl)

    #     while True:
    #         try:
    #             yield it.next()
    #         except StopIteration:
    #             it = iter(dl)

    # X, y = next(get_mnist_iterator(bs=128))
    # ae = tu.DenseAE().to('cuda')

    # with vis.fig(figsize=[10, 10]) as ctx, fit(
    #     model=ae,
    #     its=512,
    #     data_gen=([x, x] for x, y in get_mnist_iterator(bs=128)),
    #     optim_kw={'lr': 0.001}
    # ) as fit:
    #     while not fit.done:
    #         imgs = X[:64]
    #         pred_imgs = ae(imgs)

    #         # ctx.clear()
    #         T.cat([
    #             imgs.reshape(1, 1, 8, 8, 1, 28, 28),
    #             pred_imgs.reshape(1, 1, 8, 8, 1, 28, 28),
    #         ], dim=1).imshow()
