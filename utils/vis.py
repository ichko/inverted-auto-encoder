import math
import numpy as np

import matplotlib.pyplot as plt
import torchvision
import torch as T

FIG = None


def show_vid(vid_path):
    from IPython.display import HTML
    from base64 import b64encode

    v = open(vid_path, 'rb').read()
    data_url = "data:video/webm;base64," + b64encode(v).decode()
    return HTML("""<video autoplay loop controls><source src="%s"></video>""" % data_url)


def vid(vid_path, size=None, fps=30.0):
    import cv2

    class VidCTX:
        # https://stackoverflow.com/questions/49530857/python-opencv-video-format-play-in-browser

        def __init__(self):
            self.video = None
            if size is not None:
                self.init_vid(*size)

        def init_vid(self, W, H):
            self.W, self.H = W, H
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            self.video = cv2.VideoWriter(vid_path, fourcc, fps, (W, H))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.video is not None:
                self.video.release()

        def show(self):
            return show_vid(vid_path)

        def append(self, frame):
            H, W = frame.shape[:2]
            if self.video is None:
                self.init_vid(W, H)
            frame = cv2.resize(frame, (self.W, self.H))
            if frame.max() <= 1:
                frame = frame * 255
            frame = frame.astype(np.uint8)

            self.video.write(frame)

    return VidCTX()


def fig(figsize):
    if type(figsize) is int:
        figsize = figsize, figsize

    class FigCTX:
        def __enter__(self):
            global FIG
            FIG = plt.figure(figsize=figsize)
            FIG.tight_layout()
            plt.ion()
            # plt.show()

            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            global FIG
            FIG = None
            plt.close()

        def clear(self):
            plt.cla()

    return FigCTX()


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_emb(pos, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    pad = 0.01
    s = 0.05

    ax.set_xlim(pos[:, 1].min() - pad, pos[:, 1].max() + s + pad)
    ax.set_ylim(pos[:, 0].min() - pad, pos[:, 0].max() + s + pad)

    ax.scatter(pos[:, 0], pos[:, 1], c=COLORS[labels])
    plt.show()
    plt.tight_layout()


def auto_grid(imgs):
    num_dims = len(imgs.shape)
    assert num_dims == 4, 'imgs dims has to be 4'

    bs = imgs.shape[0]
    channels = imgs.shape[1]

    rows = math.ceil(math.sqrt(bs))
    cols = math.ceil(bs / rows)
    new_bs = rows * cols
    H, W = imgs.shape[2], imgs.shape[3]

    imgs = np.pad(imgs, ((0, new_bs - bs), (0, 0), (0, 0), (0, 0)))
    imgs = imgs.reshape(rows, cols, channels, H, W)

    return imgs


def concat_grid(imgs):
    if len(imgs.shape) == 4:
        imgs = auto_grid(imgs)

    assert len(imgs.shape) == 5, 'imgs dims has to be 5'

    nrow = imgs.shape[0]
    channels = imgs.shape[2]
    imgs = imgs.reshape(imgs.shape[0] * imgs.shape[1], *imgs.shape[-3:])
    imgs = torchvision.utils.make_grid(
        T.tensor(imgs), nrow=nrow, padding=2, pad_value=0).np

    if channels == 1:
        imgs = imgs[:1, ]

    # rows, cols = imgs.shape[:2]
    # imgs = np.concatenate(np.split(imgs, rows, axis=0), axis=3)
    # imgs = np.concatenate(np.split(imgs, cols, axis=1), axis=4)
    # imgs = imgs[0, 0]

    imgs = np.transpose(imgs, (1, 2, 0))
    if imgs.shape[-1] == 1:
        imgs = imgs[:, :, 0]

    return imgs


def imshow(imgs, figsize=8, cmap='viridis', show=True):
    global FIG
    fig = FIG

    if type(figsize) is int:
        figsize = figsize, figsize

    num_dims = len(imgs.shape)
    if num_dims == 3:
        imgs = imgs[np.newaxis, ...]

    num_dims = len(imgs.shape)
    if num_dims == 4 or num_dims == 5:
        imgs = imgs[np.newaxis, np.newaxis, ...]

    if fig is None:
        fig = plt.figure(figsize=figsize)

    rows, cols = imgs.shape[:2]
    for r in range(rows):
        for c in range(cols):
            img = concat_grid(imgs[r, c])

            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img, cmap=cmap)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0.05, wspace=0.05)
    # plt.margins(0, 0)

    fig.canvas.draw()

    if show:
        fig.canvas.flush_events()
        # plt.show()
    else:
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array#comment72722681_35362787
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(int(height), int(width), 3)
        plt.close()

        return image


# Extensions
T.Tensor.imshow = lambda self, figsize=8, show=True, cmap='viridis': imshow(
    self.detach().cpu().numpy(),
    figsize=figsize,
    show=show,
    cmap=cmap,
)


if __name__ == '__main__':
    imgs = T.rand(5, 1, 6, 3)
    imgs.imshow()
    plt.show()
