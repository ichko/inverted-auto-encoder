import utils.nn as nn
import utils.mp as mp
import utils.vis as vis
import utils.data as data
import utils.common as common

import torch as T
import cv2
import numpy as np
from tqdm.auto import trange


def fit(model, data_gen, its, optim_kw={}):
    tr = trange(its)

    for _ in tr:
        batch = next(data_gen)
        loss, info = model.optim_step(batch, optim_kw)
        tr.set_description(f'Loss: {loss:0.6f}')

        yield loss, info


def load_img(url, size=None):
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)

    if type(size) is int:
        H, W = img.shape[:2]
        H_new = H / W * size
        size = (size, int(H_new))

    if size is not None:
        print(size)
        img = cv2.resize(img, size)

    return T.tensor(img).permute(2, 0, 1)
