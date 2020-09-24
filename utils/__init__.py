import utils.nn as nn
import utils.mp as mp
import utils.vis as vis

import torch as T
import numpy as np
from tqdm.auto import trange


def fit(model, data_gen, its, optim_kw={}):
    tr = trange(its)

    for _ in tr:
        batch = next(data_gen)
        loss, info = model.optim_step(batch, optim_kw)
        tr.set_description(f'Loss: {loss:0.6f}')

        yield loss, info


def load_img(url):
    from PIL import Image
    import requests
    from io import BytesIO

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    return T.tensor(np.array(img)).permute(2, 0, 1)
