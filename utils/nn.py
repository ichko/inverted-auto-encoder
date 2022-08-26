import os

import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Custom Module"
        self.criterion = nn.MSELoss()

    def count_parameters(self):
        return count_parameters(self)

    def make_persisted(self, path):
        self.path = path

    def persist(self):
        T.save(self.state_dict(), self.path)

    def preload_weights(self):
        self.load_state_dict(T.load(self.path))

    def save(self, path=None):
        path = path if self.path is None else path
        T.save(self, f"{self.path}_whole.h5")

    def can_be_preloaded(self):
        return os.path.isfile(self.path)

    def configure_optim(self, lr):
        self.optim = T.optim.Adam(self.parameters(), lr=lr)

    def metrics(self, _loss, _info):
        return {}

    def optim_step(self, batch, optim_kw={}):
        X, y = batch

        y_pred = self.optim_forward(X)
        loss = self.criterion(y_pred, y)

        if loss.requires_grad:
            if not hasattr(self, "optim"):
                self.configure_optim(**optim_kw)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        metrics = self.metrics(
            loss.item(),
            {
                "X": X,
                "y_pred": y_pred,
                "y": y,
            },
        )

        return loss.item(), {
            "metrics": metrics,
            "X": X,
            "y_pred": y_pred,
            "y": y,
        }

    def set_requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def summary(self, input_size=-1):
        try:
            from torchsummary import summary

            summary(self, input_size)
            return
        except Exception:
            pass

        result = f" > {self.name[:38]:<38} | {count_parameters(self):09,}\n"
        for name, module in self.named_children():
            type = module._get_name()
            num_prams = count_parameters(module)
            result += f" >  {name[:20]:>20}: {type[:15]:<15} | {num_prams:9,}\n"

        print(result)

    def optim_forward(self, X):
        return self.forward(X)

    @property
    def device(self):
        return next(self.parameters()).device


class DenseAE(Module):
    def __init__(self, hid_size=2):
        super().__init__()
        self.hid_size = hid_size

    def forward(self, x):
        if not hasattr(self, "encoder"):
            dims = np.prod(list(x.shape[1:]))
            self.encoder = nn.Sequential(
                nn.Flatten(),
                dense(dims, 128),
                dense(128, 16),
                dense(16, self.hid_size, a=None),
            )

            self.decoder = nn.Sequential(
                dense(self.hid_size, 16),
                dense(16, 128),
                dense(128, np.prod(x.shape[1:]), a=nn.Sigmoid()),
                Reshape(-1, *x.shape[1:]),
            )

            self.criterion = nn.BCELoss(reduction="mean")
            self.to(x.device)

        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_activation():
    LEAKY_SLOPE = 0.2
    return nn.LeakyReLU(LEAKY_SLOPE, inplace=True)


def one_hot(t, one_hot_size=None):
    one_hot_size = t.max() + 1 if one_hot_size is None else one_hot_size

    hot = T.zeros((t.size(0), one_hot_size))
    hot[T.arange(t.size(0)), t] = 1
    return hot


def cat_channels():
    """
    Concatenate number of channels in a single tensor
    Converts tensor with shape:
        (bs, num_channels, channel_size, h, w)
    to tensor with shape:
        (bs, num_channels * channel_size, h, w)
    """

    class CatChannels(nn.Module):
        def forward(self, t):
            shape = t.size()
            cat_dim_size = shape[1] * shape[2]
            return t.view(-1, cat_dim_size, *shape[3:])

    return CatChannels()


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def batch_conv(x, w, p=0, s=1):
    # SRC - https://discuss.pyT.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2

    batch_size = x.size(0)
    output_size = w.size(1)

    o = F.conv2d(
        x.reshape(1, batch_size * x.size(1), x.size(2), x.size(3)),
        w.reshape(batch_size * w.size(1), w.size(2), w.size(3), w.size(4)),
        groups=batch_size,
        padding=p,
        stride=s,
        dilation=1,
    )
    o = o.reshape(batch_size, output_size, o.size(2), o.size(3))

    return o


def dense(i, o, a=get_activation()):
    l = nn.Linear(i, o)
    return l if a is None else nn.Sequential(l, a)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)


class Lambda(nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.forward = forward

    def forward(self, *args):
        return self.forward(*args)


def resize(t, size):
    return F.interpolate(t, size, mode="bicubic", align_corners=True)


def conv_block(i, o, ks, s, p, a=get_activation(), d=1, bn=True):
    block = [nn.Conv2d(i, o, kernel_size=ks, stride=s, padding=p, dilation=d)]
    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    return nn.Sequential(*block)


def deconv_block(i, o, ks, s, p, a=get_activation(), d=1, bn=True):
    block = [
        nn.ConvTranspose2d(
            i,
            o,
            kernel_size=ks,
            stride=s,
            padding=p,
            dilation=d,
        )
    ]

    if bn:
        block.append(nn.BatchNorm2d(o))
    if a is not None:
        block.append(a)

    if len(block) == 1:
        return block[0]

    return nn.Sequential(*block)


def stack_conv_blocks(block_ctor, sizes, ks, a, s, p):
    layers = [
        block_ctor(
            i=sizes[l],
            o=sizes[l + 1],
            ks=ks,
            s=s,
            p=p,
            a=(None if l == len(sizes) - 2 else a),
            d=1,
            # batch norm everywhere except the last layer
            bn=(l != len(sizes) - 2),
        )
        for l in range(len(sizes) - 1)
    ]

    return nn.Sequential(*layers)


def conv_encoder(sizes, ks=4, a=get_activation()):
    return stack_conv_blocks(conv_block, sizes, ks, a, s=2, p=ks // 2 - 1)


def conv_transform(sizes, ks=5, s=1, a=get_activation()):
    return stack_conv_blocks(conv_block, sizes, ks, a, s, p=ks // 2)


def conv_decoder(sizes, ks=4, s=2, a=get_activation()):
    return stack_conv_blocks(deconv_block, sizes, ks, a, s=s, p=ks // 2 - 1)


class FlatToConv(nn.Module):
    def __init__(self, channel_size, ks, s, a, p):
        super().__init__()

        def process_arg(*args):
            return [
                a if type(a) is list else [a] * (len(channel_size) - 1) for a in args
            ]

        ks, s, a, p = process_arg(ks, s, a, p)

        layers = []
        layers_io = list(zip(channel_size, channel_size[1:]))
        for idx, ((i, o), ks, s, a, p) in enumerate(zip(layers_io, ks, s, a, p)):
            bn = idx < len(layers_io) - 1
            l = deconv_block(i=i, o=o, ks=ks, s=s, p=p, a=a, bn=bn)
            layers.append(l)

        self.net = nn.Sequential(Reshape(-1, channel_size[0], 1, 1), *layers)

    def forward(self, x):
        return self.net(x)


class ConvToFlat(nn.Module):
    def __init__(
        self,
        channel_sizes,
        out_size,
        ks=4,
        s=2,
        a=get_activation(),
    ):
        super().__init__()
        self.channel_sizes = channel_sizes
        self.out_size = out_size
        self.ks = ks
        self.s = s
        self.a = a

    def forward(self, x):
        if not hasattr(self, "net"):
            input_size = x.shape[-2:]
            input_channels = self.channel_sizes[0]
            self.encoder = conv_transform(self.channel_sizes, self.ks, self.s, self.a)

            self.encoder_out_shape = compute_output_shape(
                self.encoder,
                (input_channels, *input_size),
            )

            self.flat_encoder_out_size = np.prod(self.encoder_out_shape[-3:])

            self.encoded_to_flat = nn.Sequential(
                nn.Flatten(),
                self.a,
                nn.Linear(self.flat_encoder_out_size, self.out_size),
            )

            self.net = nn.Sequential(
                self.encoder,
                self.encoded_to_flat,
            )

            self.net.to(x.device)

        return self.net(x)


def compute_output_shape(net, frame_shape):
    with T.no_grad():
        t = T.rand(1, *frame_shape)
        out = net(t)

        return out.shape


def extract_tensors(vec, tensor_shapes):
    slice_indices = [0] + [np.prod(t) for t in tensor_shapes]
    slice_indices = np.cumsum(slice_indices)

    tensors = []
    for i in range(len(slice_indices) - 1):
        t = vec[:, slice_indices[i] : slice_indices[i + 1]]
        t = t.reshape(-1, *tensor_shapes[i])
        tensors.append(t)

    return tensors


def spatial_transformer(i, num_channels, only_translations=False):
    class SpatialTransformer(nn.Module):
        def __init__(self):
            super().__init__()

            self.num_channels = num_channels
            self.locator = nn.Sequential(
                nn.Linear(i, num_channels * 2 * 3),
                Reshape(-1, 2, 3),
            )

            self.device = self.locator[0].bias.device
            # Taken from the pytorch spatial transformer tutorial.
            self.locator[0].weight.data.zero_()
            self.locator[0].bias.data.copy_(
                T.tensor(
                    [1, 0, 0, 0, 1, 0] * num_channels,
                    dtype=T.float,
                ).to(self.device)
            )

        def forward(self, x):
            inp, tensor_3d = x

            theta = self.locator(inp)
            (
                _,
                C,
                H,
                W,
            ) = tensor_3d.shape

            if only_translations:
                theta[:, :, :-1] = (
                    T.tensor(
                        [[1, 0], [0, 1]],
                        dtype=T.float,
                    )
                    .to(self.device)
                    .unsqueeze_(0)
                )

            grid = F.affine_grid(
                theta,
                (theta.size(dim=0), 1, H, W),
                align_corners=True,
            )

            # Last values
            self.grid = grid
            self.theta = theta

            tensor_3d = tensor_3d.reshape(-1, 1, H, W)
            tensor_3d = F.grid_sample(
                tensor_3d,
                grid,
                align_corners=True,
            )

            return tensor_3d.reshape(-1, C, H, W)

    return SpatialTransformer()


@T.jit.script
def mask_sequence(tensor, mask):
    initial_shape = tensor.shape
    bs, seq = mask.shape
    masked = T.where(
        mask.reshape(bs * seq, -1),
        tensor.reshape(bs * seq, -1),
        T.tensor(0, dtype=T.float32).to(tensor.device),
    )

    return masked.reshape(initial_shape)


def prepare_rnn_state(state, num_rnn_layers):
    """
    RNN cells expect the initial state
    in the shape -> [rnn_num_layers, bs, rnn_state_size]
    In this case rnn_state_size = state // rnn_num_layers.
    The state is distributed among the layers
    state          -> [bs, state_size]
    rnn_num_layers -> int
    """
    return T.stack(
        state.chunk(T.tensor(num_rnn_layers), dim=1),
        dim=0,
    )


def time_distribute(module, input=None):
    """
    Distribute execution of module over batched sequential input tensor.
    This is done in the batch dimension to facilitate parallel execution.
    input  -> [bs, seq, *x*]
    module -> something that takes *x*
    return -> [bs, seq, module(x)]
    """
    if input is None:
        return time_distribute_decorator(module)

    shape = input[0].size() if type(input) is list else input.size()
    bs = shape[0]
    seq_len = shape[1]

    if type(input) is list:
        input = [i.reshape(-1, *i.shape[2:]) for i in input]
    else:
        input = input.reshape(-1, *shape[2:])

    out = module(input)
    out = out.view(bs, seq_len, *out.shape[1:])

    return out


def time_distribute_decorator(module):
    class TimeDistributed(nn.Module):
        def __init__(self):
            # IMPORTANT:
            #   This init has to be here so that the
            #   passed module parameters can be part of the
            #   state (parameters) of the overall model.

            super().__init__()
            self.module = module

        def forward(self, input):
            shape = input[0].size() if type(input) is list else input.size()
            bs = shape[0]
            seq_len = shape[1]

            if type(input) is list:
                input = [i.reshape(-1, *i.shape[2:]) for i in input]
            else:
                input = input.reshape(-1, *shape[2:])

            out = module(input)
            out = out.view(bs, seq_len, *out.shape[1:])

            return out

    return TimeDistributed()


def time_distribute_13D(module):
    class TimeDistributed(nn.Module):
        def forward(self, input):
            bs, seq_len, s = [input.size(i) for i in range(3)]
            input = input.reshape(-1, s)
            out = module(input)
            return out.reshape(
                bs,
                seq_len,
                out.size(1),
                out.size(2),
                out.size(3),
            )

    return T.jit.script(TimeDistributed())


def time_distribute_31D(module):
    class TimeDistributed(nn.Module):
        def forward(self, input):
            bs, seq_len, c, h, w = [input.size(i) for i in range(5)]
            input = input.reshape(-1, c, h, w)
            out = module(input)
            return out.reshape(bs, seq_len, out.size(1))

    return T.jit.script(TimeDistributed())


class KernelEmbedding(nn.Module):
    # kernel_sizes - (in, out, ks)[]
    def __init__(self, num_embeddings, ks, channels, a=get_activation()):
        super().__init__()

        self.activation = a
        self.kernel_shapes = [[o, i, ks, ks] for i, o in zip(channels, channels[1:])]

        # self.batch_norms = nn.Sequential(
        #     *[nn.BatchNorm2d(k[0]) for k in self.kernel_shapes])

        self.kernels_flat = [np.prod(k) for k in self.kernel_shapes]

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=np.sum(self.kernels_flat),
        )

    def forward(self, x):
        tensor, kernel_indexes = x

        emb = self.embedding(kernel_indexes)
        kernels = extract_tensors(emb, self.kernel_shapes)

        transformed_tensor = tensor
        for i, k in enumerate(kernels):
            ks = self.kernel_shapes[i]
            transformed_tensor = batch_conv(
                transformed_tensor,
                k,
                p=ks[-1] // 2,
                s=1,
            )
            # TODO Should we batch-norm?
            # transformed_frame = self.batch_norms[0](transformed_frame)
            if i != len(kernels) - 1:
                transformed_tensor = self.activation(transformed_tensor)

        return transformed_tensor


# Extensions
def to_np(t):
    return t.detach().cpu().numpy()


T.Tensor.np = property(lambda self: to_np(self))

if __name__ == "__main__":
    # Sanity check mask_sequence
    tensor = T.rand(2, 3, 4)
    mask = T.rand(2, 3) > 0.5
    masked = mask_sequence(tensor, mask)
    print(mask)
    print(masked)
