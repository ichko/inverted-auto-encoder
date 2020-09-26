from pipelines.glyph_ae import ReverseAE
import utils.nn as tu
import utils as ut

import torch as T
import torch.nn as nn
import torchvision
from torchvision import transforms


class Classifier(tu.Module):
    def forward(self, x):
        in_size = x.size(1)
        if not hasattr(self, 'net'):
            self.criterion = nn.NLLLoss()

            self.net = nn.Sequential(
                nn.Flatten(),
                tu.dense(i=in_size, o=100, a=nn.Softmax(dim=1)),
                tu.dense(i=100, o=10, a=nn.Softmax(dim=1)),
            ).to(x.device)

        return self.net(x)


if __name__ == '__main__':
    from sklearn.neural_network import MLPClassifier

    msg_size = 128
    model = ReverseAE(msg_size, img_channels=1)
    model = model.to('cuda')
    model.make_persisted('.models/glyph-ae.h5')
    model.preload_weights()

    data = ut.pipe(
        ut.data.get_mnist_iterator,
        lambda g: ([model.decoder(X.to('cuda')), y.to('cuda')] for X, y in g),
        # lambda g: ut.mp.run_mp_generator(
        #     generator_ctor=lambda: g,
        #     buffer_size=4,
        #     num_processes=1
        # ),
        # lambda g: ([X.to('cuda'), y.to('cuda')] for X, y in g)
    )

    train = data(bs=128, train=True)
    test = data(bs=128, train=False)

    clf = Classifier().to('cuda')

    with ut.mp.fit(clf, train, epochs=10, optim_kw={'lr': 0.001}) as fit:
        fit.join()

    # clf = MLPClassifier(
    #     solver='adam',
    #     alpha=1e-5,
    #     hidden_layer_sizes=(100, 10),
    #     max_iter=500,
    #     random_state=1,
    #     shuffle=True,
    #     verbose=1,
    #     n_iter_no_change=9999999999,
    # )

    # clf.fit(model.decoder(X_train.to('cuda')).np, y_train.np)

    # score_train = clf.score(model.decoder(X_train.to('cuda')).np, y_train.np)
    # score_test = clf.score(model.decoder(X_test.to('cuda')).np, y_test.np)
    # print(score_train)
    # print(score_test)
