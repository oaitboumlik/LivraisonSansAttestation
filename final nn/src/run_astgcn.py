import os
from .data_preparation import read_and_generate_dataset
from model.model_config import get_backbones_param
from model.astgcn import ASTGCN
from astgcn.ASTGCN.myinit import MyInit
from .utils import compute_val_loss, evaluate
from time import time
import mxnet
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from math import inf

model_name = "ASTGCN"
n_vertex = 2
K = 3
ctx = mxnet.gpu(0)
optimizer = "adam"
learning_rate = 0.005
batch_size = 16
epochs = 15

num_of_seasons = 2
num_of_months = 2
num_of_weeks = 3
num_for_predict = 144
points_per_hour = 1
adj_filename = "../data/traffic/CE_adj.csv"

sw = None


def run_model(data_file, predicted_feature):
    """

    Parameters
    ----------
    cleaned_data_path

    Returns Nan but register the best model
    -------

    """


    rmse_min = inf
    feature = 0 if predicted_feature == "debit" else 1
    all_data = read_and_generate_dataset(data_file,
                                         feature,
                                         num_of_seasons,
                                         num_of_months,
                                         num_of_weeks,
                                         num_for_predict,
                                         points_per_hour=points_per_hour,
                                         test_points=144,
                                         merge=False)
    # test set ground truth
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # training set data loader
    train_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['train']['season'], ctx=ctx),
            nd.array(all_data['train']['month'], ctx=ctx),
            nd.array(all_data['train']['week'], ctx=ctx),
            nd.array(all_data['train']['target'], ctx=ctx)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['val']['season'], ctx=ctx),
            nd.array(all_data['val']['month'], ctx=ctx),
            nd.array(all_data['val']['week'], ctx=ctx),
            nd.array(all_data['val']['target'], ctx=ctx)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['test']['season'], ctx=ctx),
            nd.array(all_data['test']['month'], ctx=ctx),
            nd.array(all_data['test']['week'], ctx=ctx),
            nd.array(all_data['test']['target'], ctx=ctx)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # loss function MSE
    loss_function = gluon.loss.L2Loss()

    # get model's structure
    all_backbones = get_backbones_param(K,
                                        num_of_seasons,
                                        num_of_months,
                                        num_of_weeks,
                                        n_vertex,
                                        adj_filename, ctx)

    net = ASTGCN(num_for_predict, all_backbones)
    net.initialize(ctx=ctx)
    for val_w, val_d, val_r, val_t in val_loader:
        net([val_w, val_d, val_r])
        break
    net.initialize(ctx=ctx, init=MyInit(), force_reinit=True)

    # initialize a trainer to train model
    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': learning_rate})

    global_step = 1
    for epoch in range(1, epochs + 1):

        for train_w, train_d, train_r, train_t in train_loader:

            start_time = time()

            with autograd.record():
                output = net([train_w, train_d, train_r])
                l = loss_function(output, train_t)
            l.backward()
            trainer.step(train_t.shape[0])
            training_loss = l.mean().asscalar()


            print('global step: %s, training loss: %.2f, time: %.2fs'
                  % (global_step, training_loss, time() - start_time))
            global_step += 1

        # logging the gradients of parameters for checking convergence
        for name, param in net.collect_params().items():
            if not sw:
                break
            try:
                sw.add_histogram(tag=name + "_grad",
                                 values=param.grad(),
                                 global_step=global_step,
                                 bins=1000)
            except:
                print("can't plot histogram of {}_grad".format(name))

        # compute validation loss
        compute_val_loss(net, val_loader, loss_function, sw, epoch)

        # evaluate the model on testing set
        rmse = evaluate(net, test_loader, true_value, sw, epoch)

        if rmse <= rmse_min:
            # save epoch result
            params_filename = os.path.join("../bin",
                                           'model_%s_%s.params' % (predicted_feature, round(rmse, 2)))
            net.save_parameters(params_filename)
            print('save parameters to file: %s' % (params_filename))
            rmse_min = rmse


if __name__ == '__main__':
    run_model()
