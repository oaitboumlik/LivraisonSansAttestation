import os

from astgcn.ASTGCN.lib.data_preparation import read_and_generate_dataset
from astgcn.ASTGCN.model.model_config import get_backbones_param
from astgcn.ASTGCN.model.astgcn import ASTGCN
from astgcn.ASTGCN.myinit import MyInit
from astgcn.ASTGCN.lib.utils import compute_val_loss, evaluate, predict
from time import time
from mxnet import nd
from mxnet import gluon
from mxnet import autograd

model_name = "ASTGCN"

num_of_weeks = 1
num_of_days = 1
num_of_hours = 3
num_for_predict = 12
points_per_hour = 1
n_vertex = 2

K = 3
ctx = None
optimizer = "adam"
learning_rate = 0.001
batch_size = 16
epochs = 70
adj_filename = "../data/traffic/CE_adj.csv"

sw = None

def run_model():
    all_data = read_and_generate_dataset("../data/traffic/ChmpElyse_clean.csv",
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour=points_per_hour,
                                         merge=False)
    # test set ground truth
    true_value = (all_data['test']['target'].transpose((0, 2, 1))
                  .reshape(all_data['test']['target'].shape[0], -1))

    # training set data loader
    train_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['train']['week']),
            nd.array(all_data['train']['day']),
            nd.array(all_data['train']['recent']),
            nd.array(all_data['train']['target'])
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['val']['week']),
            nd.array(all_data['val']['day']),
            nd.array(all_data['val']['recent']),
            nd.array(all_data['val']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['test']['week']),
            nd.array(all_data['test']['day']),
            nd.array(all_data['test']['recent']),
            nd.array(all_data['test']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # loss function MSE
    loss_function = gluon.loss.L2Loss()

    # get model's structure
    all_backbones = get_backbones_param(K,
                                        num_of_weeks,
                                        num_of_days,
                                        num_of_hours,
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

    # params_filename = os.path.join("../log",
    #                                '%s_init.params' % (model_name))
    #
    # net.save_parameters(params_filename)
    # # compute validation loss before training
    # compute_val_loss(net, val_loader, loss_function, sw, epoch=0)
    #
    # # compute testing set MAE, RMSE, MAPE before training
    # evaluate(net, test_loader, true_value, n_vertex, sw, epoch=0)

    # train model
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

            if sw:
                sw.add_scalar(tag='training_loss',
                              value=training_loss,
                              global_step=global_step)

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
        rmse = evaluate(net, test_loader, true_value, n_vertex, sw, epoch)

        # save epoch result
        params_filename = os.path.join("../log",
                                       '%s_epoch_%s_rsme_%s.params' % (model_name,
                                                               epoch, int(10*rmse)/10))
        net.save_parameters(params_filename)
        print('save parameters to file: %s' % (params_filename))

    # close SummaryWriter
    if sw:
        sw.close()

if __name__ == '__main__':
    run_model()
