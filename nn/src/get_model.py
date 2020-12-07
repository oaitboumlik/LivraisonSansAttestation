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


def apply_model(model_path, data_path):
    from run_astgcn import K, \
        num_of_hours, num_of_days, \
        num_of_weeks, n_vertex, \
        adj_filename, ctx
    all_backbones = get_backbones_param(K,
                                        num_of_weeks,
                                        num_of_days,
                                        num_of_hours,
                                        n_vertex,
                                        adj_filename, ctx)

    net = ASTGCN(12, all_backbones)
    net.load_parameters(model_path, ctx)
    # output = net([test_w, test_d, test_r])


if __name__ == '__main__':
    apply_model("../log/ASTGCN_epoch_2_rsme_247.8.params", "")
