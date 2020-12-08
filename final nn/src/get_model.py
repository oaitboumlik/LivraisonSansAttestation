from .data_preparation import read_and_generate_test_set, read_and_generate_real_set
from model.model_config import get_backbones_param
from model.astgcn import ASTGCN
from .utils import evaluate_bis, evaluate_result
from mxnet import nd
from mxnet import gluon

model_name = "ASTGCN"
K = 3
ctx = None
optimizer = "adam"
learning_rate = 0.01
batch_size = 16
epochs = 10

n_vertex = 2
num_of_weeks = 1
num_of_days = 1
num_of_hours = 3
num_for_predict = 144
points_per_hour = 1

adj_filename = "../data/traffic/CE_adj.csv"


def apply_model(model_path, data_path, predicted_feature):
    feature = 0 if predicted_feature == "debit" else 1

    # output = net([test_w, test_d, test_r])
    all_data = read_and_generate_test_set(data_path,
                                          feature,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         144,
                                         points_per_hour=points_per_hour,
                                         )
    # test set ground truth
    true_value = (all_data['target'].transpose((0, 2, 1))
                  .reshape(all_data['target'].shape[0], -1))

    # testing set data loader
    test_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['season']),
            nd.array(all_data['month']),
            nd.array(all_data['week']),
            nd.array(all_data['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )
    # get model's structure
    all_backbones = get_backbones_param(K,
                                        num_of_weeks,
                                        num_of_days,
                                        num_of_hours,
                                        n_vertex,
                                        adj_filename, ctx)
    net = ASTGCN(144, all_backbones)
    net.load_parameters(model_path, ctx)
    evaluate_bis(net, test_loader, true_value)


def apply_model_real(model_path, data_path):
    all_data = read_and_generate_real_set(data_path,
                                          num_of_weeks,
                                          num_of_days,
                                          num_of_hours,
                                          num_for_predict,
                                          points_per_hour=points_per_hour,
                                          )

    # testing set data loader
    data_loader = gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            nd.array(all_data['season']),
            nd.array(all_data['month']),
            nd.array(all_data['week'])
        ),
        batch_size=batch_size,
        shuffle=False
    )
    # get model's structure
    all_backbones = get_backbones_param(K,
                                        num_of_weeks,
                                        num_of_days,
                                        num_of_hours,
                                        n_vertex,
                                        adj_filename, ctx)
    net = ASTGCN(144, all_backbones)
    net.load_parameters(model_path, ctx)
    return evaluate_result(net, data_loader)



if __name__ == '__main__':
    apply_model("../log/model_debit_16.3.params", "../data/traffic/ChmpElyse_clean.csv", "debit")
