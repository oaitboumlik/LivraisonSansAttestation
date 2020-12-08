import csv
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
from .metrics import mean_squared_error


def search_data(num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    '''
    Parameters
    ----------
    sequence_length: int, length of all history data

    num_of_batches: int, the number of batches will be used for training

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    units: int, week: 7 * 24, day: 24, recent(hour): 1

    points_per_hour: int, number of points per hour, depends on data

    Returns
    ----------
    list[(start_idx, end_idx)]
    '''

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12, for_train=True):
    '''
    Parameters
    ----------
    data_sequence: np.ndarray
                   shape is (sequence_length, num_of_vertices, num_of_features)

    num_of_weeks, num_of_days, num_of_hours: int

    label_start_idx: int, the first index of predicting target

    num_for_predict: int,
                     the number of points will be predicted for each sample

    points_per_hour: int, default 12, number of points per hour

    Returns
    ----------
    week_sample: np.ndarray
                 shape is (num_of_weeks * points_per_hour,
                           num_of_vertices, num_of_features)

    day_sample: np.ndarray
                 shape is (num_of_days * points_per_hour,
                           num_of_vertices, num_of_features)

    hour_sample: np.ndarray
                 shape is (num_of_hours * points_per_hour,
                           num_of_vertices, num_of_features)

    target: np.ndarray
            shape is (num_for_predict, num_of_vertices, num_of_features)
    '''

    if label_start_idx + num_for_predict > data_sequence.shape[0] and for_train:
        return None

    # season
    week_indices = search_data(num_of_weeks,
                               label_start_idx, num_for_predict,
                               3 * 28 * 24, points_per_hour)
    if not week_indices:
        return None

    # month
    day_indices = search_data(num_of_days,
                              label_start_idx, num_for_predict,
                              28 * 24, points_per_hour)
    if not day_indices:
        return None

    # week
    hour_indices = search_data(num_of_hours,
                               label_start_idx, num_for_predict,
                               7*24, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i: j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i: j]
                                  for i, j in hour_indices], axis=0)
    if not for_train:
        return week_sample, day_sample, hour_sample, None

    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1

    return A


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real[0]
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader, loss_function, sw, epoch):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    val_loader_length = len(val_loader)
    tmp = []
    for index, (val_w, val_d, val_r, val_t) in enumerate(val_loader):
        output = net([val_w, val_d, val_r])
        l = loss_function(output, val_t)
        tmp.extend(l.asnumpy().tolist())
        print('validation batch %s / %s, loss: %.2f' % (
            index + 1, val_loader_length, l.mean().asscalar()))

    validation_loss = sum(tmp) / len(tmp)
    if sw:
        sw.add_scalar(tag='validation_loss',
                      value=validation_loss,
                      global_step=epoch)
    print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))


def predict(net, test_loader):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''

    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r, _) in enumerate(test_loader):
        prediction.append(net([test_w, test_d, test_r]).asnumpy())
        print('predicting testing set batch %s / %s' % (index + 1,
                                                        test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction


def predict_real(net, test_loader):
    test_loader_length = len(test_loader)
    prediction = []
    for index, (test_w, test_d, test_r) in enumerate(test_loader):
        prediction.append(net([test_w, test_d, test_r]).asnumpy())
        print('predicting testing set batch %s / %s' % (index + 1,
                                                        test_loader_length))
    prediction = np.concatenate(prediction, 0)
    return prediction


# evaluate for train model
def evaluate(net, test_loader, true_value, sw, epoch):
    '''
    compute RMSE scores of the prediction
    for 144 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    sw: mxboard.SummaryWriter

    epoch: int, current epoch

    '''
    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    for i in [168]:
        print('current epoch: %s, predict %s points' % (epoch, i))

        rmse = mean_squared_error(true_value[0, :2 * i:2],
                                  prediction[0, :2 * i:2]) ** 0.5

        print('RMSE: %.2f' % (rmse))
        print()
        if sw:
            sw.add_scalar(tag='RMSE_%s_points' % (i),
                          value=rmse,
                          global_step=epoch)
    return rmse


# evaluate for test model
def evaluate_bis(net, test_loader, true_value):
    """

    Parameters
    ----------
    net: model
    test_loader: test set
    true_value: true value

    Returns
    -------

    """

    prediction = predict(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    for i in [144]:
        rmse = mean_squared_error(true_value[0, :2 * i:2],
                                  prediction[0, :2 * i:2]) ** 0.5
        plt.plot(true_value[0, :2 * i:2], label="true value")
        plt.plot(prediction[0, :2 * i:2], label="prediction")
        plt.title('prediction RMSE: %.2f' % (rmse))
        plt.legend()
        plt.show()


def evaluate_result(net, test_loader):
    """

    Parameters
    ----------
    net: model
    test_loader: test set
    true_value: true value

    Returns: pred: prediction
    -------

    """

    prediction = predict_real(net, test_loader)
    prediction = (prediction.transpose((0, 2, 1))
                  .reshape(prediction.shape[0], -1))
    for i in [144]:
        pred = prediction[:, :2 * i:2]

    return pred

