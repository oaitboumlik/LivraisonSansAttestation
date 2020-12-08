# -*- coding:utf-8 -*-
import numpy as np
from .utils import get_sample_indices
from .data_preprocessing import trafic_into_np


#
def normalization(train, val, test):
    """
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    """

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean, std = stat(train)

    train_norm = normalize(train, mean, std)
    val_norm = normalize(val, mean, std)
    test_norm = normalize(test, mean, std)

    return {'mean': mean, 'std': std}, train_norm, val_norm, test_norm


def stat(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def read_and_generate_real_set(graph_signal_matrix_filename,
                               num_of_weeks, num_of_days,
                              num_of_hours, horizon,
                              points_per_hour=1,
                             ):
    data_seq = trafic_into_np(graph_signal_matrix_filename)
    data_len = data_seq.shape[0]

    all_samples = []
    for idx in range(data_len):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, horizon,
                                    points_per_hour, False)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, _ = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
        ))

    known_set = [np.concatenate(i, axis=0) for i in zip(*all_samples)]
    target_set = all_samples[-1]

    known_week, known_day, known_hour = known_set
    test_week, test_day, test_hour = target_set
    print('estimated data: week: {}, day: {}, recent: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape))
    mean, std = stat(known_week)
    test_week_norm = normalize(test_week, mean, std)
    mean, std = stat(known_day)
    test_day_norm = normalize(test_day, mean, std)
    mean, std = stat(known_hour)
    test_hour_norm = normalize(test_hour, mean, std)

    all_data = {
        'season': test_week_norm,
        'month': test_day_norm,
        'week': test_hour_norm,
    }
    return all_data


def read_and_generate_test_set(graph_signal_matrix_filename,
                               predicted_feature,
                               num_of_weeks, num_of_days,
                              num_of_hours, horizon,
                              last_x_idx,
                              points_per_hour=1,
                             ):

    """
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 1, depends on data

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    """

    data_seq = trafic_into_np(graph_signal_matrix_filename)
    data_len = data_seq.shape[0]
    all_samples = []
    for idx in range(data_len):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, horizon,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, predicted_feature, :]
        ))

    known_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[:len(all_samples) - last_x_idx])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[len(all_samples) - last_x_idx: ])]

    known_week, known_day, known_hour, _kt = known_set
    test_week, test_day, test_hour, test_target = testing_set
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))
    mean, std = stat(known_week)
    test_week_norm = normalize(test_week, mean, std)
    mean, std = stat(known_day)
    test_day_norm = normalize(test_day, mean, std)
    mean, std = stat(known_hour)
    test_hour_norm = normalize(test_hour, mean, std)

    all_data = {
            'season': test_week_norm,
            'month': test_day_norm,
            'week': test_hour_norm,
            'target': test_target
    }
    return all_data


def read_and_generate_dataset(graph_signal_matrix_filename,
                              predicted_feature,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=1,
                              percent_train=0.8,
                              test_points=144,
                              merge=False):
    """
    Parameters
    ----------
    graph_signal_matrix_filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 1, depends on data

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    """

    data_seq = trafic_into_np(graph_signal_matrix_filename)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, predicted_feature, :]
        ))
    split_line2 = len(all_samples) - test_points
    split_line1 = int(split_line2 * percent_train)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set
    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)
    all_data = {
        'train': {
            'season': train_week_norm,
            'month': train_day_norm,
            'week': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'season': val_week_norm,
            'month': val_day_norm,
            'week': val_recent_norm,
            'target': val_target
        },
        'test': {
            'season': test_week_norm,
            'month': test_day_norm,
            'week': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'season': week_stats,
            'month': day_stats,
            'week': recent_stats
        }
    }

    return all_data
