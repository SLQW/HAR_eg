"""
Utilities for loading Human Activity Recognition Using Smartphones Data Set
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

@author: maffettone
"""
import os
import numpy as np
from pandas import read_csv
from torch.utils.data import TensorDataset, DataLoader
from torch import LongTensor, Tensor


def load_file(path):
    """
    Reads whitespace deliminated data file
    Parameters
    ----------
    path: basestring
        path to file to read

    Returns
    -------
    numpy array of values

    """
    df = read_csv(path, header=None, delim_whitespace=True)
    return df.values


def load_file_group(paths):
    """
    Reads a set of files and stacks their data as new channels in a channel first representation
    Parameters
    ----------
    paths: iterable of strings
        paths to files to read

    Returns
    -------
    numpy array of values
    """
    arrs = []
    for path in paths:
        data = load_file(path)
        arrs.append(data)
    return np.stack(arrs, axis=1)


def load_dataset_split(split, data_dir='../data/UCI HAR Dataset'):
    """

    Parameters
    ----------
    split: basestring in {'train', 'test'}
    data_dir: basestring, optional
        Directory of UCI dataset. Default download title from link used.
        https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

    Returns
    -------
    X: numpy array
    y: numpy array

    """
    assert split in ('train', 'test'), "Only train or test set available as dataset splits)"
    # Fixed feature set
    features = ('total_acc_x',
                'total_acc_y',
                'total_acc_z',
                'body_acc_x',
                'body_acc_y',
                'body_acc_z',
                'body_gyro_x',
                'body_gyro_y',
                'body_gyro_z')
    paths = []
    for feat in features:
        paths.append(os.path.join(data_dir, split, 'Inertial Signals', feat + '_' + split + '.txt'))
    X = load_file_group(paths)
    y = load_file(os.path.join(data_dir, split, 'y_' + split + '.txt')).squeeze().astype(np.long)
    return X, y


def _preprocess(X, y):
    """
    Default preprocessing function.
    Offsets y to be 0 indexed
    Parameters
    ----------
    X
    y

    Returns
    -------

    """
    y = y - 1
    return X, y


def load_dataset(batch_size=32, data_dir='../data/UCI HAR Dataset', preprocess=_preprocess, num_workers=1):
    """

    Parameters
    ----------
    batch_size: int
        batch size
    num_workers: int
        workers for multiprocessing on loader
    data_dir: basestring, optional
        Directory of UCI dataset. Default download title from link used.
        https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
    preprocess: function, optional
        Preprocessing function which takes in two args (X, y) and returns a preprocessed version of X and y.

    Returns
    -------
    train_loader: pytorch DataLoader
    test_loader: pytorch DataLoader

    """

    train_X, train_y = load_dataset_split('train', data_dir=data_dir)
    test_X, test_y = load_dataset_split('test', data_dir=data_dir)
    train_X, train_y = preprocess(train_X, train_y)
    test_X, test_y = preprocess(test_X, test_y)
    train_dataset = TensorDataset(Tensor(train_X), LongTensor(train_y))
    test_dataset = TensorDataset(Tensor(test_X), LongTensor(test_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train, test = load_dataset(data_dir='../data/UCI HAR Dataset')
    dataiter = iter(train)
    X, y = dataiter.next()
    print(X.shape)
    print(y.shape)
