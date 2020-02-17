"""
Recreation of initial 1D convolutional neural net presented here:
https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

@author: maffettone
"""
import time
import torch
import random
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from HAR.data_utils.UCI import load_dataset
from HAR.data_utils.pkl import save_obj


class CNN(nn.Module):

    def __init__(self, input_shape, n_classes,
                 filters=(64, 64), kernels=(3, 3), dropout=0.5, pool=2, dense=(100,)):
        """
        Module class to create variable sized 1D convolutional neural networks with a general architecture:
        Conv1d -> ... -> Conv1d -> Dropout -> MaxPool -> Dense -> ... -> Dense -> Dense(classifier)

        Parameters
        ----------
        input_shape: tuple of int
            shape of input data (features, n_x)
        n_classes: int
            Number of classes for classifier
        filters: tuple of int
            Number of filters for each convolutional layer in the model
        kernels: tuple of int
            Size of convolution kernel for each layer in the model
        dropout: float
            Dropout rate for model 0<= dropout < 1
        pool: int
            Size of maxpool window and stride
        dense: tuple of int
            Size of dense layers
        """
        super(CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        n_features, n_timesteps = input_shape
        l_in = n_timesteps  # Tracking datashape
        for i in range(len(filters)):
            if i == 0:
                self.conv_layers.append(nn.Conv1d(n_features, filters[i], kernels[i]))
            else:
                self.conv_layers.append(nn.Conv1d(filters[i - 1], filters[i], kernels[i]))
            l_in = l_in - (kernels[i] - 1) - 1
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(pool)
        l_in = int(np.floor((l_in + 2 - 1 * (pool - 1) - 1) / pool + 1))
        self.dense_layers = nn.ModuleList()
        self.flatten_size = l_in * filters[-1]
        for i in range(len(dense)):
            if i == 0:
                self.dense_layers.append(nn.Linear(self.flatten_size, dense[i]))
            else:
                self.dense_layers.append(nn.Linear(dense[i - 1], dense[i]))
        self.classifier = nn.Linear(dense[-1], n_classes)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(-1, self.flatten_size)
        for dense in self.dense_layers:
            x = F.relu(dense(x))
        x = self.classifier(x)
        return x


def test_acc(net, test_loader):
    """
    Calculate test accuracy averaged over batches as a percentage
    Parameters
    ----------
    net: nn.Module
        Neural net for calculating outputs
    test_loader: DataLoader
        Test dataset loader

    Returns
    -------
    float
        accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            X, labels = data
            outputs = net(X)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def class_breakdown(net, test_loader, n_classes):
    """
    Calculates accuracy per class
    Parameters
    ----------
    net: nn.Module
        Neural net for calculating outputs
    test_loader: DataLoader
        Test dataset loader
    n_classes: int
        number of classes in calssifer

    Returns
    -------
    List of class accuracy
    """
    class_correct = np.zeros(n_classes)
    class_total = np.zeros(n_classes)
    with torch.no_grad():
        for data in test_loader:
            X, labels = data
            outputs = net(X)
            _, predicted = torch.max(outputs, dim=1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    return list(100 * class_correct / class_total)


def test_loss(net, criterion, test_loader):
    """
    Calculate test loss per batch without gradients
    Parameters
    ----------
    net: nn.Module
        Neural net for calculating outputs
    criterion:
        Loss function
    test_loader: DataLoader
        Test dataset loader

    Returns
    -------
    Average loss per batch
    """
    with torch.no_grad():
        loss = 0.
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
        return loss / (i + 1)


def training(output_dir, n_epochs=10, seed=None, verbose=False,
             train_loader=None, test_loader=None, batch_size=32, data_dir='../data/UCI HAR Dataset',
             filters=(64, 64), kernels=(3, 3), dropout=0.5, pool=2, dense=(100,),
             lr=0.001, betas=(0.9, 0.999)
             ):
    """

    Parameters
    ----------
    output_dir: basestring
        Directory for saving of models and results
    n_epochs: int
        Number of epochs in a training round
    verbose: bool
        Verbosity argument
    train_loader: DataLoader, optional
        DataLoader for training set.
        If train_loader or test_loader are not given, then the loading options are used,
        and dataset is loaded from data_dir.
    test_loader: DataLoader, optional
        DataLoader for test set.
        If train_loader or test_loader are not given, then the loading options are used,
        and dataset is loaded from data_dir.
    batch_size: int
        Size of batch for data training
    data_dir: basestring
        Directory containing the HAR dataset
    filters: tuple of int
        Number of filters for each convolutional layer in the model
    kernels: tuple of int
        Size of convolution kernel for each layer in the model
    dropout: float
        Dropout rate for model 0<= dropout < 1
    pool: int
        Size of maxpool window and stride
    dense: tuple of int
        Size of dense layers
    lr: float
        Learning rate for Adam optimizer
    betas: (float, float)
        Betas for Adam optimizer

    Returns
    -------
    metrics: dictionary
        Dictionary of metrics across training run, saved at each epoch
    """
    start_time = time.time()

    # Presets
    n_classes = 6
    print_rate = 100
    metrics = {'train_loss': [],
               'test_loss': [],
               'train_acc': [],
               'test_acc': []}

    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Load data. Optionality allows for multiple runs with single data load step.
    if (not train_loader) or (not test_loader):
        train_loader, test_loader = load_dataset(batch_size=batch_size, data_dir=data_dir)
    input_shape = tuple(iter(train_loader).next()[0].shape[1:])

    cnn = CNN(input_shape=input_shape,
              n_classes=n_classes,
              filters=filters,
              kernels=kernels,
              dropout=dropout,
              pool=pool,
              dense=dense)
    if verbose:
        print(cnn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr, betas=betas)

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        count = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Acc
            _, predicted = torch.max(outputs.data, dim=1)
            count += labels.size(0)
            running_correct += (predicted == labels).sum().item()

            if verbose:
                if i % print_rate == print_rate - 1:
                    print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / count))

        # Update metrics dictionary
        metrics['train_loss'].append(running_loss / count)
        metrics['train_acc'].append(100 * running_correct / count)
        metrics['test_loss'].append(test_loss(cnn, criterion, test_loader))
        metrics['test_acc'].append(test_acc(cnn, test_loader))

    torch.save(cnn.state_dict(), os.path.join(output_dir, 'trained_state.pkl'))
    save_obj(metrics, os.path.join(output_dir, 'metrics_dict.pkl'))

    print("Finised training in {:.3f} minutes".format((time.time() - start_time) / 60))
    print("Test accuracy : {:.2f}%".format(metrics['test_acc'][-1]))

    if verbose:
        for i, acc in enumerate(class_breakdown(cnn, test_loader, n_classes)):
            print("Accuracy of class {}: {:.2f}%".format(i, acc))

    return metrics


def run_experiment(output_dir, repeats=10, n_epochs=10, verbose=False,
                   batch_size=32, data_dir='../data/UCI HAR Dataset',
                   filters=(64, 64), kernels=(3, 3), dropout=0.5, pool=2, dense=(100,),
                   lr=0.001, betas=(0.9, 0.999)
                   ):
    """
    Runs an experiment of multiple rounds of training given by repeats.

    Parameters
    ----------
    output_dir: basestring
        Directory for saving of models and results
    repeats: int
        Number of repeat rounds of training for an experiment
    n_epochs: int
        Number of epochs in a training round
    verbose: bool
        Verbosity argument
    batch_size: int
        Size of batch for data training
    data_dir: basestring
        Directory containing the HAR dataset
    filters: tuple of int
        Number of filters for each convolutional layer in the model
    kernels: tuple of int
        Size of convolution kernel for each layer in the model
    dropout: float
        Dropout rate for model 0<= dropout < 1
    pool: int
        Size of maxpool window and stride
    dense: tuple of int
        Size of dense layers
    lr: float
        Learning rate for Adam optimizer
    betas: (float, float)
        Betas for Adam optimizer

    Returns
    -------
    avg_metrics: dictionary
        Dictionary of metrics averaged across all training repeats
    """
    start_time = time.time()
    train_loader, test_loader = load_dataset(batch_size=batch_size, data_dir=data_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    metrics = []

    for i in range(repeats):
        print("Training for experiment {}".format(i+1))
        _dir = os.path.join(output_dir, "exp_{}".format(i + 1))
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
        results = training(_dir, n_epochs=n_epochs, verbose=verbose, batch_size=batch_size,
                           train_loader=train_loader, test_loader=test_loader,
                           filters=filters, kernels=kernels, dropout=dropout, pool=pool,
                           dense=dense, lr=lr, betas=betas)

        # Store final value of each metric
        d = {}
        for key in results:
            d[key] = results[key][-1]
        metrics.append(d)
        print("".join(["=" for _ in range(80)]))
        if verbose:
            print()

    avg_metrics = {}
    for key in metrics[0]:
        avg_metrics[key] = 0.
    for m in metrics:
        for key in m:
            avg_metrics[key] += m[key]
    for key in avg_metrics:
        avg_metrics[key] /= len(metrics)

    print("Finised training {} models in {:.3f} minutes".format(repeats, (time.time() - start_time) / 60))
    print("Average metrics over {} runs".format(repeats))
    print("============================")
    for key in avg_metrics:
        print("{}: {:8.3f}".format(key, avg_metrics[key]))

    return avg_metrics
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    _ = run_experiment('../../tmp', data_dir='../../data/UCI HAR Dataset', verbose=True, n_epochs=10)
