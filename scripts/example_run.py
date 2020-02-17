"""Script that mirrors example_run notebook."""
import sys
sys.path.append('../')
import os
import numpy as np
import matplotlib.pyplot as plt
from HAR.ml.CNN1d import training, run_experiment
from HAR.data_utils.pkl import load_obj

if __name__ == '__main__':
    avg_metrics = run_experiment('../tmp', repeats=5, verbose=False, n_epochs=10)
    exp_dirs = [os.path.join('../tmp', d) for d in os.listdir('../tmp') if os.path.isdir(os.path.join('../tmp', d))]
    fig, axes = plt.subplots(len(exp_dirs), 2, figsize=(15, 30))
    axes[0, 0].set_title('Loss', size=30)
    axes[0, 1].set_title('Accuracy (%)', size=30)
    for i, d in enumerate(exp_dirs):
        metrics = load_obj(os.path.join(d, 'metrics_dict.pkl'))
        axes[i, 0].plot(np.arange(len(metrics['train_loss'])), metrics['train_loss'], 'r', label='train')
        axes[i, 0].plot(np.arange(len(metrics['train_loss'])), metrics['test_loss'], 'b', label='test')
        axes[i, 1].plot(np.arange(len(metrics['train_loss'])), metrics['train_acc'], 'r', label='train')
        axes[i, 1].plot(np.arange(len(metrics['train_loss'])), metrics['test_acc'], 'b', label='test')
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.savefig('../figures/example_run.png')