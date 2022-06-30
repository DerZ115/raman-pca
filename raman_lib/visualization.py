import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, HBox
from IPython.display import display
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             roc_curve,
                             auc)


def plot_spectra_peaks(wns, signal, peaks, labels=None):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    line, = ax.plot(wns, signal[0, :])
    peakmarks = ax.scatter(wns[peaks[0]], signal[0, :][peaks[0]],
                           c="red", marker="x", s=50, zorder=3)
    if labels is not None:
        ax.set_title(labels[0])

    ax.set_xlim(wns[0], wns[-1])
    ax.grid()

    ax.set_xlabel("Raman Shift ($\mathregular{cm^{-1}}$)",
                  fontdict={"weight": "bold", "size": 12})

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]]
                             for peak in peaks[i]])
            if len(marks) == 0:
                peakmarks.set_visible(False)
            else:
                peakmarks.set_visible(True)
                peakmarks.set_offsets(marks)
            if labels is not None:
                ax.set_title(labels[i])

            ax.relim()
            ax.autoscale_view()
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % len(signal)
            ydata = signal[i, :]
            line.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]]
                             for peak in peaks[i]])
            if len(marks) == 0:
                peakmarks.set_visible(False)
            else:
                peakmarks.set_visible(True)
                peakmarks.set_offsets(marks)
            if labels is not None:
                ax.set_title(labels[i])

            ax.relim()
            ax.autoscale_view()
            plt.draw()

    callback = Index()

    bnext = Button(description='Next')
    bprev = Button(description='Previous')

    buttons = HBox(children=[bprev, bnext])
    display(buttons)

    bnext.on_click(callback.next)
    bprev.on_click(callback.prev)

    plt.show()


def plot_confusion_matrix(y_pred, y_true, labels, title=None):

    conf_matrices = np.asarray(
        [confusion_matrix(y_true, y_pred[i, :]) for i in range(len(y_pred))]
    )

    conf_matrix_plot = conf_matrices.mean(axis=0)

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay(conf_matrix_plot).plot(values_format=".1f", ax=ax)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=90, va="center")
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=16)

    fig.tight_layout()

    plt.plot()


def plot_roc_curve(conf_scores, y, labels, name):
    aucs = []

    fig, ax = plt.subplots()

    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for row in conf_scores:
        fpr, tpr, _ = roc_curve(y, row)
        ax.plot(fpr, tpr, color="C0", alpha=0.2, linewidth=1)
        aucs.append(auc(fpr, tpr))

    mean_fpr, mean_tpr, _ = roc_curve(y, conf_scores.mean(axis=0))

    aucs_mean = np.mean(aucs)
    aucs_std = np.std(aucs)

    ax.plot(
        mean_fpr, mean_tpr, color="C0", linewidth=2,
        label=f"{name} (AUC = {aucs_mean:.4f} $\pm$ {aucs_std:.4f})"
    )

    ax.set_xlabel(
        f"False Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.set_ylabel(
        f"True Positive Rate (Positive label: {labels[1]})",
        fontsize=12
    )

    ax.legend(loc="lower right")
    fig.tight_layout()

    plt.plot()

    return np.array((mean_fpr, mean_tpr)), np.array((aucs_mean, aucs_std))
