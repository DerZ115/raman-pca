import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, HBox
from IPython.display import display

def plot_spectra_peaks(wns, signal, peaks, labels=None):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    line, = ax.plot(wns, signal[0,:])
    peakmarks = ax.scatter(wns[peaks[0]], signal[0,:][peaks[0]], 
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
            ydata = signal[i,:]
            line.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]] for peak in peaks[i]])
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
            ydata = signal[i,:]
            line.set_ydata(ydata)

            marks = np.array([[wns[peak], signal[i][peak]] for peak in peaks[i]])
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