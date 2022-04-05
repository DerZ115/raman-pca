import numpy as np
import pandas as pd
import os

from .opus_converter import convert_opus


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    i = counts.argmax()
    return values[i]


def load_data(path):
    if path.endswith(".csv") or path.endswith(".txt") or path.endswith(".tsv"):
        data = pd.read_csv(path)

    else:
        data = []
        labels = []

        for dir in os.listdir(path):
            print(dir)
            for file in os.listdir(os.path.join(path, dir)):
                filepath = os.path.join(path, dir, file)

                if filepath.endswith(".csv"):
                    data.append(np.loadtxt(filepath, sep=","))
                elif filepath.endswith(".tsv"):
                    data.append(np.loadtxt(filepath, sep="\t"))
                elif filepath.endswith(".txt"):
                    data.append(np.loadtxt(filepath))
                else:
                    try:
                        data.append(convert_opus(filepath))
                    except:
                        raise ValueError(
                            f"File {file} does not match any inplemented file format."
                             "Use either plaintext (.csv, .tsv, .txt) or"
                             "binary OPUS (.0, .1, ...) files")

                labels.append(dir)
    
        data = np.asarray(data)

        data = pd.DataFrame(data[:,:,1], columns=data[0,:,0])
        data.insert(0, "label", labels)

    return data
