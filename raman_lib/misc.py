import numpy as np
import pandas as pd
import os

from .opus_converter import convert_opus


def mode(x):
    values, counts = np.unique(x, return_counts=True)
    i = counts.argmax()
    return values[i]


def load_data(path):
    if path.lower().endswith(".csv") or \
       path.lower().endswith(".txt") or \
       path.lower().endswith(".tsv"):
        data = pd.read_csv(path)

    else:
        data = []
        labels = []
        files = []

        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                filepath = os.path.join(path, dir, file)

                if filepath.lower().endswith(".csv"):
                    spectrum = np.loadtxt(filepath, sep=",")
                elif filepath.lower().endswith(".tsv"):
                    spectrum = np.loadtxt(filepath, sep="\t")
                elif filepath.lower().endswith(".txt"):
                    spectrum = np.loadtxt(filepath)
                else:
                    try:
                        spectrum = convert_opus(filepath)
                    except:
                        print(f"File {file} does not match any inplemented file format. Skipping...")
                
                data.append(spectrum)
                files.append(file)
                labels.append(dir)

        try:
            data = np.asarray(data, dtype=float)
        except ValueError:
            print("Data could not be combined into a single array. Perhaps some spectra cover different wavenumber ranges?")
            return None

        data = pd.DataFrame(data[:,:,1], columns=data[0,:,0])
        data.insert(0, "label", labels)
        if files:
            data.insert(1, "file", files)

    return data
