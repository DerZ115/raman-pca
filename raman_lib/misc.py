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
    elif all([os.path.isdir(os.path.join(path, x)) for x in os.listdir(path)]):
        data = []
        labels = []
        files = []

        for path_inner in os.listdir(path):
            for file in os.listdir(os.path.join(path, path_inner)):
                filepath = os.path.join(path, path_inner, file)

                if filepath.lower().endswith(".csv"):
                    try:
                        spectrum = np.loadtxt(filepath, delimiter=",")
                    except:
                        print(f'An error occured while loading the file \'{filepath}\'')
                elif filepath.lower().endswith(".tsv"):
                    try:
                        spectrum = np.loadtxt(filepath, delimiter="\t")
                    except:
                        print(f'An error occured while loading the file \'{filepath}\'')
                elif filepath.lower().endswith(".txt"):
                    try:
                        spectrum = np.loadtxt(filepath, delimiter=",")
                    except:
                        print(f'An error occured while loading the file \'{filepath}\'')
                else:
                    try:
                        spectrum = convert_opus(filepath)
                    except:
                        print(f"File {file} does not match any inplemented file format. Skipping...")
                        continue
                
                data.append(spectrum)
                files.append(file)
                labels.append(path_inner)

        try:
            data = np.asarray(data, dtype=float)
        except ValueError:
            print("Data could not be combined into a single array. Perhaps some spectra cover different wavenumber ranges?")
            return None

        data = pd.DataFrame(data[:,:,1], columns=data[0,:,0])
        data.insert(0, "label", labels)
        if files:
            data.insert(1, "file", files)
    elif all([os.path.isfile(os.path.join(path, x)) for x in os.listdir(path)]):
        data = []
        labels = []
        files = []
        
        for file in os.listdir(path):
            filepath = os.path.join(path, file)

            if filepath.lower().endswith(".csv"):
                spectrum = np.loadtxt(filepath, delimiter=",")
            elif filepath.lower().endswith(".tsv"):
                spectrum = np.loadtxt(filepath, delimiter="\t")
            elif filepath.lower().endswith(".txt"):
                spectrum = np.loadtxt(filepath, delimiter=",")
            else:
                try:
                    spectrum = convert_opus(filepath)
                except:
                    print(f"File {file} does not match any inplemented file format. Skipping...")
                    continue

            data.append(spectrum)
            files.append(file)
            labels.append(path)
            
        try:
            data = np.asarray(data, dtype=float)
        except ValueError:
            print("Data could not be combined into a single array. Perhaps some spectra cover different wavenumber ranges?")
            return None

        data = pd.DataFrame(data[:,:,1], columns=data[0,:,0])
        data.insert(0, "label", labels)
        if files:
            data.insert(1, "file", files)
    
    else:
        print("Received unclear directory structure.")
        return None
    
    return data
