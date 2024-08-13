from collections import defaultdict 
import pandas as pd
import numpy as np
import pickle
import os

class Logger(object):
    """
    This is a simple logger datastructure to store and load results on disk and convert to pandas dataframe
    """
    def __init__(self, folder = None, filename = None):
        self.folder = folder
        self.filename = filename if '.pkl' in filename else filename + '.pkl'
        
        self.data = defaultdict(list)

    def add(self, **args):
        for key, value in args.items():
            if isinstance(value, list):
                self.data[key].extend(value)
            else:
                self.data[key].append(value)
    
    def get(self, key, timestep = None):
        if timestep:
            return np.array(self.data[key][timestep])
        return np.array(self.data[key])
        
    def save(self):
        os.makedirs(self.folder, exist_ok=True)
        
        try:
            with open(os.path.join(self.folder, self.filename), 'wb') as file:
                #df = self.to_dataframe()
                pickle.dump(self.data, file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"An error occurred while saving data: {e}")

    def load(self):
        try:
            with open(os.path.join(self.folder, self.filename), 'rb') as file:
                self.data = pickle.load(file)
                return self.to_dataframe()
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return None

    def to_dataframe(self):
        self.data.pop('syn_loss', None)
        return pd.DataFrame.from_dict(self.data)
    
    def exists(self):
        return os.path.exists(os.path.join(self.folder, self.filename))

    def clear(self):
        for key, value in self.data.items():
            del self.data[key][:]
    
    def __len__(self):
        return len(self.data.values()[0])