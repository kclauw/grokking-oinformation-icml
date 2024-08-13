from pathlib import Path 
import numpy as np
import pandas as pd
import random
import torch
import shutil
import os
import json
from itertools import groupby
from operator import itemgetter
import hashlib
from copy import deepcopy



        
def prune(cfg, model, multiplets):

    layers = list(multiplets.keys())
    # pass parameters from initial model
    
    for i, (name, param) in enumerate(model.named_parameters()):
        #layer_name = name.split('.')[1]
        
        if name in layers:
            idx = list(multiplets[name])
            mask = torch.zeros(param.data.shape).to(cfg.device)
           
            if "weight" in name:
                for i in idx:	
                    mask[i] = 1	
                param.data = (param.data * mask).to(cfg.device)

            elif "bias" in name and cfg.prune_bias:
                for i in idx:	
                    mask[i] = 1	
                param.data = (param.data * mask).to(cfg.device)
           

def save_dataframe_to_pickle(folder, filename, df : pd.DataFrame):
    create_folder(folder)
    df.to_pickle(os.path.join(folder, filename))
    
def file_exists(folder, filename, extension = ""):
    return Path(os.path.join(folder, filename + extension)).exists()

def create_folder(folder):
    Path(folder).mkdir(parents=True, exist_ok=True)

def seed_everything(seed : int):
    # set seeds for determinism
    torch.manual_seed(seed)
    #torch.set_default_dtype(dtype)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    random.seed(seed)

def add_cfg_arguments_to_dataframe(cfg, df, i, feature_name, train_acc, test_acc, train_loss, test_loss, ignore_keys = []):
    df["dataset"] = [cfg.dataset.unique_name for _ in range(len(df))]
    df["parameters"] = [cfg.model_parameters for _ in range(len(df))]
    df["seed"] = [cfg.training.seed for _ in range(len(df))]
    df["step"] = [i for _ in range(len(df))]
    df["feature_name"] = [feature_name for _ in range(len(df))]
    
    df["train_acc"] = [train_acc for _ in range(len(df))]
    df["test_acc"] = [test_acc for _ in range(len(df))]
    df["train_loss"] = [train_loss for _ in range(len(df))]
    df["test_loss"] = [test_loss for _ in range(len(df))]

    
    for key, value in cfg.oinfo.items():
        df[key] = [value for _ in range(len(df))]
    
    for key, value in cfg.clustering.items():
        df[key] = [value for _ in range(len(df))]
    
    df["model"] = [cfg.model.unique_name for _ in range(len(df))]


def create_unique_filename(data_dict, exclude_cols=[]):
    # Filter out excluded columns
    filtered_dict = {key: value for key, value in data_dict.items() if key not in exclude_cols}
    
    # Sort keys to ensure consistent order
    sorted_keys = sorted(filtered_dict.keys())

    # Join key-value pairs with underscores
    filename_parts = []
    for key in sorted_keys:
        tmp_key = deepcopy(key)
        if "." in key:
            tmp_key = key.split('.')[-1]
        filename_parts.append(f"{tmp_key}_{filtered_dict[key]}")

    # Combine parts with hyphens for separation
    filename = "_".join(filename_parts)

    return filename

def remove_folder_and_contents(folder_path):
    """
    Remove a folder and its contents.

    Parameters:
    - folder_path (str): Path to the folder to be removed.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove the folder and its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents successfully removed.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error occurred: {e}")
        
def load_dataframe_from_pickle(folder, filename = None):
    try:
        #print("opening")
        if filename is not None:
            folder = os.path.join(folder, filename)
        return pd.read_pickle(folder)
    except EOFError:
        #print("Error: Ran out of input while loading the pickle file.")
        return None
    
def get_cluster_folder(cfg):
    return os.path.join(cfg.cluster_folder, 'results', cfg.dataset.unique_name, cfg.optimizer.name, cfg.model.unique_name, cfg.model_parameters, "seed_%d" % (cfg.seed))

def read_from_json(folder, filename):
    create_folder(folder)
    try:
        with open(os.path.join(folder, filename), 'r') as json_file:
            loaded_data = json.load(json_file)
        return loaded_data
    except FileNotFoundError:
        print(f"Error: File '{os.path.join(folder, filename)}' not found.")
        return []

def save_to_json(data, folder, filename):
    create_folder(folder)
    with open(os.path.join(folder, filename), 'w') as json_file:
        json.dump(data, json_file, indent=2)

from collections import defaultdict


def filter_dict_by_value(d, values_to_keep):
    if not isinstance(values_to_keep, list):
        values_to_keep = [values_to_keep]
    return {k: v for k, v in d.items() if v in values_to_keep}

def group_params_by_keys(param_list, keys):
    tmp_grouped_params = defaultdict(lambda: defaultdict(list))
    

    for param in param_list:
        run_key = tuple(filter_dict(param, keys).items())

        for k in keys:
            tmp_grouped_params[run_key][k].append(param[k])
       
    

    grouped_params = []
    for key, value in tmp_grouped_params.items():
        new_params = dict(key)
        for k, v in value.items():
            new_params[k] = v
        grouped_params.append(new_params)
  
    return grouped_params
 

def filter_dict(d, keys_to_remove):
    return {k: v for k, v in d.items() if k not in keys_to_remove}

def dict_to_hash(params_dict):
    try:
        params_json = json.dumps(params_dict, sort_keys=True)
        hash_object = hashlib.sha256(params_json.encode())
        return hash_object.hexdigest()
    except Exception as e:
        print("An error occurred:", e)
        return None
    

import subprocess

def copy_to_cluster(local_file_path, remote_user, remote_host, remote_path):
    """
    Copy a file from local disk to a remote server or cluster using SSH.

    Args:
        local_file_path (str): Path to the local file to be copied.
        remote_user (str): Username on the remote server.
        remote_host (str): Hostname or IP address of the remote server.
        remote_path (str): Path on the remote server where the file will be copied.
    """
    # Use scp to securely copy the file to the remote server
    subprocess.run(['scp', local_file_path, f'{remote_user}@{remote_host}:{remote_path}'])


def zip_files(file_paths, zip_file_path):
    """
    Zip multiple files into a single zip file on disk using the zip command in Linux.

    Args:
        file_paths (list): List of paths to the files to be zipped.
        zip_file_path (str): Path to save the zipped file.
    """
    # Use the '-r' flag to recursively zip directories
    subprocess.run(['tar', '-I', 'pigz', '-cf', zip_file_path] + file_paths)

def unzip_file(zip_file_path, extract_to):
    """
    Unzip a file on disk using the unzip command in Linux.

    Args:
        zip_file_path (str): Path to the zipped file.
        extract_to (str): Directory to extract the contents of the zip file.
    """
    subprocess.run(['unzip', zip_file_path, '-d', extract_to])
    
    