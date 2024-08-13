from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import json
import os 

import pandas as pd
from dotmap import DotMap

from utils.misc import create_folder

from loggers.simple_logger import Logger
from tasks.train import get_train_filenames
from utils.config import update_cfg_from_dict, initialize_hydra
from tasks.oinfo import get_oinfo_filename, load_oinfo_files


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


def filter_parameters_in_dataframe(dataframe, parameters):
    tmp_dict = dict()
    for key, value in parameters.items():
        if key in dataframe.columns:
            tmp_dict[key] = value
    return tmp_dict

def filter_dataframe(results, **kwargs):
    for column, value in kwargs.items():
        if isinstance(value, list):
            # Filter using multiple values (list)
            results = results[results[column].isin(value)]
        else:
            # Filter using a single value
            results = results[results[column] == value]
    return results.reset_index(drop=True)
    
def convert_column_to_row(df, column_name):
    total_col = []
    
    if not isinstance(column_name, list):
        column_name = [column_name]
   
    for cn in column_name:
        df_tmp = deepcopy(df)
        df_tmp["metric_value"] = df_tmp[cn]
        df_tmp["metric_name"] = [cn for _ in range(len(df))]
        #del df_tmp[column_name]
        total_col.append(df_tmp)
    return pd.concat(total_col).reset_index(drop=True)
    
def create_metric_dataframe(df, metric_name, metric_value, extra_columns):
    df_tmp = dict()
    df_tmp["metric_value"] = df_tmp[metric_value]
    df_tmp["metric_name"] = [metric_name for _ in range(len(df))]
    for c in extra_columns:
        df_tmp[c] = df[c]
    
    return pd.DataFrame.from_dict(df_tmp)

def download_file_from_cluster(sftp, cluster_result_folder, cfg, filename, local_folder):
    create_folder(local_folder)
    sftp.get(os.path.join(cluster_result_folder, filename), os.path.join(local_folder, filename)) 
    """
    try:
        sftp.get(os.path.join(cluster_result_folder, filename), os.path.join(local_folder, filename)) 
        return True
    except IOError:
        return False
    """

def get_missing_params(params,cfg, 
                             load_oinfo = False, 
                             load_train = False, 
                             load_model_step = None, 
                             norms = False, 
                             overwrite = False):
    missing_params = []
   
    for i, param in enumerate(params):
        
        new_cfg = update_cfg_from_dict(param, deepcopy(cfg))
        initialize_hydra(new_cfg)
       
        if load_oinfo:
            filename_oinfo, oinfo_folder = get_oinfo_filename(new_cfg)
   
            oinfo_file = load_oinfo_files(new_cfg, filename_oinfo)
            
            if oinfo_file is None or overwrite:
                missing_params.append(param)
            elif isinstance(oinfo_file, pd.DataFrame):
                max_step = max(oinfo_file.step)
                if max_step != cfg.training.steps:
                    missing_params.append(param)
            else:
                del oinfo_file
           
        if load_train:
            filename_best_results, filename_global_results = get_train_filenames(cfg = new_cfg, 
                                                                                load_model_step = load_model_step, 
                                                                                norms = norms
                                                                                )
           
            logger_global_results = Logger(new_cfg.exp_folder, filename_global_results)
            logger_best_results = Logger(new_cfg.exp_folder, filename_best_results)
            
            if not logger_global_results.exists() or not logger_global_results.exists():
                missing_params.append(param)
  
    return missing_params

def load_results_from_params(params, cfg, 
                             load_oinfo = False, 
                             load_train = False, 
                             load_model_step = None, 
                             norms = False, 
                             max_rows = None,
                             overwrite = False,
                             load_global_oinfo = False,
                             load_global_train = True):

    total_results = defaultdict(list)
    
    for i, param in enumerate(params):
        
        new_cfg = update_cfg_from_dict(param, deepcopy(cfg))
        initialize_hydra(new_cfg)
 
        if load_oinfo:
            filename_oinfo, oinfo_files_exist_on_disk, oinfo_folder = get_oinfo_filename(new_cfg)
            if oinfo_files_exist_on_disk and not overwrite:
                load_oinfo_files(new_cfg, filename_oinfo, total_results, load_global=load_global_oinfo)
       
        if load_train:
            filename_best_results, filename_global_results = get_train_filenames(cfg = new_cfg, 
                                                                                load_model_step = load_model_step, 
                                                                                norms = norms
                                                                                )

            logger_global_results = Logger(new_cfg.exp_folder, filename_global_results)
            logger_best_results = Logger(new_cfg.exp_folder, filename_best_results)
        
            if logger_global_results.exists() and load_global_train:
                df_global_train = logger_global_results.load()
                print(df_global_train)
                
                if isinstance(df_global_train, pd.core.frame.DataFrame):
                    df_global_train_acc = convert_column_to_row(df_global_train, "train_acc")
                    df_global_test_acc = convert_column_to_row(df_global_train, "test_acc")
                    df_global_train_loss = convert_column_to_row(df_global_train, "train_loss")
                    df_global_test_loss = convert_column_to_row(df_global_train, "test_loss")
                    df_train_global = pd.concat([df_global_train_acc, df_global_test_acc, df_global_train_loss, df_global_test_loss])
                    df_train_global = df_train_global.drop(columns=["test_acc", "train_acc", "train_loss", "test_loss"])
                    
                    total_results["global_train"].append(df_global_train)
            
            if logger_best_results.exists():
                df_best_train = logger_best_results.load() 
                
                if isinstance(df_best_train, pd.core.frame.DataFrame):
                    total_results["best_train"].append(df_best_train)

            if logger_global_results.exists() and logger_global_results.exists():
                from tasks.train import load_training_files
                load_training_files(logger_global_results, logger_best_results, total_results, load_global_train)
    
        if max_rows is not None and max_rows == i:
            break
  
    for key, value in total_results.items():
        total_results[key] = pd.concat(value)
   
    return DotMap(total_results)

def download_missing_files_from_cluster(params, cfg, 
                             load_oinfo = False, 
                             load_train = False, 
                             load_model_step = None, 
                             norms = False):

    from hidden_cluster_code import set_login_cluster
    sftp, ssh, cluster_root_folder = set_login_cluster(cfg)
    missing_params = []
    for i, param in enumerate(params):
        new_cfg = update_cfg_from_dict(param, deepcopy(cfg))
        initialize_hydra(new_cfg)

        if load_oinfo:
            filename_oinfo, oinfo_folder = get_oinfo_filename(new_cfg)
            cluster_result_folder = os.path.join(cluster_root_folder, new_cfg.exp_folder.split('/results/')[-1], 'oinfo')
            local_folder = os.path.join(new_cfg.exp_folder, 'oinfo')
            
            downloaded = download_file_from_cluster(sftp, cluster_result_folder, new_cfg, filename_oinfo, local_folder)

        if load_train:
            filename_best_results, filename_global_results = get_train_filenames(cfg = new_cfg, 
                                                                                load_model_step = load_model_step, 
                                                                                norms = norms
                                                                                )
            cluster_result_folder = os.path.join(cluster_root_folder, cfg.exp_folder.split('/results/')[-1])
            local_folder = os.path.join(cfg.exp_folder)
            create_folder(local_folder)
            downloaded = download_file_from_cluster(sftp, cluster_result_folder, cfg, filename_best_results + '.pkl', local_folder)
            downloaded = download_file_from_cluster(sftp, cluster_result_folder, cfg, filename_global_results + '.pkl', local_folder)

        if not downloaded:
            missing_params.append(param)
    return missing_params
    

def group_dataframe_by_columns(df, columns):
        agg_funcs = {col: lambda x: list(x) for col in columns}
        grouped_df = df.groupby([c for c in df.columns if c not in columns]).agg(agg_funcs).reset_index()
        return grouped_df


from utils.config import initialize_hydra, update_cfg_from_dict, update_model_name_and_folder

#from tasks.oinfo import load_oinfo_run
#from tasks.train import load_train_run



def convert_dataframe_to_seaborn_format(df, metric_names, argument_columns):
    df_total = []
    argument_columns = [c for c in argument_columns if c in df.columns]
    for metric_name in metric_names:
        df_tmp = deepcopy(df[argument_columns])
        df_tmp["step"] = df["step"]
        df_tmp["metric_value"] = df[metric_name]
        name = metric_name
        df_tmp["metric_name"] = [name for _ in range(len(df))]
        df_total.append(pd.DataFrame.from_dict(df_tmp))
    return pd.concat(df_total)
            

    
        

            


def load_data(df_params, group_columns, cluster_root_folder, sftp, cfg, load_train = False, load_oinfo = False):
    
    run_results = defaultdict(list)
    params_grouped = group_dataframe_by_columns(df_params, group_columns)
    n_runs = len(params_grouped[group_columns[0]].item())
    
    for index, row in df_params.iterrows():
        arguments = {column: row[column] for column in df_params.columns}
            
        for i in range(n_runs):
            for c in group_columns:
                arguments[c] = df_params[c][i]
            
            new_cfg = update_cfg_from_dict(arguments, deepcopy(cfg))
            initialize_hydra(new_cfg)

            update_model_name_and_folder(new_cfg, set_folder = True)
        
            if load_train:
                df_global, df_best = load_train_run(new_cfg, cluster_root_folder, sftp)
              
                run_results["train_global"].append(df_global)
                run_results["train_best"].append(df_best)
              
            if load_oinfo:
                df_oinfo = load_oinfo_run(new_cfg, cluster_root_folder, sftp)
                run_results["oinfo"].append(df_oinfo)
            
    for key, value in run_results.items():
        run_results[key] = pd.concat(value)
    
    return run_results

            
        
    


