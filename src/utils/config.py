from hydra.core.hydra_config import HydraConfig
from collections import defaultdict
from omegaconf import OmegaConf
from itertools import product
from copy import deepcopy 
import numpy as np
import os

#from utils.data import read_from_json
from utils.misc import create_folder, read_from_json, dict_to_hash

def load_neptune_logger(cfg, 
                        project = "Emergence-OInformation-Grokking/Grokking-Oinformation", 
                        api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Yzc3NjFhNS0wYThjLTQxNTQtODQ3YS1hMmZlYjhhYzZkMjcifQ=="
        ,tags = [], log_oinfo_params = False):
    if cfg.neptune.enabled:
        
        selected_params = ["model.parameters", "dataset.parameters", "training", "optimizer.name"]
        if log_oinfo_params:
            selected_params.extend(["oinfo", "clustering"])
            
        params = cfg_to_dict(deepcopy(cfg), ["model.parameters", "dataset.parameters", "training", "optimizer.name"])
        params["model"]["unique_name"] = cfg.model.unique_name
        params["dataset"]["unique_name"] = cfg.dataset.unique_name
        params["parameters"] = cfg.model_parameters
        
        
        import neptune
        neptune_logger = neptune.init_run(
            project=project,
            api_token=api_token,
            tags = tags
        )
      
        neptune_logger["parameters"] = params
    return neptune_logger

def cfg_to_dict(cfg, selected_keys=[]):
    """
    Converts an OmegaConf configuration object to a dictionary and selects specific keys.

    Parameters:
    - cfg (omegaconf.DictConfig): The OmegaConf configuration object.
    - selected_keys (list of str): A list of dotted string paths to the desired nested configurations (e.g., ["model.parameters"]).

    Returns:
    - dict: A nested dictionary containing only the specified paths and their values.
    """
    result = {}
    
    for path in selected_keys:
        # Use OmegaConf.select to get the value at the given path
        value = OmegaConf.select(cfg, path)
       
        # Split the path into keys
        keys = path.split('.')
        

        # Set the value in the result dictionary at the nested path
        sub_dict = result
        for key in keys[:-1]:
            if key not in sub_dict:
                sub_dict[key] = {}
            sub_dict = sub_dict[key]
        
        if isinstance(value, str):
            sub_dict[keys[-1]] = value
        else:
            sub_dict[keys[-1]] = OmegaConf.to_container(value, resolve=True)
  
    return result

def get_experiment_name():
    return OmegaConf.to_container(HydraConfig.get().runtime.choices)["experiment"]

def create_name_from_arguments(**kwargs) -> str:
    formatted_params = [
        f'{key}_{str(value).rstrip("0").rstrip(".")}' if '.' in str(value) else f'{key}_{value}'
        for key, value in kwargs.items()
    ]
    
    return '_'.join(formatted_params)

"""
LOAD CFG
"""
def get_params_from_cfg(cfg, ignore_oinfo = False, ignore_lth = False):
    if cfg.gridsearch_job_id != None:
        params = read_from_json(os.path.join('./jobs', get_experiment_name()), 'job_%d' % (int(cfg.gridsearch_job_id)))
    else:
        if cfg.gridsearch_job_id:
            return read_from_json(os.path.join('./jobs', get_experiment_name()), 'job_%d%s%s' % (cfg.gridsearch_job_id, "_train" if cfg.train_enabled else "", "_oinfo" if cfg.oinfo_enabled else ""))
        else:
            return process_gridsearch_arguments(cfg, train_enabled=cfg.train_enabled, oinfo_enabled=ignore_oinfo, lth_enabled=ignore_lth)
    return params

def generate_parameter_list_linspace(min, max, total, scale = 1, add = None):
    parameters = [round(item, 5)  for item in np.linspace(min,max,num=total)]
    if add:
        parameters += [60000]
    return np.array(parameters)

def generate_parameter_list_fixed_increment(min, increment, total):
    return [round(min + (i*increment), 5) for i in range(total)]

def load_parameters_from_cfg(cfg):
    
    if "increment" in cfg.keys():
        return generate_parameter_list_fixed_increment(**cfg)
    elif "values" in cfg.keys():
        return cfg["values"]
    else:
        return generate_parameter_list_linspace(**cfg)
    
def process_gridsearch_arguments(cfg, filter = None, train_enabled = False, oinfo_enabled = False, lth_enabled = False, ignore_keys = None):
    #Create lists of all combinations for the parameters
    parameters = {}
    for key, value in cfg.gridsearch.parameters.items():
        
        if key == "dataset":
            parameter_name = next(iter(value))
            parameters["dataset.%s" % (parameter_name)] = param
        else:
            param = load_parameters_from_cfg(value)
    
            if len(param) == 0:
                param = np.round(param, 3)
            else:
                param = list(param)
            parameters[key] = param
    
 
    if not oinfo_enabled:
        parameters = {key: parameters[key] for key in parameters if 'oinfo' not in key}
        
    if "oinfo.search_mode" in cfg.gridsearch.parameters.keys():
        parameters["oinfo.search_mode"] = cfg.gridsearch.parameters["oinfo.search_mode"]["values"]
    
   
    if not lth_enabled:
        parameters = {key: parameters[key] for key in parameters if 'lth' not in key}
    
    if filter is not None:
        parameters = {k: v for k, v in parameters.items() if k in filter}
    
    if ignore_keys is not None:
        parameters = {k: v for k, v in parameters.items() if k not in ignore_keys}
       

    print("--- gridsearch parameters ---")
    for key, value in parameters.items():
        print((key, value))

    
    print("------------------")
    
    # Create combination of all parameters
    grid_search_results = [
    {
        param_name: int(param_value) if isinstance(param_value, np.int64) else 
                    np.round(float(param_value), 5) if isinstance(param_value, np.float64) else 
                    param_value 
        for param_name, param_value in zip(parameters.keys(), combo)
    }
    | {'train_enabled': train_enabled, 'oinfo_enabled': oinfo_enabled}
    for combo in product(*parameters.values())
    ]
    
  
    return grid_search_results



"""
Update hydra 
"""

def update_cfg_from_dict(params, cfg):
    cfg = deepcopy(cfg)

    for key, value in params.items():
        #set_cfg_item(cfg, key, value)
        if isinstance(value, np.float64):
            value = float(value)
        elif isinstance(value, np.int64):
            value = int(value)
        OmegaConf.update(cfg, key, value)
        
  
    if "modular" in cfg.dataset.name and not cfg.dataset.parameters.fixed_seed:
        cfg.dataset.parameters.seed = cfg.seed
    
    cfg.training.weight_decay = float(cfg.training.weight_decay)
    cfg.training.learning_rate = float(cfg.training.learning_rate)
    return cfg

def initialize_hydra(cfg):
    cfg.model_parameters = create_name_from_arguments(lr = cfg.training.learning_rate,
                                            wd = cfg.training.weight_decay,
                                            alpha = cfg.training.alpha,
                                            bs = cfg.training.train_batch_size)

    #cfg.experiment_name = get_experiment_name()
    
    update_model_name_and_folder(cfg)
    cfg.unique_training_name = "%s/%s/%s" % (cfg.dataset.unique_name, cfg.model.unique_name, cfg.model_parameters)

def update_model_name(cfg):
    architecture = ""
    
    if "layers" in cfg.model.parameters:
        for layer in cfg.model.parameters.layers.values():
            architecture += "_%d" % (layer.width) if architecture != "" else "%d" % (layer.width)
            architecture += "_%d" % (layer.dropout) if layer.dropout != 0 else ""
        
        cfg.model.unique_name = f'{cfg.model.name}_{architecture}_{cfg.model.parameters.activation}_init_{cfg.model.parameters.initialization}{"_zero" if cfg.model.parameters.set_last_layer_to_zero else "_nozero"}_alpha_{cfg.model.parameters.alpha}_{cfg.loss}'
    elif "encoder_decoder" in cfg.model.name:
        #for layer in cfg.model.parameters.layers.values():
        cfg.model.unique_name  = "encoder"
       
        for layer in cfg.model.parameters.encoder.values():
            cfg.model.unique_name += "_" + str(layer.width)

        cfg.model.unique_name  += "_decoder"
        for layer in cfg.model.parameters.decoder.values():
            cfg.model.unique_name += "_" + str(layer.width)
            
        #cfg.model.unique_name += 
    
    if cfg.training.rescale_weight_norm:
        cfg.model.unique_name += "_rescale"
    
    if not "transformer" in cfg.model.name and cfg.set_last_layer_zero_init:
        cfg.model.unique_name += "_zero_init"

def update_model_parameters(cfg):
    cfg.model_parameters = create_name_from_arguments(lr = cfg.learning_rate,
                                            wd = cfg.weight_decay,
                                            alpha = cfg.training.alpha,
                                            bs = cfg.train_batch_size)
    

def create_name_model(cfg):
    model_name = None
    architecture = ""
    
    if "layers" in cfg.model.parameters:
    
        for layer_name, layer in cfg.model.parameters.layers.items():
            
          
            if "output" in layer_name:
                architecture += "_out"
                
            if "width" in layer.keys():
                architecture += "_%d" % (layer.width) if architecture != "" else "%d" % (layer.width)
         
            dropout_value = layer["dropout"]["value"]
            
            
            if dropout_value != 0:
                architecture += "%s_%f" % (layer["dropout"]["strategy"], layer["dropout"]["value"])
            if layer.norm is not None:
                architecture += "_%s" % (layer.norm)
       
            architecture += "_bias" if layer["bias"] else "_nobias"
        
        model_name = f'{cfg.model.name}_{architecture}_{cfg.model.parameters.activation}_init_{cfg.model.parameters.initialization}{"_zero" if cfg.model.parameters.set_last_layer_to_zero else "_nozero"}_alpha_{cfg.model.parameters.alpha}_{cfg.model.parameters.loss}'
      
    elif "encoder_decoder" in cfg.model.name:
        #for layer in cfg.model.parameters.layers.values():
        cfg.model.unique_name  = "encoder"
       
        for layer in cfg.model.parameters.encoder.values():
            cfg.model.unique_name += "_" + str(layer.width)

        cfg.model.unique_name  += "_decoder"
        for layer in cfg.model.parameters.decoder.values():
            cfg.model.unique_name += "_" + str(layer.width)
            
        #cfg.model.unique_name += 
    
    if cfg.training.rescale_weight_norm:
        model_name += "_rescale"
    
    if not "transformer" in cfg.model.name and cfg.set_last_layer_zero_init:
        model_name += "_zero_init"
    
    return model_name

def create_name_dataset(cfg):
    cfg_data = cfg.dataset.parameters
    if "modular" in cfg.dataset.name:
        name = cfg.dataset.name
        
        name += "_" + create_name_from_arguments(p = cfg_data.p,
                                   seed = cfg_data.seed,
                                   frac = cfg_data.frac,
                                   noise = cfg_data.noise_level)
        if cfg_data.fixed_seed:
            name += "_fixed_seed"
    return name

    

def update_model_name_and_folder(cfg, set_folder = True):
    
    name_model = create_name_model(cfg)
    name_dataset = create_name_dataset(cfg)
    
    cfg.model.unique_name = name_model
    cfg.dataset.unique_name = name_dataset
    
    
    if set_folder:
        
        if cfg.root_folder_type == "google_drive":
            root_folder = "/content/gdrive/MyDrive/"
        elif cfg.root_folder_type == "cluster":
            root_folder = cfg.cluster_folder
        else:
            root_folder = os.getcwd()
            
        cfg.exp_folder = os.path.join(root_folder, 'results', cfg.dataset.unique_name, cfg.optimizer.name, cfg.model.unique_name, cfg.model_parameters, "seed_%d" % (cfg.training.seed))
        
        
        
        
        """
        unique_params = {"dataset" : cfg.dataset.parameters,
                         "optimizer" : cfg.optimizer.name,
                         "model" : cfg.model.parameters,
                         "training" : cfg.training,
                         "regularization" : cfg.regularization}
        """
        #hash_model = dict_to_hash(unique_params)
        
        
        create_folder(cfg.exp_folder)
        create_folder(os.path.join(cfg.exp_folder, 'checkpoints'))
    
def add_cfg_arguments_to_dictionary(cfg, dictionary, ignore_oinfo_keys = False, ignore_lth_keys = False, ignore_keys = []):
    dictionary["dataset"].append(cfg.dataset.unique_name)
    dictionary["model"].append(cfg.model.name)  
    dictionary["parameters"].append(cfg.model_parameters)  

    if not ignore_oinfo_keys:
        for key, value in cfg.oinfo.items():
            if key not in ignore_keys:
                dictionary[key].append(OmegaConf.select(cfg, key))
    
    grid_search_parameters = list(cfg.gridsearch.parameters.keys())
    if ignore_oinfo_keys:
        grid_search_parameters = [k for k in grid_search_parameters if not "oinfo" in k]
    if ignore_lth_keys:
        grid_search_parameters = [k for k in grid_search_parameters if not "lth" in k]    
    
    
    for key in grid_search_parameters:
        if key not in ignore_keys:
            dictionary[key].append(OmegaConf.select(cfg, key))