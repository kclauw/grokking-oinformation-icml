import hydra 
from tqdm import tqdm

import os 
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7" 
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 

from utils.config import initialize_hydra, get_params_from_cfg, update_cfg_from_dict, update_model_name_and_folder
from utils.data import load_results_from_params
import utils

def run_experiment(cfg, params = None):
    
    if params:
        cfg = update_cfg_from_dict(params, cfg)
        update_model_name_and_folder(cfg)
        initialize_hydra(cfg)
    
    if cfg.train_enabled:
        from tasks.train import train
        train(cfg, save_model = cfg.save_model)
    
    if cfg.oinfo_enabled:
        from tasks.oinfo import run_oinfo
        run_oinfo(cfg)
    
    if cfg.lth_enabled:
        from tasks.lth import lth
        lth(cfg)
 
        
def run_distributed_experiments(cfg, params):
    from multiprocess import Pool, set_start_method
    set_start_method("spawn")
    
    total_params = len(params)
    run_count = 0
    
    def run_and_count(param):
        nonlocal run_count
        run_count += 1
        print("Number of runs:", run_count, "/", total_params)
        run_experiment(cfg, param)
        
    with Pool(cfg.n_jobs_pool) as p:
        #p.map(lambda param: run_experiment(cfg, param), params)
        list(tqdm(p.imap_unordered(run_and_count, params), total=total_params))
        
@hydra.main(version_base=None, config_name="main", config_path="../config")
def main(cfg) -> None:
 
    initialize_hydra(cfg)
    
    params = get_params_from_cfg(cfg, ignore_lth=cfg.lth_enabled, ignore_oinfo=cfg.oinfo_enabled)
    print("total %d" % len(params))
    
    total_params = len(params)
    
  
    #Get missing params
    if cfg.ignore_existing_files:
        params = utils.get_missing_params(params,cfg, 
                                load_oinfo = cfg.oinfo_enabled, 
                                load_train = cfg.train_enabled)
   
    print("missing %d" % len(params))
    
    """
    params = utils.download_missing_files_from_cluster(params,cfg, 
                             load_oinfo = cfg.oinfo_enabled, 
                             load_train = cfg.train_enabled)
    """
    if params is not None:
        if not cfg.gridsearch_enabled:
            print("%d / %d" % (len(params), total_params))
            from tqdm import tqdm as tqdm2
            for param in tqdm2(params):
                run_experiment(cfg, params=param)
        else:
            run_distributed_experiments(cfg, params)
        
if __name__ == "__main__":
    main()