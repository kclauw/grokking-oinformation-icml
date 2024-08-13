import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy

import os

from models.utils import load_model, evaluate_model, get_model_stats
from models.configuration import optimizer_dict, loss_function_dict
#from oinformation.oinfo_jax_original import OinfoOriginal
from datasets import load_data
import utils
from oinformation.oinfo_greedy import GreedyOinfo

def load_oinfo_run(cfg, cluster_root_folder, sftp, overwrite = False):
    #Downloads file if not exists
    filename_oinfo, oinfo_folder = get_oinfo_filename(cfg)
   
    df_oinfo = None
    
    if utils.file_exists(oinfo_folder, filename_oinfo):
        df_oinfo = load_oinfo_files(cfg, filename_oinfo)
    
   
    if sftp is not None and df_oinfo is None or overwrite:
        cluster_result_folder = os.path.join(cluster_root_folder, cfg.exp_folder.split('/results/')[-1], 'oinfo')
        local_folder = os.path.join(cfg.exp_folder, 'oinfo')     
        utils.download_file_from_cluster(sftp, cluster_result_folder, cfg, filename_oinfo, local_folder)
        df_oinfo = load_oinfo_files(cfg, filename_oinfo)
  
    return df_oinfo
            
                    
def load_oinfo_files(cfg, filename_oinfo, total_results = None):
    
    if utils.file_exists(os.path.join(cfg.exp_folder, 'oinfo'), filename_oinfo):
        df_oinfo_global = utils.load_dataframe_from_pickle(os.path.join(cfg.exp_folder, 'oinfo'), filename_oinfo)
        if total_results is not None:
            if isinstance(df_oinfo_global, pd.core.frame.DataFrame):
                total_results["global_oinfo"].append(df_oinfo_global)
        return df_oinfo_global
    else:
        return None
    
def get_oinfo_filename(cfg, step = None):
    
    oinfo_folder = os.path.join(cfg.exp_folder, 'oinfo')
    oinfo_params = "%s_%s" % (cfg.oinfo.search_mode, cfg.oinfo.features)
    filename_oinfo = 'oinfo_%s_' % (oinfo_params) + cfg.oinfo.data
    
    #filename_best = 'best_oinfo_%s_' % (oinfo_params) + "_" + cfg.oinfo.data
    if step:
        filename_oinfo += "_step_%d" % (step)
        #filename_best += "_step_%d" % (step)
   
    if cfg.oinfo.search_mode == "clustering":
        
        filename_oinfo += "_clustered_%d" % (cfg.oinfo.cluster_size)
        
        if cfg.clustering.normalize:
            filename_oinfo += "_normalize"
       
        if cfg.clustering.linkage_method:
            filename_oinfo += "_%s" % (cfg.clustering.linkage_method)
        
        if cfg.clustering.metric:
            filename_oinfo += "_%s" % (cfg.clustering.metric)
   
    return filename_oinfo, oinfo_folder

def exhaustive_loop_zerolag_normal(x, cfg, minsize=3, maxsize=4, verbose=None):
    from oinformation.hoi.metrics.oinfo import Oinfo
    model = Oinfo(x, verbose=verbose)
   
    return model.fit_exhaustive(minsize=minsize, maxsize=maxsize, method="gcmi")

def estimate_oinformation(train_acc, test_acc, train_loss, test_loss, features, cfg, data_folder, i, overwrite = False):
  
    df_total_global, df_total_best = [], []

    x = features[cfg.oinfo.features].astype(np.float64)
    
    if cfg.oinfo.search_mode == "greedy":
        oinfo = GreedyOinfo(x, verbose=True)
                    
        df_syn, df_red = oinfo.fit(minsize=2, maxsize_exhaustive=cfg.oinfo.max_exhaustive, maxsize_greedy=cfg.oinfo.max_greedy)
    
    elif cfg.oinfo.search_mode == "clustering":
        if cfg.clustering.cluster_method == "hierarchical":
            from clustering.hierarchical import hierarchical_clustering
            from oinformation.oinfo_jax_original import OinfoOriginal
            if cfg.clustering.normalize:
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                scaler = StandardScaler()
                #scaler = MinMaxScaler()
                tmp_x = deepcopy(x)
                scaler.fit(tmp_x)
                clustered_labels_to_neurons = hierarchical_clustering(tmp_x, cfg.clustering.num_clusters, 
                                    linkage_method = cfg.clustering.linkage_method, 
                                    metric = cfg.clustering.metric, verbose=True, normalize = cfg.clustering.normalize)
            else:
                clustered_labels_to_neurons = hierarchical_clustering(x, cfg.clustering.num_clusters, 
                                        linkage_method = cfg.clustering.linkage_method, 
                                        metric = cfg.clustering.metric, verbose=True, normalize = cfg.clustering.normalize)
            
            model = OinfoOriginal(x, clustered_labels_to_neurons = clustered_labels_to_neurons, verbose=True)
            df, df_syn, df_red = model.fit_exhaustive(minsize=2, maxsize=cfg.oinfo.max_clusters, method=cfg.oinfo.normalization, n_clusters = cfg.clustering.num_clusters, batch_size = cfg.oinfo.batch_size)
    else:
        df, df_syn, df_red = exhaustive_loop_zerolag_normal(x=x, cfg = cfg, maxsize=cfg.oinfo.max_clusters)
    
    
    utils.add_cfg_arguments_to_dataframe(cfg, df_syn, i, cfg.oinfo.features, train_acc, test_acc, train_loss, test_loss)
    utils.add_cfg_arguments_to_dataframe(cfg, df_red, i, cfg.oinfo.features, train_acc, test_acc, train_loss, test_loss)
    
    df_total_global.append(pd.concat([df_syn, df_red]))
     
    df_total_global = pd.concat(df_total_global)

    return df_total_global

def run_oinfo(cfg):
    
    if cfg.neptune.enabled:
        neptune_logger = utils.load_neptune_logger(cfg, tags = ["OInfo"], log_oinfo_params = True)
   
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7" 
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" 
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    utils.initialize_hydra(cfg)
    utils.seed_everything(cfg.training.seed)
    
    df_total_global = []
    
    train_loader, train_loader_for_eval, test_loader = load_data(cfg, cfg.device)
    filename_oinfo, oinfo_folder = get_oinfo_filename(cfg)
    
    loss_fn = loss_function_dict[cfg.model.parameters.loss]
    model = load_model(cfg, cfg.device)
    
    if utils.file_exists(oinfo_folder, filename_oinfo):
        df_oinfo = utils.load_dataframe_from_pickle(oinfo_folder, filename_oinfo)
        #df_syn = utils.filter_dataframe(df_oinfo, metric_name = "synergy", step = 0)
        #df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
       
        steps = [i for i in range(0, cfg.training.steps, cfg.oinfo.evaluation_step) if i not in list(df_oinfo.step.unique())]
       
        df_total_global.append(df_oinfo)
    else:
        steps = [i for i in range(0, cfg.training.steps, cfg.oinfo.evaluation_step)]
    
    for i in steps:
        model.load_state_dict(torch.load(os.path.join(cfg.exp_folder, 'checkpoints', 'model_%d.pt' % (i)), map_location=torch.device('cpu')))
        with torch.no_grad():
            model.eval()
            
            train_loss, train_acc, train_margin, train_features = evaluate_model(cfg.model.name, train_loader_for_eval, model, loss_fn, cfg)
            test_loss, test_acc, test_margin, test_features = evaluate_model(cfg.model.name, test_loader, model, loss_fn, cfg)
            
            total_features = train_features
            
            
            print("oinfo step %d train acc %f test acc %f" % (i, train_acc, test_acc))
            
            #print()
            
            #x = train_features[cfg.oinfo.features].astype(np.float64)
            #oinfo = GreedyOinfo(x, verbose=True)
            #df_syn, df_red = oinfo.fit(minsize=2, maxsize_exhaustive=cfg.oinfo.max_exhaustive, maxsize_greedy=cfg.oinfo.max_greedy)
         
            df_oinfo_global = estimate_oinformation(train_acc,  test_acc, train_loss, test_loss, total_features, cfg, os.path.join(cfg.exp_folder, 'oinfo'), i, overwrite = True)
            
            df_syn = utils.filter_dataframe(df_oinfo_global, metric_name = "synergy")
            df_red = utils.filter_dataframe(df_oinfo_global, metric_name = "redundancy")
            
            df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
            df_red_best = df_red.loc[df_red.groupby(['step', 'seed'])['metric_value'].idxmax()]
           

            if cfg.neptune.enabled:
                neptune_logger["train_acc"].append(value=train_acc,step=i)
                neptune_logger["test_acc"].append(value=test_acc,step=i)
                neptune_logger["train_loss"].append(value=train_loss,step=i)
                neptune_logger["test_loss"].append(value=test_loss,step=i)
                
                neptune_logger["synergy"].append(value=df_syn_best["metric_value"].item(),step=i)
                #neptune_logger["synergy_multiplets"].append(value=df_syn_best["multiplet"].item(),step=i)
                neptune_logger["synergy_size"].append(value=df_syn_best["size"].item(),step=i)
                
                neptune_logger["redundancy"].append(value=df_red_best["metric_value"].item(),step=i)
                #neptune_logger["redundancy_multiplets"].append(value=df_red_best["multiplet"].item(),step=i)
                neptune_logger["redundancy_size"].append(value=df_red_best["size"].item(),step=i)
            
            
            #neptune_logger["global_step"].append(value=step,step=step)
            
            df_total_global.append(df_oinfo_global)
            utils.save_dataframe_to_pickle(oinfo_folder, filename_oinfo, pd.concat(df_total_global))
            
 
    
    