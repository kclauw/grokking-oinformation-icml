
import utils
from tasks.oinfo import get_oinfo_filename, load_oinfo_files
from tasks.train import train


def get_best_oinfo(cfg):
    filename_oinfo, oinfo_folder = get_oinfo_filename(cfg)
  
    df_oinfo_global = load_oinfo_files(cfg, filename_oinfo)
    df_oinfo_global = df_oinfo_global[df_oinfo_global['step'] != 0]
    
    df_syn = utils.filter_dataframe(df_oinfo_global, metric_name = "synergy")
    df_red = utils.filter_dataframe(df_oinfo_global, metric_name = "redundancy")
    df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
    df_red_best = df_red.loc[df_red.groupby(['step', 'seed'])['metric_value'].idxmax()]
    return df_syn_best, df_red_best

def lth(cfg, multiplets = None, load_model_step = None, norms = False, multiplet_name = None, save_model = False):
    #Load synergy and redundancy file
    filename_oinfo, oinfo_folder = get_oinfo_filename(cfg)
    
    
    df_oinfo_global = load_oinfo_files(cfg, filename_oinfo)
    #df_oinfo_global = df_oinfo_global[df_oinfo_global['step'] != 0]
   
    df_syn = utils.filter_dataframe(df_oinfo_global, metric_name = "synergy")
   
    df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
    
    best_syn = df_syn_best.loc[df_syn_best['metric_value'].idxmin()]
  
    if cfg.oinfo.search_mode == "greedy":
        best_syn_multiplets = best_syn["multiplet"]
    else:
        best_syn_multiplets = best_syn["multiplets"]
        
    
    #df_red = utils.filter_dataframe(df_oinfo_global, metric_name = "redundancy")
    #df_red_best = df_red.loc[df_red.groupby(['step', 'seed'])['metric_value'].idxmax()]
    
    
    

    
    
  

    
    feature_name = 'model.%s.weight' % (cfg.lth.layer.split('_')[0])
    
    if cfg.lth.metric == "best_synergy":
        multiplets = {
            feature_name : best_syn_multiplets
        }
        step = best_syn["step"]
    elif cfg.lth.metric == "best_synergy_inverse":
        multiplets = {
            feature_name : [i for i in range(cfg.model.parameters.layers.f1.width) if i not in best_syn_multiplets]
        }
        step = best_syn["step"]
    elif cfg.lth.metric == "generalizing":
        df_syn_best = df_syn_best[df_syn_best['step'] != 0]
        generalizing_step = df_syn_best.loc[df_syn_best[df_syn_best['test_acc'] > 0.98]["step"].idxmin()]["step"]

        
        best_syn_generalizing = df_syn_best.loc[df_syn_best['step'] == generalizing_step]
    
        best_syn_generalizing_multiplets = best_syn_generalizing["multiplets"]
        
        multiplets = {
            feature_name : best_syn_generalizing_multiplets
        }
        step = generalizing_step
    elif cfg.lth.metric == "generalizing_inverse":
        df_syn_best = df_syn_best[df_syn_best['step'] != 0]
        generalizing_step = df_syn_best.loc[df_syn_best[df_syn_best['test_acc'] > 0.98]["step"].idxmin()]["step"]

        best_syn_generalizing = df_syn_best.loc[df_syn_best['step'] == generalizing_step]
    
        best_syn_generalizing_multiplets = best_syn_generalizing["multiplets"]
        
        
        multiplets = {
            feature_name : [i for i in range(cfg.model.parameters.layers.f1.width) if i not in best_syn_generalizing_multiplets]
        }
        step = generalizing_step
    
    elif "synergy_step" in cfg.lth.metric:
        step = int(cfg.lth.metric.split('_')[-1])
        df_syn = utils.filter_dataframe(df_oinfo_global, metric_name = "synergy", step = step)
        df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
        
        if cfg.oinfo.search_mode == "greedy":
            multiplets = df_syn_best["multiplet"].item()
        else:
            multiplets = df_syn_best["multiplets"].item()

        multiplets = {
            feature_name : multiplets
        }
        
        step = step
    
    elif "inverse_synergy_step" in cfg.lth.metric:
        step = int(cfg.lth.metric.split('_')[-1])
        df_syn = utils.filter_dataframe(df_oinfo_global, metric_name = "synergy", step = step)
        df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
        multiplets = df_syn_best["multiplets"]
            
        if cfg.oinfo.search_mode == "greedy":
            multiplets = df_syn_best["multiplet"]
        else:
            multiplets = df_syn_best["multiplets"]
        
       
        multiplets = {
            feature_name : [i for i in range(cfg.model.parameters.layers.f1.width) if i not in multiplets]
        }
          
        step = step
  
    #load_model_step = best_syn["step"]   
    
   
    train(cfg, oinfo_name= "lth_" + filename_oinfo, multiplets = multiplets, norms = False, multiplet_name = cfg.lth.metric, metric_step = step)