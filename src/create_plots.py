import hydra 
import pandas as pd
from copy import deepcopy
from collections import defaultdict
import os

from utils.config import initialize_hydra, get_params_from_cfg, update_cfg_from_dict, update_model_name_and_folder, get_experiment_name, update_model_parameters
from utils.data import filter_dataframe, filter_parameters_in_dataframe, load_results_from_params
from utils.visualization import lineplot
from tasks.oinfo import load_oinfo_run
from tasks.train import load_train_run
import itertools
import utils

def extract_arguments_with_multiple_values(cfg):
    arguments = dict()
    for key, value in cfg.gridsearch.parameters.items():
        if len(value["values"]) > 1:
            arguments[key] = value
    # Extract the parameter names
    param_names = list(arguments.keys())
    print(arguments)
    
    # Generate all unique pairs of parameter names
    param_pairs = list(itertools.combinations(param_names, 2))

    # Generate the required tuples
    result_tuples = []
    for pair in param_pairs:
        remaining_params = [param for param in param_names if param not in pair]
        result_tuples.append((pair[0], pair[1], remaining_params))
        
    return result_tuples

def training_heatmaps(cfg, params, logscale = False):
    
    
    results = load_results_from_params(params, cfg, 
                            load_oinfo = cfg.oinfo_enabled, 
                            load_train = True,
                            load_model_step = None, 
                            norms = False, 
                            overwrite = cfg.overwrite,
                            load_global_oinfo=True,
                            load_global_train = False)
        
    columns = [k for k in list(cfg.gridsearch.parameters.keys()) if not "oinfo" in k and not "lth" in k and not "norm" in k]

    #print(df_global_training_results.metric_name.unique())
   
    #df_best_train = utils.convert_column_to_row(df_best_train, ["train_test_difference", "best_train_acc", "best_test_acc", "best_train_loss", "best_test_loss", "train_acc_step"])
    df_best_train = results.best_train
    
    
    arguments_multiplets = extract_arguments_with_multiple_values(cfg)
   
   
    for cfg_heatmap in arguments_multiplets:
        
        x, y, z = cfg_heatmap
       
        arguments = list(set([a for a in columns if a in df_best_train and a not in [x, y, "seed"]]))
        
        arguments.append("dataset")
        arguments.append("model")

        #df_sub = utils.filter_dataframe(df_best_train, **{x : cfg.gridsearch.parameters[x]["values"], y : cfg.gridsearch.parameters[y]["values"]})
        #print({x : cfg.gridsearch.parameters[x]["values"], y : cfg.gridsearch.parameters[y]["values"]})
       
        extra_params = utils.process_gridsearch_arguments(cfg = cfg, train_enabled = cfg.train_enabled, filter=z)
        

        z_values = ["train_test_difference", "test_acc_step", "train_acc_step", "best_train_acc", "best_test_acc", "best_train_loss", "best_test_loss"]
        df_best_train_run = df_best_train.groupby(arguments + [x, y], as_index=False)[z_values].mean()
       
        for argument_values in extra_params:
        #for argument_values, df_best_train_run in df_best_train.groupby(arguments):
           
         
            #df_best_train_run = df_best_train.groupby(arguments + [x, y], as_index=False)[z].mean()
            
            
            parameters = dict()
            
           
            folder_name = ""
            for key, value in argument_values.items():
                if key not in ["train_enabled", "oinfo_enabled"]:
                    parameters[key] = value
                    folder_name += "%s_%s" % (key, str(value))
  
          
            
            
        
            for z_value in ["train_test_difference", "test_acc_step", "train_acc_step", "best_train_acc", "best_test_acc", "best_train_loss", "best_test_loss"]:
                folder = os.path.join('./plots', get_experiment_name(), 'heatmap', folder_name)
               
                filename = "_".join([x, "vs", y, z_value])
               
                utils.heatmap(utils.filter_dataframe(df_best_train_run, **parameters), x = x, y = y, z = z_value, show = False, 
                            folder = folder, filename = filename, title = z_value + " " + folder_name, x_label=x.replace('_', ' '), y_label=y.replace('_', ' '))
          


def plot_train(param, cfg,  cluster_root_folder, sftp, logscale):
    
  
    new_cfg = update_cfg_from_dict(utils.filter_dict(param, ["seed"]), deepcopy(cfg))
    #update_model_parameters(new_cfg)
    #initialize_hydra(new_cfg)
    update_model_name_and_folder(new_cfg, set_folder = False)
    
    results = load_run(param, new_cfg, cluster_root_folder, sftp, load_train = True)
    
    df_global_train_run = results["train_global"]
    
    df_train_results = []
    for m in ["train_acc", "test_acc"]:
        df_train_results.append(convert_row_to_column(df_global_train_run, m))
    df_train_results = pd.concat(df_train_results)
    
    
    #df_global_train_run = convert_row_to_column(df_global_train_run, "train_acc")
    #df_global_train_run = convert_row_to_column(df_global_train_run, "test_acc")
    params = str(df_global_train_run.parameters.unique()[0])
   
    
    #filename = "_".join([str(new_cfg.model.unique_name), str(new_cfg.dataset.unique_name), "alpha_%f"% (cfg.alpha)])

    lineplot(filter_dataframe(df_train_results, metric_name = ["train_acc", "test_acc"]), x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'acc'), 
            filename = params, 
            #title = 'Seed %d' % (seed), 
            xlabel = "steps",
            logscale = logscale,
            show = False)


def create_oinfo_lineplots(df_params, cluster_root_folder, sftp, cfg, logscale, max_epochs = None, overwrite = False):
    
    group_columns = deepcopy(cfg.gridsearch.groupby)
    group_columns.append("training.seed")

    params_grouped = utils.group_dataframe_by_columns(df_params, group_columns)
  
    #Loop over grouper parameters
    for index, row in params_grouped.iterrows():
        global_arguments = {column: row[column] for column in df_params.columns if not "enabled" in column}
       
        #Loop over the runs in this group
        df_total_train_acc, df_total_train_loss = [], []
        df_total_syn_best, df_total_red_best = [], []
 
        filename = "_".join([key + "_" + "_".join([str(c) for c in list(set(value))]) for key, value in global_arguments.items() if key in group_columns])
     
        for argument_comb in zip(*[global_arguments[key] for key in group_columns]):
            arguments = deepcopy(global_arguments)
         
            for i, arg_value in enumerate(argument_comb):
                arguments[group_columns[i]] = arg_value
            
            
            folder_name = utils.create_name_from_arguments(lr = arguments["training.learning_rate"],
                                            wd =  arguments["training.weight_decay"])
            
            new_cfg = update_cfg_from_dict(arguments, deepcopy(cfg))
            initialize_hydra(new_cfg)
            update_model_name_and_folder(new_cfg, set_folder = True)
            
            
            df_oinfo = load_oinfo_run(new_cfg, cluster_root_folder, sftp, overwrite = overwrite)
            
            
            df_oinfo = df_oinfo[(df_oinfo['step'] != 0)]
            max_epochs = max(df_oinfo['step'].unique())
         
            print(max_epochs)
            if max_epochs is not None:
                df_oinfo = df_oinfo[df_oinfo['step'] < max_epochs]
               
                
            df_syn = utils.filter_dataframe(df_oinfo, metric_name = "synergy")
            df_red = utils.filter_dataframe(df_oinfo, metric_name = "redundancy")
            df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
            df_red_best = df_red.loc[df_red.groupby(['step', 'seed'])['metric_value'].idxmax()]
            
            #print(utils.filter_dataframe(df_syn, step = 100))
            
            
            df_global, df_best = load_train_run(new_cfg, cluster_root_folder, sftp, overwrite = False)

            #df_train_acc = utils.convert_dataframe_to_seaborn_format(df_global, ["test_acc"], arguments.keys())
            #df_train_loss = utils.convert_dataframe_to_seaborn_format(df_global, ["train_loss", "test_loss"], arguments.keys())
           
            name = ""
            for m in cfg.gridsearch.groupby:
                
                metric_name = m
                if "synergy" in m:
                    metric_name = "synergy"
                elif "redundancy" in m:
                    metric_name = "redundancy"

                name += " %s_%s" % (metric_name, arguments[m])
            
            df_syn_best['metric_name'] = df_syn_best['metric_name'] + [name for _ in range(len(df_syn_best))]
            df_red_best['metric_name'] = df_red_best['metric_name'] + [name for _ in range(len(df_red_best))]
            df_total_syn_best.append(df_syn_best)
            df_total_red_best.append(df_red_best)
        
            #df_train_acc['metric_name'] = df_train_acc['metric_name'] + [name for _ in range(len(df_train_acc))]
            #df_train_loss['metric_name'] = df_train_loss['metric_name'] + [name for _ in range(len(df_train_loss))]
          
            #df_total_train_acc.append(df_train_acc)
            #df_total_train_loss.append(df_train_loss)
            
            del df_global
            del df_best
        
        df_total_syn_best = pd.concat(df_total_syn_best)
        df_total_red_best = pd.concat(df_total_red_best)
        
        df_total_syn_best = utils.normalize_column(df_total_syn_best, remove_outliers = False, inverse_data = True, smooth = True, normalize = True)
        df_total_red_best = utils.normalize_column(df_total_red_best, remove_outliers = False, inverse_data = False, smooth = True, normalize = True)
          
          
        #df_total_train_acc = pd.concat(df_total_train_acc)
        #df_total_train_loss = pd.concat(df_total_train_loss)
        
        if max_epochs is not None:
            if max_epochs is not None:
                df_total_syn_best = df_total_syn_best[df_total_syn_best['step'] < max_epochs]
                df_total_red_best = df_total_red_best[df_total_red_best['step'] < max_epochs]
                #df_total_train_acc = df_total_train_acc[df_total_train_acc['step'] < max_steps[i]]
                #df_total_train_loss = df_total_train_loss[df_total_train_loss['step'] < max_steps[i]]
        
        df_total_syn_best["metric_name"] = [name.replace('_', ' ') for name in df_total_syn_best["metric_name"]]
        df_total_red_best["metric_name"] = [name.replace('_', ' ') for name in df_total_red_best["metric_name"]]
        
        logscale_plot = "logscale" if logscale else "normalscale"
        print(os.path.join('./plots', get_experiment_name(), 'syn', folder_name, logscale_plot))
       
        lineplot(df_total_syn_best, x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'syn', folder_name, logscale_plot), 
            filename = filename, 
            #title = 'Seed %d' % (seed), 
            xlabel = "Epochs",
            logscale = logscale,
            show = True)

        lineplot(df_total_red_best, x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'red', folder_name, logscale_plot), 
            filename = filename, 
            #title = 'Seed %d' % (seed), 
            xlabel = "Epochs",
            logscale = logscale)
   
   

def create_training_lineplots(df_params, cluster_root_folder, sftp, cfg, logscale, max_epochs = None):
    
    group_columns = deepcopy(cfg.gridsearch.groupby)
    group_columns.append("training.seed")
   
    
    params_grouped = utils.group_dataframe_by_columns(df_params, group_columns)
    
    
    #Loop over grouper parameters
    for index, row in params_grouped.iterrows():
        global_arguments = {column: row[column] for column in df_params.columns if not "enabled" in column}
        
        #Loop over the runs in this group
        df_total_train_acc, df_total_train_loss = [], []
 
        
        filename = "_".join([key + "_" + "_".join([str(c) for c in list(set(value))]) for key, value in global_arguments.items() if key in group_columns])
        #frac = global_arguments["dataset.parameters.frac"]
        
        #filename += "_frac_%f" % (frac)
        print(filename)
        for argument_comb in zip(*[global_arguments[key] for key in group_columns]):
            arguments = deepcopy(global_arguments)
            
            for i, arg_value in enumerate(argument_comb):
                arguments[group_columns[i]] = arg_value
    
          
            folder_name = utils.create_name_from_arguments(lr = arguments["training.learning_rate"],
                                            wd =  arguments["training.weight_decay"])
            
            ignore_keys = ['seed', 'clustering.normalize', 'model.parameters.initialization', 'clustering.linkage_method', 'clustering.metric', 'model.parameters.activation']
         
            folder_name = "_".join(["%s_%s" % (k, v) if not "." in k else "%s_%s" % (k.split('.')[-1], v) for k, v in arguments.items() if k not in ignore_keys])
            
            
            new_cfg = update_cfg_from_dict(arguments, deepcopy(cfg))
            initialize_hydra(new_cfg)
            update_model_name_and_folder(new_cfg, set_folder = True)
            
            
            df_global, df_best = load_train_run(new_cfg, cluster_root_folder, sftp)

            df_train_acc = utils.convert_dataframe_to_seaborn_format(df_global, ["train_acc", "test_acc"], arguments.keys())
            df_train_loss = utils.convert_dataframe_to_seaborn_format(df_global, ["train_loss", "test_loss"], arguments.keys())
            
            name = ""
            for m in cfg.gridsearch.groupby:
                
                metric_name = m
                if "synergy" in m:
                    metric_name = "synergy"
                elif "redundancy" in m:
                    metric_name = "redundancy"

                name += " %s_%s" % (metric_name, arguments[m])

            df_train_acc['metric_name'] = df_train_acc['metric_name'] + [name for _ in range(len(df_train_acc))]
            df_train_loss['metric_name'] = df_train_loss['metric_name'] + [name for _ in range(len(df_train_loss))]
          
            df_total_train_acc.append(df_train_acc)
            df_total_train_loss.append(df_train_loss)
            
            del df_global
            del df_best
            
        df_total_train_acc = pd.concat(df_total_train_acc)
        df_total_train_loss = pd.concat(df_total_train_loss)
       
        if max_epochs is not None:
            if max_epochs is not None:
                df_total_train_acc = df_total_train_acc[df_total_train_acc['step'] < max_epochs]
                df_total_train_loss = df_total_train_loss[df_total_train_loss['step'] < max_epochs]
        
        
        df_total_train_acc["metric_name"] = [name.replace('_', ' ') for name in df_total_train_acc["metric_name"]]
        df_total_train_loss["metric_name"] = [name.replace('_', ' ') for name in df_total_train_loss["metric_name"]]
        
        train_acc_names = [name for name in df_total_train_acc["metric_name"].unique() if "train" in name]
        test_acc_names = [name for name in df_total_train_acc["metric_name"].unique() if "test" in name]
        
        lineplot(utils.filter_dataframe(df_total_train_acc, metric_name = train_acc_names), x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'train_acc', folder_name), 
            filename = filename, 
            #title = 'Seed %d' % (seed), 
            xlabel = "epochs",
            logscale = logscale,
            show = True)
        
        lineplot(utils.filter_dataframe(df_total_train_acc, metric_name = test_acc_names), x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'test_acc', folder_name), 
            filename = filename, 
            #title = 'Seed %d' % (seed), 
            xlabel = "epochs",
            logscale = logscale,
            show = False)
        
        
        lineplot(df_total_train_loss, x='step', y='metric_value', hue='metric_name', 
            folder = os.path.join('./plots', get_experiment_name(), 'loss', folder_name), 
            filename = filename, 
            #title = 'Seed %d' % (seed), 
            xlabel = "epochs",
            logscale = logscale)
      
   
    

def group_unique_params(param_list, keys):
  """Groups unique parameter combinations based on specified keys.

  Args:
      param_list: A list of dictionaries, where each dictionary represents a set of parameters.
      keys: A list of keys to use for grouping.

  Returns:
      A list of lists, where each inner list contains dictionaries with the same values for the specified keys.
  """

  grouped_params = []
  seen = set()

  for params in param_list:
    key_tuple = tuple(params[key] for key in keys)
    if key_tuple not in seen:
      seen.add(key_tuple)
      grouped_params.append([])
    grouped_params[-1].append(params)

  return grouped_params

def load_run(param, cfg, cluster_root_folder, sftp, load_oinfo = False, load_train = False):
    run_results = defaultdict(list)

    for seed in param["training.seed"]:
        #print(seed)
        #run_param = deepcopy(param)
        #run_param["seed"] = seed 
        
        new_cfg = deepcopy(cfg)
        new_cfg.seed = seed
        update_model_name_and_folder(new_cfg, set_folder = False)
        
        #new_cfg = update_cfg_from_dict(run_param, deepcopy(cfg))
        initialize_hydra(new_cfg)
        
        if load_oinfo:
            df_oinfo = load_oinfo_run(new_cfg, cluster_root_folder, sftp)
            run_results["oinfo"].append(df_oinfo)
            
        if load_train:   
            df_global, df_best = load_train_run(new_cfg, cluster_root_folder, sftp)
            #print(df_oinfo.seed.unique())
            
            run_results["train_global"].append(df_global)
            run_results["train_best"].append(df_best)
      
    for key, value in run_results.items():
        run_results[key] = pd.concat(value)
    
    return run_results

def dict_values_to_unique_name(d, symbol='_'):
    return symbol.join(map(str, d.values()))

def convert_row_to_column(df, metric_name):
    df_tmp = deepcopy(df)
    df_tmp["metric_name"] = [metric_name for _ in range(len(df_tmp))]
    df_tmp["metric_value"] = df[metric_name]
    return df_tmp
        
def phase_plots(cfg, params, logscale = False, plot_seeds = False, truncate_steps = False, download_cluster = False):
    
    params = utils.group_params_by_keys(params, ["training.seed"])
    
    sftp, cluster_root_folder = None, None
    #if download_cluster:
    #    from hidden_cluster_code import set_login_cluster
    #    sftp, ssh, cluster_root_folder = set_login_cluster(cfg)
   
    for i, param in enumerate(params):
       
        new_cfg = update_cfg_from_dict(utils.filter_dict(param, ["training.seed"]), deepcopy(cfg))
        initialize_hydra(new_cfg)
        update_model_name_and_folder(new_cfg)
      
        results = load_run(param, new_cfg, cluster_root_folder, sftp, load_oinfo=True, load_train=True)
       
        
        results["oinfo"] = results["oinfo"][(results["oinfo"]['step'] != 0)]
      
        normalized_results = defaultdict(list)
        for seed in param["training.seed"]:
            df_syn = utils.filter_dataframe(results["oinfo"], metric_name = "synergy", seed = seed)
            df_red = utils.filter_dataframe(results["oinfo"], metric_name = "redundancy", seed = seed)
            #df_norms = utils.filter_dataframe(results["train_global"], step = list(df_syn.step.unique()), seed = seed).reset_index(drop=True)
            
            if truncate_steps:
                optimal_step = min(results["oinfo"][results["oinfo"]["test_acc"] >= 0.98].head(1).step.item() + 100, cfg.steps)
       
                df_syn = df_syn[df_syn['step'] < optimal_step]
                df_red = df_red[df_red['step'] < optimal_step]
           
                #df_norms = df_norms[df_norms['step'] < optimal_step]
                
            """
            df_norms["metric_name"] = ["avg_norm" for _ in range(len(df_norms))]
            df_norms["metric_value"] = list(df_norms["model.f1.weight_avg_l2_norm"])
           
            df_norms = utils.normalize_column(df_norms, remove_outliers = False, inverse_data = False, smooth = False, normalize = True)
            """
            #avg_norms = list(df_train_acc.avg_l2_norm)
        
            #df_syn = df_syn.groupby(arguments).apply(lambda x: normalize_column(x, subset_metric_name="synergy", column="metric_value", remove_outliers=None, inverse_data=True, smooth=None, normalize=True))
            #df_red = df_red.groupby(arguments).apply(lambda x: normalize_column(x, subset_metric_name="redundancy", column="metric_value", remove_outliers=None, inverse_data=False, smooth=None, normalize=True))
        
        
            df_syn_best = df_syn.loc[df_syn.groupby(['step', 'seed'])['metric_value'].idxmin()]
            df_red_best = df_red.loc[df_red.groupby(['step', 'seed'])['metric_value'].idxmax()]
            
            df_syn_best_raw = deepcopy(df_syn_best)
            df_red_best_raw = deepcopy(df_red_best)
            
            df_syn_best_raw["metric_name"] = ["synergy_raw" for syn in range(len(df_syn_best_raw))]
            df_red_best_raw["metric_name"] = ["redundancy_raw" for syn in range(len(df_red_best_raw))]
           
            df_syn_best_normalized = utils.normalize_column(df_syn_best, remove_outliers = False, inverse_data = True, smooth = True, normalize = True)
            df_red_best_normalized = utils.normalize_column(df_red_best, remove_outliers = False, inverse_data = False, smooth = True, normalize = True)
          
            df_train_acc = convert_row_to_column(df_syn_best, "train_acc")
            df_test_acc = convert_row_to_column(df_syn_best, "test_acc")
            df_train_loss = convert_row_to_column(df_syn_best, "train_loss")
            df_test_loss = convert_row_to_column(df_syn_best, "test_loss")
           
            if "size" in df_syn_best.columns:
                df_size_syn = convert_row_to_column(df_syn_best, "size")
            else:
                df_size_syn = convert_row_to_column(df_syn_best, "size_multiplets")
           
           
            df_size_syn = utils.normalize_column(df_size_syn, remove_outliers = True, inverse_data = False, smooth = True, normalize = True)
           
            #df_size_red = convert_row_to_column(df_red_best, "size_multiplets")
            #df_size_red = utils.normalize_column(df_size_red, remove_outliers = False, inverse_data = False, smooth = False, normalize = True)
           
           
            df_train_loss = utils.normalize_column(df_train_loss, remove_outliers = False, inverse_data = False, smooth = False, normalize = True)
            df_test_loss = utils.normalize_column(df_test_loss, remove_outliers = False, inverse_data = False, smooth = False, normalize = True)
            normalized_results["syn_size"].append(df_size_syn)
            #normalized_results["red_size"].append(df_size_red)
            normalized_results["syn_best"].append(df_syn_best_normalized)
            normalized_results["red_best"].append(df_red_best_normalized)
            #normalized_results["syn_best_raw"].append(df_syn_best_raw)
            #normalized_results["red_best_raw"].append(df_red_best_raw)
            normalized_results["train_acc"].append(df_train_acc)
            normalized_results["test_acc"].append(df_test_acc)
            normalized_results["train_loss"].append(df_train_loss)
            normalized_results["test_loss"].append(df_test_loss)
            #normalized_results["norms"].append(df_norms)
            
        
        for k, v in normalized_results.items():
            normalized_results[k] = pd.concat(v)
      
        df_phase_transitions_acc = pd.concat([normalized_results["syn_best"], normalized_results["red_best"], normalized_results["train_acc"], normalized_results["test_acc"], normalized_results["syn_size"]])
        df_phase_transitions_loss = pd.concat([normalized_results["syn_best"], normalized_results["red_best"], normalized_results["train_loss"], normalized_results["test_loss"], normalized_results["syn_size"]])
  
        plot_folder = os.path.join('./plots', utils.get_experiment_name(), 'phase_plot_acc', param["oinfo.features"], "logscale" if logscale else "normalscale", "truncated" if truncate_steps else "normal")
        filename = "_".join([new_cfg.model.unique_name, new_cfg.dataset.unique_name, new_cfg.model_parameters])
     
        #unique_name(param)
        #print(utils.filter_dict_by_value(params, ["model"]))
        #print(utils.filter_dict(param, ["oinfo.features", "train_enabled", "oinfo_enabled", "seed"]))
        title = '%s' % (new_cfg.model_parameters)
        
        utils.create_folder(os.path.join(plot_folder, 'data'))
        
        utils.save_dataframe_to_pickle(os.path.join(plot_folder, 'data'), filename, df_phase_transitions_acc)
        """
        utils.lineplot(df_phase_transitions_acc, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, filename = filename, 
                 title = title, xlabel = None, ylabel = None, logscale = logscale)
        """
       
        plot_folder = os.path.join('./plots', utils.get_experiment_name(), 'phase_plot_loss', param["oinfo.features"], "logscale" if logscale else "normalscale", "truncated" if truncate_steps else "normal")
        #utils.save_dataframe_to_pickle(plot_folder, filename, df_phase_transitions_loss)
        #pd.concat([df_phase_transitions_loss, normalized_results["syn_best_raw"], normalized_results["red_best_raw"]])
        print(os.path.join(plot_folder, 'data'))
        print(filename)
        print(df_phase_transitions_loss.step.unique())
        utils.save_dataframe_to_pickle(os.path.join(plot_folder, 'data'), filename, df_phase_transitions_loss)
       
        utils.lineplot(df_phase_transitions_loss, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, filename = filename, 
                 title = title, xlabel = None, ylabel = None, logscale = logscale, show = False)
        utils.lineplot(df_phase_transitions_acc, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, filename = filename, 
                 title = title, xlabel = None, ylabel = None, logscale = False, show = True)
       
def plot_lth(cfg,  cluster_root_folder, sftp):
    
    from tasks.train import get_train_filenames
    from tasks.lth import get_best_oinfo
    
    run_params = utils.filter_dataframe(pd.DataFrame.from_dict(get_params_from_cfg(cfg, ignore_lth=False, ignore_oinfo=cfg.oinfo_enabled)))
    #lth_params = pd.DataFrame.from_dict(get_params_from_cfg(cfg, ignore_lth=False, ignore_oinfo=True))
 
    for index, row in run_params.iterrows():
        param = dict(row)
        
        
        results_acc, results_loss = defaultdict(list), defaultdict(list)
        for seed in cfg.gridsearch.parameters["training.seed"]["values"]:
            #Load train data
            cfg_train = update_cfg_from_dict(utils.filter_dict(param, ["training.seed"]), deepcopy(cfg))
            cfg_train.seed = seed
            initialize_hydra(cfg_train)
            update_model_name_and_folder(cfg_train)
            
            #df_global, df_best = load_train_run(cfg_train, cluster_root_folder, sftp)
            df_best_syn, df_best_red = get_best_oinfo(cfg_train)
         
            df_train_acc = convert_row_to_column(deepcopy(df_best_syn), "train_acc")
            df_train_acc["metric_name"] = [metric_name + "_original" for metric_name in df_train_acc["metric_name"]]
            results_acc[df_train_acc.metric_name.unique()[0]].append(df_train_acc)
            
            df_test_acc = convert_row_to_column(deepcopy(df_best_syn), "test_acc")
            df_test_acc["metric_name"] = [metric_name + "_original" for metric_name in df_test_acc["metric_name"]]
            results_acc[df_test_acc.metric_name.unique()[0]].append(df_test_acc)
            
            df_train_loss = convert_row_to_column(deepcopy(df_best_syn), "train_loss")
            df_train_loss["metric_name"] = [metric_name + "_original" for metric_name in df_train_loss["metric_name"]]
            results_loss[df_train_loss.metric_name.unique()[0]].append(df_train_loss)
            
            df_test_loss = convert_row_to_column(deepcopy(df_best_syn), "test_loss")
            df_test_loss["metric_name"] = [metric_name + "_original" for metric_name in df_test_loss["metric_name"]]
            results_loss[df_test_loss.metric_name.unique()[0]].append(df_test_loss)
           
            
            best_syn_step = df_best_syn.loc[df_best_syn['metric_value'].idxmin()]["step"]
            
            generalizing_step = df_best_syn.loc[df_best_syn[df_best_syn['test_acc'] > 0.98]["step"].idxmin()]["step"]
            best_syn_generalizing = df_best_syn.loc[df_best_syn['step'] == generalizing_step]
            
            for metric in cfg.gridsearch.parameters["lth.metric"]["values"]:
                
                cfg_train.seed = seed
                cfg_train.lth.metric = metric
                initialize_hydra(cfg_train)
                update_model_name_and_folder(cfg_train)
                
            
                df_global_metric, df_best_metric = load_train_run(cfg_train, cluster_root_folder, sftp, metric_step= generalizing_step if "generalizing" in metric else best_syn_step, multiplet_name= metric)
                
                df_metric_train_acc = convert_row_to_column(deepcopy(df_global_metric), "train_acc")
                df_metric_train_acc["metric_name"] = [metric_name + "_" + metric for metric_name in df_metric_train_acc["metric_name"]]
                
                df_metric_test_acc = convert_row_to_column(deepcopy(df_global_metric), "test_acc")
                df_metric_test_acc["metric_name"] = [metric_name + "_" + metric for metric_name in df_metric_test_acc["metric_name"]]
                
                df_metric_train_loss = convert_row_to_column(deepcopy(df_global_metric), "train_loss")
                df_metric_train_loss["metric_name"] = [metric_name + "_" + metric for metric_name in df_metric_train_loss["metric_name"]]
                
                df_metric_test_loss = convert_row_to_column(deepcopy(df_global_metric), "test_loss")
                df_metric_test_loss["metric_name"] = [metric_name + "_" + metric for metric_name in df_metric_test_loss["metric_name"]]
                
                results_acc[metric + "_train"].append(df_metric_train_acc)
                results_acc[metric + "_test"].append(df_metric_test_acc)
                
                results_loss[metric + "_train"].append(df_metric_train_loss)
                results_loss[metric + "_test"].append(df_metric_test_loss)
                
        
        df_results_acc_train, df_results_acc_test, df_results_loss_train, df_results_loss_test = [], [], [], []
        for k, v in results_acc.items():
            df_combined = pd.concat(v)
          
            if "train" in k:
                df_results_acc_train.append(df_combined)
            else:
                df_results_acc_test.append(df_combined)
        
        df_results_acc_train = pd.concat(df_results_acc_train)
        df_results_acc_test = pd.concat(df_results_acc_test)
        
        
        for k, v in results_loss.items():
            df_combined = pd.concat(v)
          
            if "train" in k:
                df_results_loss_train.append(df_combined)
            else:
                df_results_loss_test.append(df_combined)
        
        df_results_loss_train = pd.concat(df_results_loss_train)
        df_results_loss_test = pd.concat(df_results_loss_test)
        
       
        
        #utils.create_folder(os.path.join(plot_folder, 'data'))
        #utils.save_dataframe_to_pickle(os.path.join(plot_folder, 'data'), filename, df_phase_transitions_acc)
        plot_folder = os.path.join('./plots', utils.get_experiment_name(), 'lth')
        
        df_results_loss_train["metric_name"] = [name.replace('_', ' ') for name in df_results_loss_train["metric_name"]]
        df_results_loss_test["metric_name"] = [name.replace('_', ' ') for name in df_results_loss_test["metric_name"]]
        df_results_acc_train["metric_name"] = [name.replace('_', ' ') for name in df_results_acc_train["metric_name"]]
        df_results_acc_test["metric_name"] = [name.replace('_', ' ') for name in df_results_acc_test["metric_name"]]
        
        filename = "_".join(["train_acc", cfg_train.model.unique_name, cfg_train.dataset.unique_name, cfg_train.model_parameters])
        print(plot_folder)
        print(filename)
        utils.lineplot(df_results_acc_train, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, 
                 filename = filename, 
                 #title = title, 
                 xlabel = "Epochs", ylabel = None, show = False, logscale=False)
        
        filename = "_".join(["train_loss", cfg_train.model.unique_name, cfg_train.dataset.unique_name, cfg_train.model_parameters])
        
        utils.lineplot(df_results_loss_train, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, 
                 filename = filename, 
                 #title = title, 
                 xlabel = "Epochs", ylabel = None, show = False, logscale=False)
        
        
        filename = "_".join(["test_acc", cfg_train.model.unique_name, cfg_train.dataset.unique_name, cfg_train.model_parameters])
        
        utils.lineplot(df_results_acc_test, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, 
                 filename = filename, 
                 #title = title, 
                 xlabel = "Epochs", ylabel = None, show = False, logscale=False)
        
        filename = "_".join(["test_loss", cfg_train.model.unique_name, cfg_train.dataset.unique_name, cfg_train.model_parameters])
        
        utils.lineplot(df_results_loss_test, x='step', y='metric_value', hue='metric_name', 
                 folder = plot_folder, 
                 filename = filename, 
                 #title = title, 
                 xlabel = "Epochs", ylabel = None, show = False, logscale=False)
   
        
@hydra.main(version_base=None, config_name="main", config_path="../config")
def main(cfg) -> None:
  
    initialize_hydra(cfg)
   
    params = get_params_from_cfg(cfg, ignore_lth=cfg.lth_enabled, ignore_oinfo=cfg.oinfo_enabled)
    download_cluster = True
    overwrite = False
    print("total %d" % len(params))
    
    sftp, cluster_root_folder = None, None
    #if download_cluster:
    #    from hidden_cluster_code import set_login_cluster
    #    sftp, ssh, cluster_root_folder = set_login_cluster(cfg)
        
    df_params = pd.DataFrame.from_dict(params)
    
    if cfg.train_enabled:
        #training_heatmaps(cfg, params)
        
        create_training_lineplots(df_params = df_params, cluster_root_folder = cluster_root_folder, sftp = sftp, cfg = cfg,
                                                logscale = False, max_epochs=cfg.gridsearch.max_epochs) 
       
        
    if cfg.oinfo_enabled:
        #Store the oinfo results to allow adding custom lines in jupyter notebook
        folder_best_oinfo_path = os.path.join('./plots', utils.get_experiment_name(), 'data')
       
        phase_plots(cfg, params, logscale = True, plot_seeds = False, truncate_steps = False, download_cluster=download_cluster)
        #create_oinfo_lineplots(df_params = df_params, cluster_root_folder = cluster_root_folder, sftp = sftp, cfg = cfg,
        #                                        logscale = False, overwrite = overwrite, max_epochs=cfg.gridsearch.max_epochs) 
        #create_oinfo_lineplots(df_params = df_params, cluster_root_folder = cluster_root_folder, sftp = sftp, cfg = cfg,
        #                                        logscale = True, overwrite = overwrite, max_epochs=cfg.gridsearch.max_epochs)
                                            
        #utils.save_dataframe_to_pickle(folder_best_oinfo_path, "best_oinfo_exp_data", results.best_oinfo)
        #utils.save_dataframe_to_pickle(folder_best_oinfo_path, "global_oinfo_exp_data", results.global_oinfo)
        #phase_plots(cfg, params, logscale = True, plot_seeds = False)
        #phase_plots(cfg, params, logscale = False, plot_seeds = False, truncate_steps = False, download_cluster=download_cluster)
        
        
    if cfg.lth_enabled:
        plot_lth(cfg, cluster_root_folder, sftp)
        
        
    
if __name__ == "__main__":
    main()