from copy import deepcopy
import omegaconf

import torch 
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import pandas as pd
import os

from models.configuration import optimizer_dict, loss_function_dict
from models.utils import evaluate_model, get_model_stats
from loggers.simple_logger import Logger
from datasets import load_data


import utils
from models.utils import load_model

def load_train_run(cfg, cluster_root_folder, sftp, load_model_step = None, norms = False, multiplet_name = None, overwrite = False, metric_step = None):
    #Downloads file if not exists
    filename_best_results, filename_global_results = get_train_filenames(cfg = cfg, 
                                                                         norms = norms, 
                                                                         multiplet_name = multiplet_name,
                                                                         metric_step = metric_step)
    
    logger_global_results = Logger(cfg.exp_folder, filename_global_results)
    logger_best_results = Logger(cfg.exp_folder, filename_best_results)
    """
    if not logger_global_results.exists() or overwrite:
        cluster_result_folder = os.path.join(cluster_root_folder, cfg.exp_folder.split('/results/')[-1])
        local_folder = os.path.join(cfg.exp_folder)
        utils.download_file_from_cluster(sftp, cluster_result_folder, cfg, filename_global_results + ".pkl", local_folder)
        utils.download_file_from_cluster(sftp, cluster_result_folder, cfg, filename_best_results+ ".pkl", local_folder)
    """
    logger_global_results.load()
    logger_best_results.load()
    
 
    df_global = logger_global_results.to_dataframe()
    df_best = logger_best_results.to_dataframe()
    
    return df_global, df_best

def get_train_filenames(cfg, load_model_step = None, norms = False, multiplet_name = None, metric_step = None):
    
    filename_best_results = "best_results_evalsteps_%d_steps_%d" % (cfg.training.evaluation_step, cfg.training.steps)
    filename_global_results = "global_results_evalsteps_%d_steps_%d" % (cfg.training.evaluation_step, cfg.training.steps)
    
    if load_model_step != None:
        filename_best_results += "_initialization_%s_%d" % (multiplet_name, load_model_step)
        filename_global_results += "_initialization_%s_%d" % (multiplet_name, load_model_step)
        
    if multiplet_name != None:
        filename_best_results += "_metric_%s_%d" % (multiplet_name, metric_step)
        filename_global_results += "_metric_%s_%d" % (multiplet_name, metric_step)
        
    if norms:
        filename_best_results += "_norms"
        filename_global_results += "_norms"
    
    if cfg.train_percentage_enabled and "parity" in cfg.dataset.name:
        filename_best_results += "_train_percentage_%f_datasize_%d" % (cfg.train_percentage, cfg.train_percentage_size)
        filename_global_results += "_train_percentage_%f_datasize_%d" % (cfg.train_percentage, cfg.train_percentage_size)
    
    if cfg.regularization.synergy_weight != 0:
        filename_best_results += "_synergy_%f" % (cfg.regularization.synergy_weight)
        filename_global_results += "_synergy_%f" % (cfg.regularization.synergy_weight)
    
    if cfg.regularization.redundancy_weight != 0:
        filename_best_results += "_redundancy_%f" % (cfg.regularization.redundancy_weight)
        filename_global_results += "_redundancy_%f" % (cfg.regularization.redundancy_weight)
    
    return filename_best_results, filename_global_results

def train(cfg, oinfo_name = None, multiplets = None, load_model_step = None, norms = False, multiplet_name = None, save_model = False, metric_step = None):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dtype = torch.float64
    device = cfg.device
    utils.seed_everything(cfg.training.seed)

    if cfg.wandb.enabled:
        import wandb
        tmp = deepcopy(cfg)
        tmp.paths.output_dir = ""
        tmp.paths.work_dir = ""
        
        wandb.init(project="grokking_%s" % (cfg.experiment_name),settings=wandb.Settings(start_method="thread"), config=omegaconf.OmegaConf.to_container(
            tmp, resolve=True, throw_on_missing=True
        ))
        wandb.define_metric("*", step_metric="global_step")
    
    if cfg.neptune.enabled:
        neptune_logger = utils.load_neptune_logger(cfg, tags = ["Training"])
    
    train_loader, train_loader_for_eval, test_loader = load_data(cfg, device)
  
    filename_best_results, filename_global_results = get_train_filenames(cfg = cfg, 
                                                                         load_model_step = load_model_step, 
                                                                         norms = norms, 
                                                                         multiplet_name = multiplet_name,
                                                                         metric_step = metric_step)

    if oinfo_name is not None:
        filename_best_results = oinfo_name + "_" + filename_best_results
        filename_global_results = oinfo_name + "_" + filename_global_results
  
    logger_global_results = Logger(cfg.exp_folder, filename_global_results)
    logger_best_results = Logger(cfg.exp_folder, filename_best_results)
    
    
    loss_fn = loss_function_dict[cfg.model.parameters.loss]
    model = load_model(cfg, device)
    
    if "arguments" in cfg.optimizer:
        optimizer = optimizer_dict[cfg.optimizer.name.split('_')[0]](model.parameters(), lr=cfg.training.learning_rate, weight_decay = cfg.training.weight_decay, **cfg.optimizer.arguments)
    else:
        optimizer = optimizer_dict[cfg.optimizer.name.split('_')[0]](model.parameters(), lr=cfg.training.learning_rate, weight_decay = cfg.training.weight_decay)
   
    if cfg.training.alpha > 0:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.endswith("weight"):
                    p.data *= cfg.training.alpha
            norm = np.sqrt(sum(p.pow(2).sum().item() for n, p in model.named_parameters() if n.endswith("weight")))
  
    if cfg.model.name != "encoder_decoder" and cfg.set_last_layer_zero_init and cfg.training.alpha != 1:
        model.set_last_layer_zero_init()
    

    if load_model_step is not None: 
        if device == "cpu":
            model.load_state_dict(torch.load(os.path.join(cfg.exp_folder, 'checkpoints', 'model_%d.pt' % (load_model_step)), map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(os.path.join(cfg.exp_folder, 'checkpoints', 'model_%d.pt' % (load_model_step))))
   
    #Train model
    model.train()
    
  
    best_train_acc, best_test_acc, best_train_loss, best_test_loss = 0, 0, np.inf, np.inf
    train_acc_step, test_acc_step, optimal_grokking_step, optimal_grokking_difference = None, None, None, None
    train_found, test_found = False, False
    
    for step in range(cfg.training.steps):
        
        if save_model and (step % cfg.oinfo.evaluation_step == 0 or step == 0):
            torch.save(model.state_dict(), os.path.join(cfg.exp_folder, 'checkpoints', 'model_%d.pt' % (step)))
            
        if multiplets is not None:
            utils.prune(cfg, model, multiplets)
            
        for bid, (batch_x, batch_y) in enumerate(train_loader):    
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
           

            if step % cfg.training.evaluation_step == 0:
                with torch.no_grad():
                    model.eval()
                   
                    train_loss, train_acc, train_margin, features = evaluate_model(cfg.model.name, train_loader_for_eval, model, loss_fn, cfg)
                    test_loss, test_acc, test_margin, features = evaluate_model(cfg.model.name, test_loader, model, loss_fn, cfg)
                    
                    print("step acc {} acc {:3.3f} | {:3.3f} | loss {:3.3f} | {:3.3f}".format(step, train_acc, test_acc, train_loss, test_loss))
                   
                    #Log results + names for pandas dataframe + norms
                    logger_global_results.add(
                        step = step,
                        train_acc = train_acc,
                        test_acc = test_acc,
                        train_loss = train_loss,
                        test_loss = test_loss,
                        test_margin = test_margin,
                        train_margin = train_margin,
                        load_model_step = load_model_step,
                    )
                    utils.add_cfg_arguments_to_dictionary(cfg, logger_global_results.data, ignore_oinfo_keys = True, ignore_lth_keys = True)

                    norm_stats = get_model_stats(model)
           
                    for norm_key, norm_value in norm_stats.items():
                        logger_global_results.add(
                            **{norm_key: norm_value}
                        )
 
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                
                if train_acc > 0.98 and not train_found:
                    
                    if cfg.wandb.enabled:
                        wandb.log({"optimal_train_step" : step})
                    
                    if cfg.neptune.enabled:
                        neptune_logger["optimal_train_step"].append(value=step,step=step)
 
                        
                    train_acc_step = step
                    train_found = True
                
                if test_acc > 0.98 and not test_found:
                    if cfg.wandb.enabled:
                        wandb.log({"optimal_test_step" : step})
                    if cfg.neptune.enabled:
                        neptune_logger["optimal_test_step"].append(value=step,step=step)
                        
                    test_acc_step = step
                    test_found = True
                    
                    
                    if cfg.wandb.enabled:
                        wandb.log({"optimal_grokking_step" : step})
                        wandb.log({"optimal_grokking_difference" : test_acc_step - train_acc_step})
                    
                    if cfg.neptune.enabled:
                        neptune_logger["optimal_grokking_step"].append(value=step,step=step)
                        neptune_logger["optimal_grokking_difference"].append(value=test_acc_step - train_acc_step,step=step)
                        
                
                if cfg.wandb.enabled:
                    wandb.log({"global_step" : step, "train_acc" : train_acc, "test_acc" : test_acc, "train_loss" : train_loss, "test_loss" : test_loss})

                if cfg.neptune.enabled:
                    neptune_logger["global_step"].append(value=step,step=step)
                    neptune_logger["train_acc"].append(value=train_acc,step=step)
                    neptune_logger["test_acc"].append(value=test_acc,step=step)
                    neptune_logger["train_loss"].append(value=train_loss,step=step)
                    neptune_logger["test_loss"].append(value=test_loss,step=step)
     
                        
                model.train()
                
            break #Sample a single batch (TODO: is there a cleaner way to do this?)
        if test_found and cfg.grokking_stop and step < test_acc_step + 100:
            break
                              
        optimizer.zero_grad(set_to_none=True)
        
       
        features, out = model(batch_x)
       
        if isinstance(loss_fn, nn.MSELoss):
            batch_y = F.one_hot(batch_y, num_classes=out.shape[-1]).to(dtype=torch.float32)
            loss = loss_fn(out, batch_y).to(device=device)
        else:
            batch_y = batch_y.to(torch.long)
            loss = loss_fn(out, batch_y).to(device=device)

        """
        if cfg.regularization.synergy_weight != 0 or cfg.regularization.redundancy_weight != 0:
            from tasks.oinfo import OinfoOriginal
            from clustering.hierarchical import hierarchical_clustering

            with torch.no_grad():
                model.eval()
                
                x = features[cfg.regularization.features].cpu().detach().numpy().astype(np.float64)
                
                clustered_labels_to_neurons = hierarchical_clustering(x, cfg.clustering.num_clusters, 
                                        linkage_method = cfg.clustering.linkage_method, 
                                        metric = cfg.clustering.metric, verbose=True, normalize = cfg.clustering.normalize)
    
                
                #model = OinfoOriginal(x, clustered_labels_to_neurons = clustered_labels_to_neurons, verbose=True)
                #df, df_syn, df_red = model.fit_exhaustive(minsize=2, maxsize=cfg.oinfo.max_clusters, method=cfg.oinfo.normalization, n_clusters = cfg.clustering.num_clusters, batch_size = cfg.oinfo.batch_size)
            
                
                
                    
                m = OinfoOriginal(x, clustered_labels_to_neurons = clustered_labels_to_neurons, verbose=False)
                df, df_syn, df_red = m.fit_exhaustive(minsize=2, maxsize=cfg.regularization.max_clusters, method=cfg.oinfo.normalization, n_clusters = cfg.clustering.num_clusters, batch_size = cfg.oinfo.batch_size)
            
                model.train()
            
                if cfg.regularization.synergy_weight != 0:
                    syn_loss = df_syn["metric_value"].item() * cfg.regularization.synergy_weight
                    loss += torch.tensor(syn_loss)
                    logger_global_results.add(syn_loss = syn_loss) 
                    #wandb.log({"global_step" : step, "syn_loss" : syn_loss})
                     
                if cfg.regularization.redundancy_weight != 0:
                    red_loss = df_syn["metric_value"].item() * cfg.regularization.redundancy_weight
                    loss += torch.tensor(red_loss)
                    #wandb.log({"global_step" : step, "red_loss" : red_loss})
        """
        if cfg.model.parameters.loss == "hinge":
            loss = loss.mean()
      
        loss.backward()
        optimizer.step()
        #if cfg.optimizer.scheduler:
        #    scheduler.step()
        
        # rescale weights such that the weight norm remains a constant in training.
        if cfg.training.rescale_weight_norm:
            #L2_new = L2(model)
            #rescale(model, torch.sqrt(L2_/L2_new))
            with torch.no_grad():
                new_norm = np.sqrt(
                    sum(p.pow(2).sum().item() for n, p in model.named_parameters() if n.endswith("weight")))
                for n, p in model.named_parameters():
                    if n.endswith("weight"):
                        p.data *= norm / new_norm
                            
    if test_acc_step == None or train_acc_step == None:
        train_test_difference = None
    else:
        train_test_difference = int(test_acc_step - train_acc_step)

    logger_best_results.add(
        best_train_acc = best_train_acc,
        best_test_acc = best_test_acc,
        best_train_loss = best_train_loss,
        best_test_loss = best_test_loss,
        #train_samples = train_samples,
        #test_samples = test_samples,
        train_acc_step = int(train_acc_step) if train_acc_step is not None else train_acc_step,
        test_acc_step = int(test_acc_step) if test_acc_step is not None else test_acc_step,
        train_test_difference = train_test_difference,
        load_model_step = load_model_step,
    )

    utils.add_cfg_arguments_to_dictionary(cfg, logger_best_results.data, ignore_oinfo_keys=True, ignore_keys = ["steps", "evaluation_step"])
    logger_best_results.save()
    logger_global_results.save()
    
    if cfg.wandb.enabled:
        wandb.finish()


def load_training_files(logger_global_results, logger_best_results, total_results, load_global_train = True):

    if load_global_train:
        df_train_global = logger_global_results.load()
        if isinstance(df_train_global, pd.core.frame.DataFrame):
            df_global_train_acc = utils.convert_column_to_row(df_train_global, "train_acc")
    
            df_global_test_acc = utils.convert_column_to_row(df_train_global, "test_acc")
            df_global_train_loss = utils.convert_column_to_row(df_train_global, "train_loss")
            df_global_test_loss = utils.convert_column_to_row(df_train_global, "test_loss")
        
            df_train_global = pd.concat([df_global_train_acc, df_global_test_acc, df_global_train_loss, df_global_test_loss])
            df_train_global = df_train_global.drop(columns=["test_acc", "train_acc", "train_loss", "test_loss"])
            total_results["global_train"].append(df_train_global)
    
    df_train_best = logger_best_results.load()
    if isinstance(df_train_best, pd.core.frame.DataFrame):
        total_results["best_train"].append(df_train_best)
