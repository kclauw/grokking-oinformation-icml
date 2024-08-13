import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import numpy as np


@torch.no_grad()
def check_accuracy_grokking(
    X:torch.Tensor, Y:torch.Tensor, model: nn.Module, device: str, mask
) -> tuple:
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    #x_wrong = []
   

    
    if mask is not None:
        features, scores = model(X, mask)
    else:
        features, scores = model(X)
        
    """
    if scaler is None:
        features, scores = model.masked_forward(X, mask.to(device))
    else:
        with torch.cuda.amp.autocast():
            features, scores = model(X)
    """
   
    _, preds = scores.max(1)
    

    num_correct += (preds == Y).sum()
    num_samples += preds.size(0)
    #x_wrong.append(X[Y != preds])
    acc = float(num_correct) / num_samples

    return features, num_correct, num_samples, acc

def acc_calc(dataloader, model, loss = None, mask_idx=None, device='cuda', width=None, faithfulness=False, return_features = False, dataset_name = None, dtype = None):
    model.eval()
    
    total_acc, total = 0, 0
    total_features = defaultdict(list)
  
    for bid, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        if mask_idx is not None:
            # create mask
            for layer_name, mask in mask_idx.items():
                idx = torch.LongTensor([mask for _ in range(len(y_batch))])
                new_mask = torch.zeros(len(y_batch), width)
                new_mask.scatter_(1, idx, 1.).to(device)
                mask_idx[layer_name] = new_mask.to(device)
            features, pred = model(x_batch, mask_idx)
        else:
            features, pred = model(x_batch)
        
      
        if loss == "hinge":
            if (faithfulness):
                acc = (torch.sign(torch.squeeze(pred)) == torch.sign(torch.squeeze(fullmodel_pred))).sum().item()    
            else:
                acc = (torch.sign(torch.squeeze(pred)) == y_batch).sum().item()
            acc = acc / y_batch.shape[0]
        else:
            if (faithfulness):
                fullmodel_pred = model(x_batch)
                features, num_correct, num_samples, acc = check_accuracy_grokking(
                        X = x_batch, Y = fullmodel_pred, model = model, device = device, mask = mask_idx)
            else:
                features, num_correct, num_samples, acc = check_accuracy_grokking(
                        X = x_batch, Y = y_batch, model = model, device = device, mask = mask_idx)
        
        
        total_acc += acc
        for key, feature in features.items():
            total_features[key].append(feature.detach().cpu().numpy())
            
        total += 1
    
  
    if return_features:
        for key, value in total_features.items():
            total_features[key] = np.concatenate(value, axis=0)
        return total_acc / total, total_features
   
    return total_acc / total

def loss_calc(dataloader, model, loss_fn, dataset_name, device='cuda'):
    model.eval()

    loss, total = 0, 0
    for id, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
   
        features, pred = model(x_batch)
        
        if isinstance(loss_fn, nn.MSELoss):
            y_batch = F.one_hot(y_batch, num_classes=pred.shape[-1]).to(device=device)
        elif isinstance(loss_fn, nn.CrossEntropyLoss):
            y_batch = y_batch.to(torch.long)
        
        loss += loss_fn(pred, y_batch).sum().item()
       
        total += x_batch.shape[0]

    return loss / total

def evaluate_model(model_name, dataloader, model, loss, cfg):
    if "fcn" in model_name or "encoder_decoder" in model_name:
        acc, features = acc_calc(dataloader, model, device = cfg.device, return_features = True, dataset_name = cfg.dataset.name, loss = cfg.model.parameters.loss)
        
        loss = loss_calc(dataloader, model, device = cfg.device, loss_fn=loss, dataset_name = cfg.dataset.name)
        margin = 0
        return loss, acc, margin, features

def load_model(cfg, device):
    if "modular" in cfg.dataset.name:
        input_size = 2*cfg.dataset.parameters.p
        output_size = cfg.dataset.parameters.p
        flatten = True
    elif "parity" in cfg.dataset.name:
        input_size = cfg.dataset.parameters.input_size
        output_size = cfg.dataset.parameters.output_size
        flatten = True
    elif "repr" in cfg.dataset.name:
        input_size = cfg.dataset.parameters.symbol_rep_dim
        output_size = input_size 
        
    if "fcn" in cfg.model.name:
        from models.architectures.fully_connected_network import FCNN
        model = FCNN(activation_function=cfg.model.parameters.activation, input_size=input_size, output_size=output_size, layers=cfg.model.parameters.layers, flatten_initial=flatten, init_method=cfg.model.parameters.initialization)
    elif "encoder_decoder" in cfg.model.name:
        from models.architectures.encoder_decoder import EncoderDecoder
        model = EncoderDecoder(internal_rep_dim = cfg.model.parameters.internal_rep_dim, activation_function=cfg.model.parameters.activation, input_size=input_size, output_size=output_size, cfg_encoder=cfg.model.parameters.encoder, cfg_decoder=cfg.model.parameters.encoder, init_method=cfg.model.parameters.initialization)
        
        
        
        
    #model = FCNN(input_size, output_size, cfg.model.parameters.layers, flatten, init_method = cfg.model.parameters.initialization, set_last_layer_to_zero = cfg.model.parameters.set_last_layer_to_zero)
    
    return model.to(device)

@torch.no_grad()
def get_model_stats(model):
    stats = {}
  
    stats[f'global_avg_l2_norm'] = np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))
    
    for name, param in model.named_parameters():
        stats[f'{name}_avg_l2_norm'] = np.sqrt(param.pow(2).sum().item())

        #stats[f'norm_{name}'] = cur_norm2 ** 0.5
     
        if not "bias" in name:
            stats[f'norm_feats_{name}'] = torch.linalg.norm(param, dim=1).detach().cpu().numpy()
         
        if "output" in name:
            stats[f'norm_conx_{name}'] = torch.squeeze(param).detach().cpu().numpy()

    return stats
