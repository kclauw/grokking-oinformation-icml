#https://github.com/ejmichaud/grokking-squared/blob/main/toy/train_add.py
#https://github.com/ejmichaud/grokking-squared/blob/main/scripts/run_toy_model.py
import numpy as np 
import random
import torch
from itertools import islice, product

def gen_mod_add_representation(seed, frac, p, operation, symbol_rep_dim, device, train_num = 1000, label_scale = 1.0):
    
    # -----------------------------Dataset------------------------------#
    # the full dataset contains p(p+1)/2 samples. Each sample has input (a, b).
    all_num = (
        p * (p + 1) // 2
    )  # for addition (abelian group), we deem a+b and b+a the same sample
    D0_id = []
    xx_id = []
    yy_id = []
    for i in range(p):
        for j in range(i, p):
            D0_id.append((i, j))
            xx_id.append(i)
            yy_id.append(j)
    xx_id = np.array(xx_id)
    yy_id = np.array(yy_id)
    
    inputs_id = np.transpose(np.array([xx_id, yy_id]))
    out_id = xx_id + yy_id
    
    y_templates = np.random.normal(0, 1, size=(2 * p - 1, symbol_rep_dim)) * label_scale
    y_templates = torch.tensor(y_templates, dtype=torch.float, requires_grad=True).to(
        device
    )
    
    train_id = np.random.choice(all_num, train_num, replace=False)
    test_id = np.array(list(set(np.arange(all_num)) - set(train_id)))
    
    labels_train = y_templates[out_id[train_id]].detach().clone().requires_grad_(True)
    in_id_train = inputs_id[train_id]
    out_id_train = out_id[train_id]

    labels_test = y_templates[out_id[test_id]].detach().clone().requires_grad_(True)
    in_id_test = inputs_id[test_id]
    out_id_test = out_id[test_id]
    print(in_id_train.shape)
    print(labels_train.shape)
    exit(0)
    
    # Define data set
    symbol_reps = dict()
    for i in range(2 * p - 1):
        symbol_reps[i] = torch.randn((1, symbol_rep_dim)).to(device)
    
    table = dict()
    pairs = [(i, j) for (i, j) in product(range(p), range(p)) if i <= j]
    for (i, j) in pairs:
        if operation == "add":
            table[(i, j)] = (i + j)
    train_pairs = random.sample(pairs, int(len(pairs) * frac))
    test_pairs = [pair for pair in pairs if pair not in train_pairs]
    train_data = (
        torch.cat([torch.cat((symbol_reps[i], symbol_reps[j]), dim=1) for i, j in train_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in train_pairs])
    )
   
    test_data = (
        torch.cat([torch.cat((symbol_reps[i], symbol_reps[j]), dim=1) for i, j in test_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in test_pairs])
    )
  
    return train_data, test_data
    
    
    
  
    
    