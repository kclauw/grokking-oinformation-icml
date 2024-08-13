#https://github.com/ejmichaud/grokking-squared/blob/main/toy/train_add.py
#https://github.com/ejmichaud/grokking-squared/blob/main/scripts/run_toy_model.py
import numpy as np 
import random
import torch
from itertools import islice, product

def gen_mod_add_representation(seed, frac, p, operation, symbol_rep_dim, device, train_num = 1000, label_scale = 1.0):
    
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
    
    
    
  
    
    