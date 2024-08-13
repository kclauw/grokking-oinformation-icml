
from datasets.modular_dataset import gen_mod_add
from datasets.modular_representation_dataset import gen_mod_add_representation
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

def compute_train_test_split(cfg):
    if cfg.train_percentage_enabled:
        train_samples = int(cfg.dataset.parameters.samples * cfg.dataset.parameters.frac)
        test_samples = int(cfg.dataset.parameters.samples - train_samples)
    else:
        train_samples = cfg.dataset.train_samples
        test_samples = cfg.dataset.test_samples

    return train_samples, test_samples

def load_data(cfg, device):
    if cfg.dataset.name.split('_')[0] == "modular":
        cfg_dataset = cfg.dataset.parameters
        train_data, test_data = gen_mod_add(cfg_dataset.seed, cfg_dataset.frac, cfg_dataset.p, cfg_dataset.operation, noise_level = cfg_dataset.noise_level, fixed_seed = cfg_dataset.fixed_seed, device = device)
       
        train_dataset = TensorDataset(train_data[0].to(device=device), train_data[1].to(device=device))
        test_dataset = TensorDataset(test_data[0].to(device=device), test_data[1].to(device=device))
    elif "repr" in cfg.dataset.name:
        cfg_dataset = cfg.dataset.parameters
        train_data, test_data = gen_mod_add_representation(cfg_dataset.seed, cfg_dataset.frac, cfg_dataset.p, cfg_dataset.operation, cfg_dataset.symbol_rep_dim, device = device)
      
        train_dataset = TensorDataset(train_data[0].to(device=device), train_data[1].to(device=device))
        test_dataset = TensorDataset(test_data[0].to(device=device), test_data[1].to(device=device))
    elif cfg.dataset.name == "sparse_parity":
        from datasets.sparse_parity import create_parity_dataset
        train_samples, test_samples = compute_train_test_split(cfg)
        train_dataset, test_dataset = create_parity_dataset(cfg, train_samples, test_samples)
       
    
    train_loader = DataLoader(deepcopy(train_dataset), batch_size=cfg.dataset.batch_size, shuffle=True)

    train_loader_for_eval = DataLoader(deepcopy(train_dataset), batch_size=cfg.dataset.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.eval_batch_size, shuffle=False)
    
    return train_loader, train_loader_for_eval, test_loader
