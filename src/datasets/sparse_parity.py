from torch.utils.data import TensorDataset, DataLoader
import torch
import random

def create_parity_dataset(cfg, train_samples, test_samples):
    train_data = sparse_parity(cfg.dataset.parameters.input_size, cfg.dataset.parameters.k, train_samples, seed=cfg.seed*17)
    train_dataset = TensorDataset(train_data[0], train_data[1])
    
    test_data = sparse_parity(cfg.dataset.parameters.input_size, cfg.dataset.parameters.k, test_samples, seed=cfg.test_data_seed)
    test_dataset = TensorDataset(test_data[0], test_data[1])

    """
    train_data = sparse_parity(cfg.dataset.parameters.input_size, cfg.dataset.parameters.k, cfg.dataset.train_samples, seed=cfg.seed*17)
    train_dataset = TensorDataset(train_data[0], train_data[1])
    test_data = sparse_parity(cfg.dataset.parameters.input_size, cfg.dataset.parameters.k, cfg.dataset.test_samples, seed=cfg.test_data_seed)
    test_dataset = TensorDataset(test_data[0], test_data[1])
    """
    return train_dataset, test_dataset

def sparse_parity(n, k, n_samples, seed=42):
    random.seed(seed)
    samples = torch.Tensor([[random.choice([-1, 1]) for j in range(n)] for i in range(n_samples)])
    # targets = torch.prod(input[:, n//2:n//2+k], dim=1) # parity hidden in the middle
    targets = torch.prod(samples[:, :k], dim=1) # parity hidden in first k bits
    return samples, targets