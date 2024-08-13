#https://github.com/vfleaking/grokking-dichotomy

import torch 
from torch.utils.data import Dataset
from torch.nn import functional as F
from copy import deepcopy
class BinOpDataset(Dataset):
    def __init__(self, p, op_type='add'):
        self.p = p
        self.op_type = op_type

        self.data = torch.tensor([(x1, x2, self.op(x1, x2)) for x1, x2 in self.op_domain()])
        self.x1 = self.data[:, 0]
        self.x2 = self.data[:, 1]
        self.target = self.data[:, 2]
    
    def op_domain(self):
        if self.op_type == 'div':
            return [(x1, x2) for x1 in range(self.p) for x2 in range(1, self. p)]
        else:
            return [(x1, x2) for x1 in range(self.p) for x2 in range(self. p)]
    
    def op(self, x1: int, x2: int):
        if self.op_type == 'add':
            return (x1 + x2) % self.p
        elif self.op_type == 'max':
            return max(x1, x2)
        elif self.op_type == 'x':
            return x1
        elif self.op_type == 'x2+xy':
            return (x1 ** 2 + x1 * x2) % self.p
        elif self.op_type == 'x3+xy':
            return (x1 ** 3 + x1 * x2) % self.p
        elif self.op_type == 'div':
            for y in range(self.p):
                if (y * x2) % self.p == x1:
                    return y
            assert False
        elif self.op_type == 'zero':
            return 0
        elif self.op_type == 'rand':
            return torch.randint(0, self.p, size=[]).item()
        elif self.op_type == 'tricky':
            return 1 if x1 == 0 or x2 == 0 else 0
    
    def __getitem__(self, i):
        return self.x1[i], self.x2[i], self.target[i]
    
    def __len__(self):
        return self.data.shape[0]


def gen_mod_add(seed, frac, p, operation, device, noise_level, fixed_seed):
    
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    
    

    dataset = BinOpDataset(p, operation)
   
    data_perm = torch.randperm(len(dataset)).tolist()
    
    n_train = int(len(data_perm) * frac)
    n_noise = int(noise_level * n_train)
  
    Y = dataset.target
    
    if fixed_seed == True:
        torch.random.manual_seed(0)
        random_labels = torch.randint(0, p, (n_noise,))
        labels_noise = deepcopy(Y)
        labels_noise[:n_noise] = random_labels
    else:
        random_labels = torch.randint(0, p, (n_noise,))
        labels_noise = deepcopy(Y)
        
        # labels_noise[torch.randperm(train_size)[:n_noise]] = random_labels
        labels_noise[:n_noise] = deepcopy(random_labels)
        # count = 0
        # for i in range(n_noise):
        #     if random_labels[i] == Y[i]:
        #         count += 1
        #         labels_noise[i] = torch.randint(0, p, (1,)).item()
        # print(f'# of changed corrupted labels = {count}')
    
    if noise_level != 0:
        dataset.target = labels_noise
  
    train_indices = data_perm[:n_train]
    test_indices = data_perm[n_train:]
 
    train_inputs1, train_inputs2, train_targets = dataset[train_indices]
    
    
    
    train_inputs = F.one_hot(train_inputs1, p * 2) + F.one_hot(train_inputs2 + p, p * 2)
  
    train_inputs = train_inputs.to(torch.get_default_dtype())

    test_inputs1, test_inputs2, test_targets = dataset[test_indices]
    test_inputs = F.one_hot(test_inputs1, p * 2) + F.one_hot(test_inputs2 + p, p * 2)
    test_inputs = test_inputs.to(torch.get_default_dtype())

    torch.set_rng_state(old_state)
    
    return (train_inputs.to(device=device), train_targets.to(device=device)), (test_inputs.to(device=device), test_targets.to(device=device))