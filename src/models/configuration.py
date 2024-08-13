import torch 
import torch.nn as nn

optimizer_dict = {
    'adamw': torch.optim.AdamW,
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}

class nnPower(nn.Module):
    def __init__(self, power):
        super().__init__()
        self.power = power

    def forward(self, x):
        return torch.pow(x, self.power)

activation_dict = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'power': nnPower(2)
}

dropout_dict = {
    "dropout": lambda p: nn.Dropout(p=p),
    "alpha": lambda p: nn.AlphaDropout(p=p),
    "featurealpha": lambda p: nn.FeatureAlphaDropout(p=p)
}

norm_dict = {
    "bn": lambda p: nn.BatchNorm1d(p),
    "ln": lambda p: nn.LayerNorm(p)
}

class MyHingeLoss(torch.nn.Module):
    def __init__(self):
        super(MyHingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(torch.squeeze(output), torch.squeeze(target))
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss
    
loss_function_dict = {
    'mse': nn.MSELoss(),
    'crossentropy': nn.CrossEntropyLoss(),
    'hinge' : MyHingeLoss()
}