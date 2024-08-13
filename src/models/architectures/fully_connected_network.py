import torch 
import torch.nn as nn
import numpy as np

from models.configuration import activation_dict, dropout_dict, norm_dict

class FCNN(torch.nn.Module):
    def __init__(self, input_size, output_size, layers, activation_function = "relu", init_method = "None", flatten_initial = True):
        super(FCNN, self).__init__()
        
        self.flatten_initial = flatten_initial
        self.input_size = input_size

        self.flatten = nn.Flatten()
        #self.activation = torch.nn.ReLU()
        self.activation = activation_dict[activation_function]
       
        self.model = nn.ModuleDict()
        
        for i, (layer_name, layer_properties) in enumerate(layers.items()):
            if layer_name != "output":
                if i == 0:
                    self.model[layer_name] = nn.Linear(input_size, layer_properties.width, bias = layer_properties.bias)
                else:
                    self.model[layer_name] =  nn.Linear(previous_layer_size, layer_properties.width, bias = layer_properties.bias)
                previous_layer_size = layer_properties.width
        
        self.model["output"] = torch.nn.Linear(previous_layer_size, output_size, bias=False)
        self.layers = layers
  
        if init_method == 'he':
            for i, (layer_name, layer) in enumerate(self.model.items()):
                if "output" in layer_name:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                else:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
 
                if layer.bias is not None:
                    layer.bias.data.zero_()
        elif init_method == "xavier":
            for name, param in self.model.named_parameters():
                if param.dim() > 1:
                    if self.input_size in param.shape:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                    else:
                        nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)
        elif init_method == "sqrt_dim":
            for name, param in self.model.named_parameters():
                if param.dim() > 1:
                    if self.input_size in param.shape:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                    else:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                else:
                    nn.init.constant_(param, 0)        
      
    def set_last_layer_zero_init(self):
        self.model["output"].weight.data.zero_()
        
    def forward(self, x, mask = None):
        features = {}
       
        if self.flatten_initial:
            x = self.flatten(x)
            x = x.flatten(1)
        
        for i, (layer_name, layer) in enumerate(self.model.items()):
         
            if layer_name != "output":
                x = layer(x)
                
                layer_properties = self.layers[layer_name]
               
                if layer_properties.dropout.value != 0:
                    #x = nn.Dropout(p=layer_properties.dropout)(x).to(x.device)
                    
                    x = dropout_dict[layer_properties.dropout.strategy](p=layer_properties.dropout.value)(x).to(x.device)
                    features["layer%d_post_do" % (i + 1)] = x.clone()
           
                if layer_properties.norm:
                    norm = norm_dict[layer_properties.norm](p=x.shape[1]).to(x.device)
                   
                    x = norm(x).to(x.device)
                    features["layer%d_post_bn" % (i + 1)] = x.clone()
     
                x = self.activation(x)
                features["layer%d_post" % (i + 1)] = x.clone()
         
                if mask is not None:
                    x = x * mask[layer_name]

        x = self.model["output"](x)
        features["%s_post" % ("output")] = x.clone()
        
        layer_properties = self.layers[layer_name]
        if layer_properties.dropout.value != 0:
            #x = nn.Dropout(p=layer_properties.dropout)(x).to(x.device)
            x = dropout_dict[layer_properties.dropout.strategy](p=layer_properties.dropout.value)(x).to(x.device)
            features["output_post_do"] = x.clone()
       
        return features, x