import torch 
import torch.nn as nn
import numpy as np

from models.configuration import activation_dict

class EncoderDecoder(torch.nn.Module):
    def __init__(self, input_size, output_size, cfg_encoder,  cfg_decoder,
                 activation_function = "relu", 
                 init_method = "None", 
                 flatten_initial = True, 
                 dropout_strategy = None,
                 internal_rep_dim = 1,
                 device = "cuda"):
        super(EncoderDecoder, self).__init__()
        
        self.flatten_initial = flatten_initial
        self.input_size = input_size
        self.dropout_strategy = dropout_strategy
        self.flatten = nn.Flatten()
        #self.activation = torch.nn.ReLU()
        self.activation = activation_dict[activation_function]
        
    
        # ------ Create Encoder ------
        encoder_layers = []
        encoder_depth = len(cfg_encoder.keys())
        for i, (layer_name, layer) in enumerate(cfg_encoder.items()):
            width = layer.width
            if i == 0:
                encoder_layers.append(nn.Linear(input_size, width))
                encoder_layers.append(self.activation)
            elif i == encoder_depth - 1:
                encoder_layers.append(nn.Linear(previous_width, width))
            else:
                encoder_layers.append(nn.Linear(previous_width, width))
                encoder_layers.append(self.activation)
            previous_width = width
            
        self.encoder = nn.Sequential(*encoder_layers).to(device)
        
       
        # ------ Create Decoder ------
        decoder_layers = []
        decoder_depth = len(cfg_decoder.keys())
        for i, (layer_name, layer) in enumerate(cfg_decoder.items()):
            width = layer.width
            if i == 0:
                decoder_layers.append(nn.Linear(previous_width, width))
                decoder_layers.append(self.activation)
            elif i == decoder_depth - 1:
                decoder_layers.append(nn.Linear(previous_width, input_size))
            else:
                decoder_layers.append(nn.Linear(previous_width, width))
                decoder_layers.append(self.activation)
            previous_width = width
        self.decoder = nn.Sequential(*decoder_layers).to(device)
       
        self.cfg_encoder = cfg_encoder
        self.cfg_decoder = cfg_decoder
        """"
        if init_method == 'he':
            for i, (layer_name, layer) in enumerate(self.encoder.items()):
                if "output" in layer_name:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                else:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
 
                if layer.bias is not None:
                    layer.bias.data.zero_()
            for i, (layer_name, layer) in enumerate(self.decoder.items()):
                if "output" in layer_name:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='linear')
                else:
                    torch.nn.init.kaiming_normal_(layer.weight.data, nonlinearity='relu')
 
                if layer.bias is not None:
                    layer.bias.data.zero_()
                    
        elif init_method == "xavier":
            for name, param in self.encoder.named_parameters():
                if param.dim() > 1:
                    if self.input_size in param.shape:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                    else:
                        nn.init.xavier_normal_(param)
                else:
                    nn.init.constant_(param, 0)
        elif init_method == "sqrt_dim":
            for name, param in self.encoder.named_parameters():
                if param.dim() > 1:
                    if self.input_size in param.shape:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                    else:
                        nn.init.normal_(param, 0, 1 / np.sqrt(self.input_size))
                else:
                    nn.init.constant_(param, 0)        
        """
      
    def set_last_layer_zero_init(self):
        self.encoder[-1].weight.data.zero_()
        self.decoder[-1].weight.data.zero_()
   
    def forward(self, x, mask = None):
        features = {}

        
        x1 = x[..., :self.input_size]
        x2 = x[..., self.input_size:]
      
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        
        x = self.decoder(x1 + x2)
        
        features["encoder_x1"] = x1.clone()
        features["encoder_x2"] = x2.clone()
        features["decoder"] = x.clone()
    
        return features, x