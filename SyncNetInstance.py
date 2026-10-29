#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
from SyncNetModel import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();
        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).to(device)

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=device, weights_only=True)
        self.__S__.load_state_dict(loaded_state)
