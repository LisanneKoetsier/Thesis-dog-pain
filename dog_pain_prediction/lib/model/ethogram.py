import torch.nn as nn
import torch
from .conv_LSTM import ConvLSTM
from . import model_utils as utils
import torch.nn.functional as F

class Ethogram_only(nn.Module):
    def __init__(self, cfg):
        super(Ethogram_only, self).__init__()
        self.model_type = cfg.MODEL.TYPE
        self.etho_dim = cfg.MODEL.ETHOGRAM_EMBEDDING_DIM
        self.etho_embed = nn.Linear(10, self.etho_dim)
        self._init_ethogram_weights()
    
    def _init_ethogram_weights(self):
        nn.init.xavier_uniform_(self.etho_embed.weight)
        nn.init.constant_(self.etho_embed.bias, 0)

    def forward(self, ethogram):
        ethogram = ethogram.float()
        ethogram_embedding = self.etho_embed(ethogram)
        return ethogram_embedding