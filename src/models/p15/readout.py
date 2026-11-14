"""Three-task residual location and conditional absolute error scale."""
import torch
from torch import nn
class ResourceReadout(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.hidden=nn.Sequential(nn.Linear(features,8),nn.Tanh())
        self.head=nn.Linear(8,6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    def forward(self,x):
        location,log_scale=self.head(self.hidden(x)).chunk(2,dim=-1)
        return location, .08 + torch.nn.functional.softplus(log_scale)

