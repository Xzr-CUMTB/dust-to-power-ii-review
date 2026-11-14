"""Ordered positive quantile increments anchored to the issue's weather-matched mature donor library."""
import torch
from torch import nn
class Tail(nn.Module):
 def __init__(self,n,k):
  super().__init__();self.time=nn.Sequential(nn.Linear(11,12),nn.SiLU(),nn.Linear(12,8))
  self.attn=nn.Linear(8,1);self.head=nn.Sequential(nn.Linear(n+8+k,24),nn.SiLU(),nn.Linear(24,k+8))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias);self.k=k
 def forward(self,x,w,anchor):
  s=self.time(w);a=(s*self.attn(s).softmax(1)).sum(1);v=self.head(torch.cat((x,a,anchor),1))
  inc=torch.diff(anchor,dim=1,prepend=torch.zeros_like(anchor[:,:1])).clamp_min(.012)
  q=(inc*torch.exp(1.5*torch.tanh(v[:,:self.k]))).cumsum(1)
  t=.75*torch.tanh(v[:,self.k:])
  return q,t

