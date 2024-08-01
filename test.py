import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from import_tool import *
from torch import nn, Tensor

ta = torch.tensor([0.4980,0.4937,0.4856,0.5044,0.4749,0.4717,0.5088,0.5063])
tb = torch.tensor([0.4482,0.2050,0.7852,0.4915,0.1697,0.7993,0.0089,0.0840])
bce = nn.MSELoss()
print(bce(ta,tb))