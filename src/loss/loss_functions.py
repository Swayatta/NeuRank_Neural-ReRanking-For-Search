import torch.nn as nn
from torch.nn import MSELoss, BCEWithLogitsLoss

def mse_loss(logits, labels):
    loss = MSELoss()
    return loss(logits.squeeze(-1), labels.float())

def bce_loss(logits, labels):
    loss = BCEWithLogitsLoss()
    return loss(logits.squeeze(-1), labels.float())