import torch
import numpy as np


def mae(et_dm, gt_dm):
    return torch.abs(et_dm.sum() - gt_dm.sum())

def mse(et_dm, gt_dm):
    return (et_dm - gt_dm) * (et_dm - gt_dm)
 
def bce(et_dm,gt_dm):
   return torch.mean(-(0.7*gt_dm* torch.log(et_dm)+(1-gt_dm)*torch.log(1-et_dm)), dim=(1,2))