import torch
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists


from train_network import Model


"""
This function is simply to evaluate model performance on various
datasets and values
"""

model = Model(38, 60, 27)
model_path = "train_semaphore.pt"

if exists(model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded model at ' + model_path)
    model.eval()
else:
    ValueError("No classifier model was found at {}".format(model_path))

