import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import json

from ann import RNN
from trainer import train_model
from data_manager import *
from generator import *
from settings import hyperparameters





# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
generator_name = "wonderland"

json_file = generator_name+"-encoder.json"
save_meta_files(json_file)
int2char, char2int, vocabulary = load_meta_files(json_file)


dataset_list = get_datasets(char2int, sequence_length=hyperparameters["sequence-length"])

"""
net = RNN(input_size=len(int2char))
net = train_model(net, dataset_list, start_fresh=True)
torch.save(net, generator_name+"-net.pt")
"""

net = torch.load(generator_name+"-net.pt")

generate_text(net, int2char, char2int, vocabulary)