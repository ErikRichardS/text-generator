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


network_name = input("Please enter a network name:\n>")

start_anew = None
while start_anew == None:
	start_anew = input("Start fresh? y/n\n>")

	if start_anew == "y":
		start_anew = True
	elif start_anew == "n":
		start_anew = False
	else:
		start_anew = None


json_file = network_name+"-encoder.json"
save_meta_files(json_file)
int2char, char2int, vocabulary = load_meta_files(json_file)

dataset_list = get_datasets(char2int, sequence_length=hyperparameters["sequence-length"])


net = RNN(num_layers=4, hidden_size=128, input_size=len(int2char))
net = train_model(net, dataset_list, start_fresh=start_anew)
torch.save(net, network_name+"-net.pt")


net = torch.load(network_name+"-net.pt")


generate_text(net, int2char, char2int, vocabulary)
