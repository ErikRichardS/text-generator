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




json_file = network_name+"-encoder.json"
save_meta_files(json_file)
int2char, char2int, vocabulary = load_meta_files(json_file)


dataset_list = get_datasets(char2int, sequence_length=hyperparameters["sequence-length"])


train = None
while train == None:
	inp = input("Train a network? y/n\n>")
	if inp == "y":
		train = True
	elif inp == "n":
		train = False

if train:
	start_anew = None
	while start_anew == None:
		inp = input("Start fresh? y/n\n>")

		if inp == "y":
			start_anew = True
		elif inp == "n":
			start_anew = False

	net = RNN(num_layers=4, hidden_size=128, input_size=len(int2char))
	net = train_model(net, dataset_list, start_fresh=start_anew)
	torch.save(net, network_name+"-net.pt")



generate = None
while generate == None:
	inp = input("Generate text? y/n\n>")
	if inp == "y":
		generate = True
	elif inp == "n":
		generate = False


if generate:
	net = torch.load(network_name+"-net.pt")
	start_text = ""
	while len(start_text) < 1:
		start_text = input("please write a starting text at least one character long.\n>")

	generate_text(net, int2char, char2int, vocabulary, text=start_text)
