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


text_file = "wonderland.txt"
json_file = "wonderland-encoder.json"

# https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

#create_char_encoder(text_file, json_file)


int2char, char2int = load_char_encoader(json_file)
vocabulary = create_vocabulary(text_file)



dataset = Text_Dataset(text_file, char2int, sequence_length=hyperparameters["sequence-length"])

net = torch.load("wonderland-net.pt")

#net = RNN(input_size=len(int2char))

#net = train_model(net, dataset)

torch.save(net, "wonderland-net.pt")

generate_text_raw(net, int2char, char2int)

#generate_text(net, int2char, char2int, vocabulary)