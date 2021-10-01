import torch
import torch.utils.data
from torch.utils.data import DataLoader

import json
import re


# Classes

class Text_Dataset(torch.utils.data.Dataset):
	def __init__(self, textfile, char2int, sequence_length=10):
		self.input_dim = len(char2int)
		self.sequence_length = sequence_length

		text = open(textfile, "r", encoding="utf-8").read()

		self.char_idx_seq = torch.tensor(
										[char2int[c] for c in text]
										)

		self.length = len(text) - sequence_length

	def __getitem__(self, idx):
		data = torch.zeros(self.sequence_length+1, self.input_dim)
		for i in range(self.sequence_length+1):
			data[i, self.char_idx_seq[idx+i]] = 1

		return ( 
			self.char_idx_seq[idx:idx+self.sequence_length], 
			self.char_idx_seq[idx+1:idx+self.sequence_length+1]
			)
		

	def __len__(self):
		return self.length



# Functions

def create_char_encoder(filename):
	file = open(filename, "r", encoding="utf-8")

	int2char = list(set(file.read()))
	int2char.sort()
	int2char.insert(0, "<PAD>")

	char2int = {}

	for i, c in enumerate(int2char):
		char2int[c] = i

	#with open(json_file, 'w') as fout:
	#	json.dump( (int2char, char2int) , fout)

	return int2char, char2int


def load_char_encoader(json_file):
	with open(json_file, "r") as read_file:
		(int2char, char2int) = json.load(read_file)
	return int2char, char2int


def create_vocabulary(filename):
	file = open(filename, "r", encoding="utf-8")

	vocabulary_raw = set( re.split("[\\s\n]", file.read()) )
	vocabulary_clean = set()

	regex_word = "[a-zA-Z]+(-[a-zA-Z]+)?('[a-zA-Z]+)?"

	for word in vocabulary_raw:
		match = re.search(regex_word, word)

		if match:
			correct_word = match.group(0)
			vocabulary_clean.add( correct_word.lower() )

	return vocabulary_clean





