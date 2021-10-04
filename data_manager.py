import torch
import torch.utils.data
from torch.utils.data import DataLoader

import json
import re
import os


# Classes

class Text_Dataset(torch.utils.data.Dataset):
	def __init__(self, textfile, char2int, sequence_length):
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

def create_char_encoder():
	int2char = set()

	for file in os.listdir("Data"):
		if file.endswith(".txt"):
			filename = os.path.join("Data", file)

			file = open(filename, "r", encoding="utf-8")

			int2char = int2char.union( set(file.read()) ) 

	
	int2char = list(int2char)
	int2char.sort()
	int2char.insert(0, "<PAD>")

	char2int = {}

	for i, c in enumerate(int2char):
		char2int[c] = i

	return int2char, char2int


def create_vocabulary():
	vocabulary_clean = set()

	for file in os.listdir("Data"):
		if file.endswith(".txt"):
			filename = os.path.join("Data", file)
			file = open(filename, "r", encoding="utf-8")

			vocabulary_raw = set( re.split("[\\s\n]", file.read()) )
			
			regex_word = "[a-zA-Z]+(-[a-zA-Z]+)?('[a-zA-Z]+)?"

			for word in vocabulary_raw:
				match = re.search(regex_word, word)

				if match:
					correct_word = match.group(0)
					vocabulary_clean.add( correct_word.lower() )

	return vocabulary_clean


def save_meta_files(json_file):
	int2char, char2int = create_char_encoder()
	vocabulary = create_vocabulary()

	with open(json_file, "w") as write_file:
		json.dump( [int2char, char2int, list(vocabulary)], write_file )


def load_meta_files(json_file):
	with open(json_file, "r") as read_file:
		int2char, char2int, vocabulary = json.load(read_file)
	return int2char, char2int, set(vocabulary)


def get_datasets(char2int, sequence_length):

	dataset_list = []

	for file in os.listdir("Data"):
		if file.endswith(".txt"):
			dataset_list.append(
						Text_Dataset(os.path.join("Data", file), char2int, sequence_length)
					)


	return dataset_list
