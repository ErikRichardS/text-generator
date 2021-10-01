import torch
import re
import random





def copy_memory(memory):
	return (
			torch.clone(memory[0]), 
			torch.clone(memory[1])
		)


def generate_text(net, int2char, char2int, vocabulary, text="Alice ", text_length=100):
	if len(text) < 1:
		raise ValueError("Starting text must have at least one character")
	if text_length < 0:
		raise ValueError("Cannot generate a negative number of words")

	# Update the memory with the starting text
	memory = net.init_state(sequence_length=1)
	for i in range(len(text)-1):
		_, memory = generate_character(net, memory, int2char, char2int, text)



	for i in range(text_length):
		while not text[-1].isalpha():
			character, memory = generate_character(net, memory, int2char, char2int, text)
			text = text + character

		text, memory = generate_word(net, memory, int2char, char2int, vocabulary, text)

	print(text)


def generate_text_raw(net, int2char, char2int, text="Alice ", text_length=1000):
	if len(text) < 1:
		raise ValueError("Starting text must have at least one character")
	if text_length < 0:
		raise ValueError("Cannot generate a negative number of words")

	# Update the memory with the starting text
	memory = net.init_state(sequence_length=1)
	for i in range(len(text)-1):
		_, memory = generate_character(net, memory, int2char, char2int, text)



	for i in range(text_length):
		character, memory = generate_character(net, memory, int2char, char2int, text)
		text = text + character

	print(text)


def generate_character(net, memory, int2char, char2int, text):
	# Create tensor of last character
	input_char = torch.tensor(char2int[text[-1]]).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
	c, memory = net(input_char, memory)
	memory = ( memory[0].detach(), memory[1].detach() )
	character = int2char[ torch.argmax(c.squeeze()) ]

	return character, memory


def generate_word(net, memory, int2char, char2int, vocabulary, text):
	# Generate a word.
	regex_word = "[a-zA-Z]+(-[a-zA-Z]+)?('[a-zA-Z]+)?$"

	original_memory = copy_memory(memory)

	word = text[-1]
	is_capital = word.isupper()
	while re.search(regex_word, word):
		character, memory = generate_character(net, memory, int2char, char2int, word)
		word = word + character

	# Check if the word is valid.
	if word[:-1] not in vocabulary:
		# Find the words closest to the mispelled word.
		words = spell_correct_word(word[:-1], vocabulary)
		word = random.choice(words) + word[-1]

		memory = copy_memory(original_memory)

		for i in range(1,len(word)):
			_, memory = generate_character(net, memory, int2char, char2int, word[i])

	if is_capital:
		word = word[0].upper() + word[1:]

	text = text[:-1] + word

	

	return text, memory




def spell_correct_word(word_mispelled, vocabulary):
	word_mispelled = word_mispelled.lower()

	closest_words = []
	smallest_distance = 100

	distance_matrix = torch.zeros([len(word_mispelled),50])
	distance_matrix[:,0] = torch.tensor( range(len(word_mispelled)) )
	distance_matrix[0,:] = torch.tensor( range(50) )

	for word_correct in vocabulary:
		# If the words have no letters in common, skip
		if not re.search("["+word_mispelled+"]", word_correct):
			continue

		# Build the distance matrix
		for i in range(1,len(word_mispelled)):
			for j in range(1,len(word_correct)):
				change = distance_matrix[i-1, j-1].item()
				if word_mispelled[i-1] != word_correct[j-1]:
					change += 1

				remove = distance_matrix[i-1, j].item() + 1
				add = distance_matrix[i, j-1].item() + 1

				distance_matrix[i,j] = min([change, remove, add])

		# See if the distance is small enough
		distance = distance_matrix[-1, len(word_correct)-1].item()
		if distance < smallest_distance:
			closest_words = [word_correct]
			smallest_distance = distance

		elif distance == smallest_distance:
			closest_words.append(word_correct)



	return closest_words
				

"""
  m e x t
n 0 1 2 3
e 1 1 2 3
x 2 2 1 2
t 3 3 2 1
"""