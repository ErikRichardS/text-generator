import torch
import re
import random




def copy_memory(memory):
	return (
			torch.clone(memory[0]), 
			torch.clone(memory[1])
		)

def update_memory(net, memory, int2char, char2int, text):
	for i in range(len(text)-1):
		_, memory = generate_character(net, memory, int2char, char2int, text)

	return memory



def generate_text(net, int2char, char2int, vocabulary, text="V", paragraph_length=1000, paragraph_number=1):
	if len(text) < 1:
		raise ValueError("Starting text must have at least one character")
	if paragraph_length < 0 or paragraph_number < 0:
		raise ValueError("Cannot generate a negative number of words")

	# Update the memory with the starting text
	memory = net.init_state(sequence_length=1)
	update_memory(net, memory, int2char, char2int, text)


	memory_original = copy_memory(memory)
	text_full = ""
	for p in range(paragraph_number):
		for i in range(paragraph_length):
			character, memory = generate_character(net, memory, int2char, char2int, text)
			text = text + character

		text_full += spell_check_text(text, vocabulary)
		text = text_full
		memory = update_memory(net, memory_original, int2char, char2int, text)


	print(text_full)


def generate_text_raw(net, int2char, char2int, text="V", text_length=1000):
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



def spell_check_text(text, vocabulary):
	tokens = text.split(" ")

	regex_word = "[a-zA-Z]+(-[a-zA-Z]+)?('[a-zA-Z]+)?"

	corrected_text = ""

	for word in tokens:
		if word == "":
			continue

		elif word.lower() in vocabulary:
			corrected_text += " "+word
			continue
		
		match = re.search(regex_word, word)

		if match:
			# Check if there's more in the token than the word.
			# If so, extract them as prefixes and suffixes. 
			beg_idx = match.start(0)
			end_idx = match.end(0)
			prefix = ""
			suffix = ""
			if end_idx < len(word):
				suffix = word[end_idx:]
			if beg_idx > 0:
				prefix = word[:beg_idx]


			# Check if the first letter is a capital letter. 
			is_capital = word[0].isupper()

			# Spell check to find possible words that are correct.
			possible_words = spell_correct_word(match.group(0), vocabulary)

			# Select a correct word at random. 
			new_word = random.choice(possible_words)

			if is_capital:
				corrected_text += " "+prefix+new_word[0].upper()+new_word[1:]+suffix
			else:
				corrected_text += " "+prefix+new_word+suffix



	return corrected_text


def spell_correct_word(word_mispelled, vocabulary):
	word_mispelled = word_mispelled.lower()

	closest_words = []
	smallest_distance = 100

	distance_matrix = torch.zeros([len(word_mispelled),50])
	distance_matrix[:,0] = torch.tensor( range(len(word_mispelled)) )
	distance_matrix[0,:] = torch.tensor( range(50) )

	for word_correct in vocabulary:
		# If the words have no letters in common, 
		# or the length difference is smaller than 
		# the smallest distance so far,
		# skip
		length_difference = abs(len(word_mispelled) - len(word_correct))
		if length_difference > smallest_distance:
			continue

		distance = 0
		# Edge case if a word is one character long:
		if len(word_correct) == 1 or len(word_mispelled) == 1:
			if word_mispelled == word_correct:
				smallest_distance = 0
				closest_words = [word_correct]
			elif word_correct in word_mispelled or word_mispelled in word_correct:
				distance = length_difference

		else:
			# Build the distance matrix
			for i in range(1,len(word_mispelled)):
				for j in range(1,len(word_correct)):
					change = distance_matrix[i-1, j-1].item()
					if word_mispelled[i-1] != word_correct[j-1]:
						change += 1

					remove = distance_matrix[i-1, j].item() + 1
					add = distance_matrix[i, j-1].item() + 1

					distance_matrix[i,j] = min([change, remove, add])

				if distance_matrix[i, len(word_correct)-1] > smallest_distance:
					distance_matrix[-1, len(word_correct)-1] = 100
					break;

			# See if the distance is small enough
			distance = distance_matrix[-1, len(word_correct)-1].item()


		if distance < smallest_distance:
			closest_words = [word_correct]
			smallest_distance = distance

		elif distance == smallest_distance:
			closest_words.append(word_correct)



	return closest_words

	
