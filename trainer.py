import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


from time import time
from os.path import isfile
from os import remove
from copy import deepcopy


from settings import hyperparameters


checkpoint_path = "checkpoint.pt"


def checkpoint_exists():
	return isfile( checkpoint_path )


def save_checkpoint(net, epoch, learning_rate, current_loss):
	torch.save({
				"model-state-dict" : net.state_dict(),
				"epoch" : epoch,
				"learning-rate" : learning_rate,
				"loss" : current_loss,
			}, checkpoint_path )


def load_checkpoint(net):
	checkpoint = torch.load( checkpoint_path )
	net.load_state_dict(checkpoint["model-state-dict"])
	num_epochs = checkpoint["epoch"]
	learning_rate = checkpoint["learning-rate"]
	loss = checkpoint["loss"]
	
	return net, num_epochs, learning_rate, loss


def delete_checkpoint():
	if isfile( checkpoint_path ):
		remove( checkpoint_path ) 



# Trains a NN model. 
# Returns 1 if it starting to overfit before all epochs are done
# Returns 0 otherwise
def train_model(net, dataset_list, start_fresh=False):

	# Hyper Parameters
	num_epochs = hyperparameters["number-epochs"]
	batch_size = hyperparameters["batch-size"]
	learning_rate = hyperparameters["learning-rate"]
	wt_decay = hyperparameters["weight-decay"]
	lr_decay = hyperparameters["learning-decay"]

	sequence_length = hyperparameters["sequence-length"]


	# Wrap the datasets in loaders that will handle the fetching of data and labels
	# batch_size - How many datapoints is loaded each fetch.
	# shuffle - Randomize the order loaded datapoints. 
	loader_list = []
	for dataset in dataset_list:
		loader_list.append(
				torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
			)


	# Loss calculates the error of the output
	# Optimizer does the backprop to adjust the weights of the NN
	criterion = nn.CrossEntropyLoss() 
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


	current_loss = float('inf')
	epoch_start = 0

	# Check if checkpoint of saved progress exists. 
	# If it does, load and continue from saved checkpoint. 
	if not start_fresh and checkpoint_exists():
		print("Loading progress from last successful epoch...", end="\r")
		net, epoch_start, learning_rate, current_loss = load_checkpoint(net)
		optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
		print("Progress loaded. Continuing training from last successful epoch.")
		print("Loss :\t %0.3f" % (current_loss))
		print("Learning rate :\t %f" % (learning_rate))
		print("Epochs left :\t %d" % (epoch_start))

	
	# Train the Model
	print("Begin training...")
	for epoch in range(epoch_start, num_epochs):
		t1 = time()

		loss_sum = 0

		for loader in loader_list:
			state_h, state_c = net.init_state(sequence_length)

			for i, (data, labels) in enumerate(loader):
				# Load data into GPU using cuda
				data = data.cuda()
				labels = labels.cuda()

				# Forward + Backward + Optimize
				outputs, (state_h, state_c) = net(data, (state_h, state_c))

				optimizer.zero_grad()
				loss = criterion(outputs.transpose(1, 2), labels)
				loss.backward()
				optimizer.step()

				loss_sum += loss

				# Cut off backward in time pass to prevent memory overload
				state_h = state_h.detach()
				state_c = state_c.detach()
				
				
		t2 = time()
			

		print("Epoch time : %0.3f m \t Loss : %0.3f" % ( (t2-t1)/60 , loss_sum ))


		# Reduce learning rate for next epoch
		learning_rate *= (1-lr_decay)
		optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #, weight_decay=wt_decay)
		print("New learning rate : %f" % (learning_rate))



		# Save training progress
		print("Saving training progress...", end="\r")
		save_checkpoint(
				net=net, 
				epoch=epoch, 
				learning_rate=learning_rate, 
				current_loss=current_loss 
			)
		print("Progress has been saved. Epoch %d of %d done." % (epoch+1, num_epochs))
		print()

		
	delete_checkpoint()
	return net













